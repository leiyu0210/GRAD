import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math
import numpy as np
from typing import Tuple


def getScore_nips(reward, cpa, cpa_constraint):
    beta = 2
    penalty = 1
    if cpa > cpa_constraint:
        coef = cpa_constraint / (cpa + 1e-10)
        penalty = pow(coef, beta)
    return penalty * reward

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config['n_embd'] % config['n_head'] == 0
        self.key = nn.Linear(config['n_embd'], config['n_embd'])
        self.query = nn.Linear(config['n_embd'], config['n_embd'])
        self.value = nn.Linear(config['n_embd'], config['n_embd'])

        self.attn_drop = nn.Dropout(config['attn_pdrop'])
        self.resid_drop = nn.Dropout(config['resid_pdrop'])

        # 1*1*n_ctx*n_ctx
        self.register_buffer("bias",
                             torch.tril(torch.ones(config['n_ctx'], config['n_ctx'])).view(1, 1, config['n_ctx'],
                                                                                           config['n_ctx']))
        self.register_buffer("masked_bias", torch.tensor(-1e4))

        self.proj = nn.Linear(config['n_embd'], config['n_embd'])
        self.n_head = config['n_head']

    def forward(self, x, mask): 
        B, T, C = x.size() # T=seq*num_item, C=emb_dim

        # batch*n_head*T*C // self.n_head
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        mask = mask.view(B, -1)
        # batch*1*1*(seq*3)
        mask = mask[:, None, None, :]
        # 1->0, 0->-10000
        mask = (1.0 - mask) * -10000.0
        # batch*n_head*T*T
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = torch.where(self.bias[:, :, :T, :T].bool(), att, self.masked_bias.to(att.dtype))
        att = att + mask
        att = F.softmax(att, dim=-1)
        self._attn_map = att.clone()
        att = self.attn_drop(att)
        # batch*n_head*T*C // self.n_head
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        return y

class RegressionNet(nn.Module):
    def __init__(self, params=None):
        super(RegressionNet, self).__init__()
        if params is not None:
            self.params = torch.FloatTensor(params)

    def forward(self, period, time_ind, cate, cpa_ind, x_segment, x):
        val = self.params[period, time_ind, cate, cpa_ind, x_segment]
        return x * val[:, 0] + val[:, 1]

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config['n_embd'])
        self.ln2 = nn.LayerNorm(config['n_embd'])
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config['n_embd'], config['n_inner']),
            nn.GELU(),
            nn.Linear(config['n_inner'], config['n_embd']),
            nn.Dropout(config['resid_pdrop']),
        )

    def forward(self, inputs_embeds, attention_mask): # batch*(seq*3)*dim, batch*(seq*3)
        x = inputs_embeds + self.attn(self.ln1(inputs_embeds), attention_mask)
        x = x + self.mlp(self.ln2(x))
        return x

class SwitchRouter(nn.Module):
    def __init__(self, hidden_size, num_experts):
        super().__init__()
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.num_experts = num_experts
        
    def forward(self, x):
        logits = self.gate(x)  # [batch, seq_len, num_experts]
        logits = logits - logits.max(dim=-1, keepdim=True).values
        expert_weights = F.softmax(logits, dim=-1)
        expert_weights, expert_idx = expert_weights.max(dim=-1)  #
        return expert_weights, expert_idx


class SwitchMoE(nn.Module):
    def __init__(self, hidden_size, num_experts, expert_capacity_factor=1.1):
        super().__init__()
        self.num_experts = num_experts
        self.router = SwitchRouter(hidden_size, num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, hidden_size * 4),
                nn.GELU(),
                nn.Linear(hidden_size * 4, hidden_size)
            ) for _ in range(num_experts)
        ])
        self.expert_capacity_factor = expert_capacity_factor
        self.load_balancing_loss = 0.0

    def forward(self, x):
        batch_size, seq_len, hidden_size = x.shape
        expert_weights, expert_idx = self.router(x)  # [batch, seq_len]
        
        tokens_per_expert = seq_len * batch_size / self.num_experts
        expert_capacity = int(tokens_per_expert * self.expert_capacity_factor)
        
        output = torch.zeros_like(x)
        total_tokens = 0
        expert_counts = torch.zeros(self.num_experts, device=x.device)
        
        for expert_id in range(self.num_experts):
            token_mask = (expert_idx == expert_id)
            num_tokens = token_mask.sum().item()
            expert_counts[expert_id] = num_tokens
            
            if num_tokens > 0:
                if num_tokens > expert_capacity:
                    
                    selected_indices = torch.multinomial(
                        token_mask.float(), expert_capacity, replacement=False
                    )
                    token_mask = torch.zeros_like(token_mask)
                    token_mask[selected_indices] = True
                    num_tokens = expert_capacity
                
                expert_input = x[token_mask]
                expert_output = self.experts[expert_id](expert_input)
                output[token_mask] = expert_output
                total_tokens += num_tokens
        
        uniform_load = torch.ones_like(expert_counts) * tokens_per_expert
        self.load_balancing_loss = 1.0 - F.cosine_similarity(
            expert_counts.unsqueeze(0), uniform_load.unsqueeze(0), dim=-1
        ) * 0.01  

        return output, self.load_balancing_loss

class ActionMoE(nn.Module):
    def __init__(self, hidden_size, act_dim, num_experts, reduction_factor=4):
        """
        ActionMoE with shared expert and specialized experts
        Args:
            hidden_size: Input feature dimension
            act_dim: Action dimension
            num_experts: Number of specialized experts (excluding shared expert)
            reduction_factor: Dimension reduction factor for experts
        """
        super().__init__()
        self.num_experts = num_experts
        self.act_dim = act_dim
        self.expert_dim = max(64, hidden_size // reduction_factor)  # Ensure minimum dimension
        
        # Shared expert (persistent pathway)
        self.shared_expert = nn.Sequential(
            nn.Linear(hidden_size, self.expert_dim),
            nn.GELU(),
            nn.Linear(self.expert_dim, act_dim)
        )
        
        # Specialized experts (conditional pathway)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, self.expert_dim),
                nn.GELU(),
                nn.Linear(self.expert_dim, act_dim)
            ) for _ in range(num_experts)
        ])
        
        # Routing network (top-1 selection)
        self.router = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.GELU(),
            nn.Linear(128, num_experts)
        )
        
        # MLP for residual transformation
        self.residual_mlp = nn.Sequential(
            nn.Linear(act_dim, 128),
            nn.GELU(),
            nn.Linear(128, act_dim)
        )

        self.value_estimator = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state_rep, base_action, attention_mask=None):
        """
        state_rep: State representation [batch, seq, hidden_size]
        base_action: Base action prediction [batch, seq, act_dim]
        attention_mask: Optional attention mask [batch, seq]
        Returns:
          final_action: Final action output [batch, seq, act_dim]
          selection_probs: Expert selection probabilities [batch, seq, num_experts]
          candidate_actions: Candidate actions [batch, seq, num_experts, act_dim]
        """
        batch_size, seq_len, _ = state_rep.shape
        return_rtg = torch.sigmoid(self.value_estimator(state_rep))  # [batch, seq, 1]
        # 1. Shared expert processing (always active)
        shared_output = self.shared_expert(state_rep)  # [batch, seq, act_dim]
        
        # 2. Generate candidate actions from base action
        scaling_factors = torch.linspace(0.8, 1.2, self.num_experts, 
                                       device=state_rep.device).view(1, 1, self.num_experts, 1)
        candidate_actions = base_action.unsqueeze(2) * scaling_factors  # [batch, seq, num_experts, act_dim]
        
        # 3. Expert routing
        router_logits = self.router(state_rep)  # [batch, seq, num_experts]
        selection_probs = F.softmax(router_logits, dim=-1)
        
        # 4. Process through specialized experts in parallel
        # Flatten sequence dimension for batch processing
        flat_state = state_rep.view(-1, state_rep.size(-1))  # [batch*seq, hidden_size]
        
        # Process all experts in parallel
        expert_outputs = []
        for expert in self.experts:
            expert_out = expert(flat_state)  # [batch*seq, act_dim]
            expert_outputs.append(expert_out.unsqueeze(1))  # [batch*seq, 1, act_dim]
        
        # Stack and reshape expert outputs
        expert_outputs = torch.cat(expert_outputs, dim=1)  # [batch*seq, num_experts, act_dim]
        expert_outputs = expert_outputs.view(batch_size, seq_len, self.num_experts, self.act_dim)  # [batch, seq, num_experts, act_dim]
        
        # 5. Combine shared and routed outputs
        if self.training:
            # Weighted combination during training
            weighted_expert = torch.sum(
                selection_probs.unsqueeze(-1) * expert_outputs, 
                dim=2
            )  # [batch, seq, act_dim]
        else:
            # Inference: select top-1 expert
            max_idx = torch.argmax(selection_probs, dim=-1)  # [batch, seq]
            
            # Create index tensor for gathering
            batch_idx = torch.arange(batch_size).view(-1, 1, 1).expand(-1, seq_len, 1).to(max_idx.device)
            seq_idx = torch.arange(seq_len).view(1, -1, 1).expand(batch_size, -1, 1).to(max_idx.device)
            
            # Gather the selected expert outputs
            weighted_expert = expert_outputs[batch_idx, seq_idx, max_idx.unsqueeze(-1)].squeeze(2)  # [batch, seq, act_dim]
        
        # 6. Combine shared and routed outputs
        moe_output = shared_output + weighted_expert
        
        # 7. Generate residual vector
        residual = self.residual_mlp(moe_output)  # [batch, seq, act_dim]
        
        # 8. Apply residual connection to get final action
        final_action = weighted_expert + residual  # [batch, seq, act_dim]
        
        # 9. Apply attention mask if provided
        if attention_mask is not None:
            # Expand mask to match final_action dimensions
            mask_expanded = attention_mask.unsqueeze(-1)  # [batch, seq, 1]
            mask_expanded = mask_expanded.expand(-1, -1, self.act_dim)
            
            # Apply mask
            final_action = final_action * mask_expanded
        
        return final_action, candidate_actions, selection_probs, return_rtg
        

class GRAD(nn.Module):

    def __init__(self, state_dim, act_dim, state_mean, state_std, hidden_size=512, action_tanh=False, K=10,
                 max_ep_len=48, scale=2000,
                 target_return=1, target_ctg = 1., device="cpu",
                 baseline_method = 'grad',
                 reweight_w = 0.2,
                 learning_rate=1e-5
                 ):
        super(GRAD, self).__init__()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.num_experts = 6 
        self.top_k = 1 
        self.explore_weight = 0.2
        self.diversity_weight=0.01
        self.explore_noise_std = 0.1
        self.usage_reward_coeff=0.05
        self.length_times = 3
        self.reweight_w = reweight_w
        self.baseline_method = baseline_method
        self.hidden_size = 512
        self.state_mean = state_mean
        self.state_std = state_std
        self.max_length = K
        self.max_ep_len = max_ep_len

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.scale = scale
        self.target_return = target_return
        self.target_ctg = target_ctg
        self.global_step = 0
        self.warmup_steps = 10000
        self.weight_decay = 0.0001
        self.learning_rate = learning_rate
        self.time_dim = 8
        self.block_config = {
            "n_ctx": 1024,
            "n_embd": self.hidden_size ,  # 512
            "n_layer": 10,
            "n_head": 16,
            "n_inner": 512,
            "activation_function": "relu",
            "n_position": 1024,
            "resid_pdrop": 0.1,
            "attn_pdrop": 0.1
        }
        block_config = self.block_config
        
        self.hyperparameters = {
            "n_ctx": self.block_config['n_ctx'],
            "n_embd": self.block_config['n_embd'],
            "n_layer": self.block_config['n_layer'],
            "n_head": self.block_config['n_head'],
            "n_inner": self.block_config['n_inner'],
            "activation_function": self.block_config['activation_function'],
            "n_position": self.block_config['n_position'],
            "resid_pdrop": self.block_config['resid_pdrop'],
            "attn_pdrop": self.block_config['attn_pdrop'],
            "length_times": self.length_times,
            "hidden_size": self.hidden_size,
            "state_mean": self.state_mean,
            "state_std": self.state_std,
            "max_length": self.max_length,
            "K": K,
            "state_dim": state_dim,
            "act_dim": act_dim,
            "scale": scale,
            "target_return": target_return,
            "warmup_steps": self.warmup_steps,
            "weight_decay": self.weight_decay,
            "learning_rate": self.learning_rate,
            "time_dim":self.time_dim

        }

        self.act_dim = act_dim        
    
        self.action_moe = ActionMoE(
            hidden_size=self.hidden_size,
            act_dim=self.act_dim,
            num_experts=self.num_experts
        )
        
        

        # n_layer of Block
        self.transformer = nn.ModuleList([Block(block_config) for _ in range(block_config['n_layer'])])

        self.embed_timestep = nn.Embedding(self.max_ep_len, self.time_dim)
        self.embed_return = torch.nn.Linear(1, self.hidden_size)
        self.embed_reward = torch.nn.Linear(1, self.hidden_size)
        self.embed_state = torch.nn.Linear(self.state_dim, self.hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, self.hidden_size)
        self.embed_ctg = torch.nn.Linear(1, self.hidden_size)

        self.trans_return = torch.nn.Linear(self.time_dim+self.hidden_size, self.hidden_size)
        self.trans_reward = torch.nn.Linear(self.time_dim+self.hidden_size, self.hidden_size)
        self.trans_state = torch.nn.Linear(self.time_dim+self.hidden_size, self.hidden_size)
        self.trans_action = torch.nn.Linear(self.time_dim+self.hidden_size, self.hidden_size)
        self.trans_cost = torch.nn.Linear(self.time_dim+self.hidden_size, self.hidden_size)
        self.trans_ctg = torch.nn.Linear(self.time_dim+self.hidden_size, self.hidden_size)

        self.embed_ln = nn.LayerNorm(self.hidden_size)
        self.predict_state = torch.nn.Linear(self.hidden_size, self.state_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(self.hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
        )
        
        self.predict_return = nn.Sequential(
            nn.Linear(self.hidden_size, 128),
            nn.GELU(),
            nn.Linear(128, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )

        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,
                                                           lambda steps: min((steps + 1) / self.warmup_steps, 1))

        self.init_eval()

    def forward(self, states, actions, rewards, returns_to_go, ctg, score_to_go, timesteps, attention_mask=None):
        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)

        returns_to_go  = returns_to_go + self.reweight_w * ctg
        rtg_embeddings = self.embed_return(returns_to_go)
        rewards_embeddings = self.embed_reward(rewards)
        time_embeddings = self.embed_timestep(timesteps)

        # To achieve a stable and good dt baseline, we use concat instead of common add, to let the model be aware of time
        state_embeddings = torch.cat((state_embeddings, time_embeddings), dim=-1)
        action_embeddings = torch.cat((action_embeddings, time_embeddings), dim=-1)
        rtg_embeddings = torch.cat((rtg_embeddings, time_embeddings), dim=-1)
        rewards_embeddings = torch.cat((rewards_embeddings, time_embeddings), dim=-1)

        state_embeddings = self.trans_state(state_embeddings)
        action_embeddings = self.trans_action(action_embeddings)
        rtg_embeddings = self.trans_return(rtg_embeddings)
        rewards_embeddings = self.trans_reward(rewards_embeddings)
        # batch*self.length_times*seq*dim->batch*(seq*self.length_times)*dim
        stacked_inputs = torch.stack(
            (rtg_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, self.length_times * seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        # batch*(seq_len * self.length_times)*embedd_size
        stacked_attention_mask = torch.stack(
            ([attention_mask for _ in range(self.length_times)]), dim=1
        ).permute(0, 2, 1).reshape(batch_size, self.length_times * seq_length).to(stacked_inputs.dtype)

        x = stacked_inputs
        for block in self.transformer:
            x = block(x, stacked_attention_mask)

        # batch*3*seq*dim
        x = x.reshape(batch_size, seq_length, self.length_times, self.hidden_size).permute(0, 2, 1, 3)  
        # predict the action based on the state embedding part 
        action_preds = self.predict_action(x[:, -2])

        return_preds = self.predict_return(x[:, 2])

        action_moe, candidate_actions, selection_probs, return_rtg = self.action_moe(
                x[:, -2], action_preds
            )
        
        if self.training:
            
            return x[:, -2], action_preds, action_moe, return_rtg, selection_probs

        return x, action_preds, None, None

    def get_action(self, states, actions, rewards, returns_to_go, ctg, score_to_go, timesteps, **kwargs):
        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        rewards = rewards.reshape(1, -1, 1)
        ctg = ctg.reshape(1, -1, 1)
        score_to_go = score_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        if self.max_length is not None:
            states = states[:, -self.max_length:]
            actions = actions[:, -self.max_length:]
            returns_to_go = returns_to_go[:, -self.max_length:]
            rewards = rewards[:, -self.max_length:]
            timesteps = timesteps[:, -self.max_length:]
            ctg = ctg[:, -self.max_length:]
            score_to_go = score_to_go[:, -self.max_length:]

            attention_mask = torch.cat([torch.zeros(self.max_length - states.shape[1]), torch.ones(states.shape[1])])
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)
            states = torch.cat(
                [torch.zeros((states.shape[0], self.max_length - states.shape[1], self.state_dim),
                             device=states.device), states],
                dim=1).to(dtype=torch.float32)
            actions = torch.cat(
                [torch.zeros((actions.shape[0], self.max_length - actions.shape[1], self.act_dim),
                             device=actions.device), actions],
                dim=1).to(dtype=torch.float32)
            returns_to_go = torch.cat(
                [torch.zeros((returns_to_go.shape[0], self.max_length - returns_to_go.shape[1], 1),
                             device=returns_to_go.device), returns_to_go],
                dim=1).to(dtype=torch.float32)
            rewards = torch.cat(
                [torch.zeros((rewards.shape[0], self.max_length - rewards.shape[1], 1), device=rewards.device),
                 rewards],
                dim=1).to(dtype=torch.float32)
            ctg = torch.cat(
                [torch.zeros((ctg.shape[0], self.max_length - ctg.shape[1], 1),
                             device=ctg.device), ctg],
                dim=1).to(dtype=torch.float32)
            score_to_go = torch.cat(
                [torch.zeros((score_to_go.shape[0], self.max_length - score_to_go.shape[1], 1),
                             device=score_to_go.device), score_to_go],
                dim=1).to(dtype=torch.float32)
            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], self.max_length - timesteps.shape[1]), device=timesteps.device),
                 timesteps],
                dim=1).to(dtype=torch.long)
        else:
            attention_mask = None
        if self.training:
            x, action_preds, action_moe, _, _ = self.forward(
                states=states, actions=actions, rewards=rewards, returns_to_go=returns_to_go, ctg=ctg, score_to_go=score_to_go, timesteps=timesteps, attention_mask=attention_mask)
        else:
            x, action_preds, _, _ = self.forward(
                states=states, actions=actions, rewards=rewards, returns_to_go=returns_to_go, ctg=ctg, score_to_go=score_to_go, timesteps=timesteps, attention_mask=attention_mask)
        return x, action_preds[0, -1]   # x is the embedding
    
   
    def step(self, states, actions, rewards, dones, rtg, timesteps, attention_mask, ctg, score_to_go, costs):
        self.global_step += 1
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        costs = costs.to(self.device)   # cost is cost(s_t, a_t), which is every single step's true cost
        dones = dones.to(self.device)
        rtg = rtg.to(self.device)
        timesteps = timesteps.to(self.device)
        attention_mask = attention_mask.to(self.device)
        ctg = ctg.to(self.device)
        score_to_go = score_to_go.to(self.device)

        rewards_target, action_target, rtg_target, costs_target = torch.clone(rewards), torch.clone(actions), torch.clone(rtg), torch.clone(costs)

        state_rep, action_preds, action_moe, return_rtg, selection_probs = self.forward(
            states=states, actions=actions, rewards=rewards, returns_to_go=rtg[:, :-1], ctg=ctg[:,:-1], score_to_go=score_to_go[:, :-1], timesteps=timesteps, attention_mask=attention_mask,
        )
        
        # batch*seq*action_dim
        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_moe = action_moe.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        return_rtg = return_rtg.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        selection_probs = selection_probs.reshape(-1, self.num_experts)[attention_mask.reshape(-1) > 0]
        
        balance_loss, anchor_loss = self.compute_moe_losses(action_moe, state_rep, selection_probs)
        
        loss = self.compute_loss(
            action_preds=action_preds,
            target_actions=action_target,  
            action_moe=action_moe,
            moe_lambda = return_rtg
        )
        loss = loss + balance_loss + anchor_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), .25)
        self.optimizer.step()

        return loss.detach().cpu().item()


    def compute_moe_losses(
        self,
        final_action: torch.Tensor, 
        state_rep: torch.Tensor,
        selection_probs: torch.Tensor,
        w_aux: float = 0.1,
        lambda_shared: float = 0.1,
        shared_expert: nn.Module = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute MoE balancing loss and shared expert anchoring loss
        
        Args:
            final_action: Final action output [batch, seq, act_dim]
            state_rep: State representation [batch, seq, hidden_size]
            selection_probs: Expert selection probabilities [batch, seq, num_experts]
            w_aux: Auxiliary loss weight for router balancing
            lambda_shared: Weight for shared expert anchoring loss
            shared_expert: Shared expert module (for anchoring loss)
        
        Returns:
            balance_loss: MoE balancing loss scalar
            anchor_loss: Shared expert anchoring loss scalar
        """
        # 1. Compute MoE balancing loss
        batch_size, seq_len, num_experts = selection_probs.shape
        
        # Get expert selection indices
        expert_idx = torch.argmax(selection_probs, dim=-1)  # [batch, seq]
        
        # Create one-hot representation of expert selection
        expert_mask = F.one_hot(expert_idx, num_classes=num_experts)  # [batch, seq, num_experts]
        
        # Calculate expert utilization
        expert_utilization = expert_mask.float().mean(dim=(0, 1))  # [num_experts]
        
        # Calculate routing probability mean
        routing_prob_mean = selection_probs.mean(dim=(0, 1))  # [num_experts]
        
        # Avoid log(0) issues
        eps = 1e-8
        routing_prob_mean = routing_prob_mean.clamp(min=eps)
        expert_utilization = expert_utilization.clamp(min=eps)
        
        # Calculate symmetric cross entropy
        ce_p_u = -torch.sum(expert_utilization * torch.log(routing_prob_mean))
        ce_u_p = -torch.sum(routing_prob_mean * torch.log(expert_utilization))
        
        balance_loss = w_aux * (ce_p_u + ce_u_p)
        
        # 2. Compute shared expert anchoring loss
        if shared_expert is not None:
            # Get intermediate representation from shared expert
            # Assuming shared_expert is nn.Sequential with at least one layer
            if len(shared_expert) >= 2:
                # Pass through first layer and activation
                intermediate_rep = shared_expert[0](state_rep)  # [batch, seq, expert_dim]
                intermediate_rep = shared_expert[1](intermediate_rep)  # after activation
                
                # Calculate MSE between state_rep and intermediate representation
                # Project state_rep to same dimension for comparison
                if state_rep.size(-1) != intermediate_rep.size(-1):
                    proj = nn.Linear(state_rep.size(-1), intermediate_rep.size(-1)).to(state_rep.device)
                    state_proj = proj(state_rep)
                    anchor_loss = lambda_shared * F.mse_loss(state_proj, intermediate_rep)
                else:
                    anchor_loss = lambda_shared * F.mse_loss(state_rep, intermediate_rep)
            else:
                # Fallback to output comparison if no intermediate layer
                shared_output = shared_expert(state_rep)
                anchor_loss = lambda_shared * F.mse_loss(state_rep[:, :, :shared_output.size(-1)], shared_output)
        else:
            anchor_loss = torch.tensor(0.0, device=final_action.device)
        
        return balance_loss, anchor_loss
        
    def compute_loss(self, 
                 action_preds: torch.Tensor, 
                     target_actions: torch.Tensor,
                     action_moe: torch.Tensor = None,
                     moe_lambda: torch.Tensor = None) -> torch.Tensor:
    
        base_loss = F.mse_loss(action_preds, target_actions, reduction='none') 
        
        if action_moe is not None and moe_lambda is not None:
            moe_loss = F.mse_loss(action_moe, target_actions, reduction='none')  
            
            # entropy = -(moe_lambda * moe_lambda.log()).sum(dim=-1, keepdim=True)
            # moe_lambda = 1 - entropy / np.log(moe_lambda.shape[-1]) 
            
            moe_lambda = moe_lambda.expand_as(base_loss) 
            
            moe_lambda = torch.nn.Parameter(torch.tensor(0.5), requires_grad=True)
       
            # moe_lambda = moe_loss.detach() / (base_loss.detach() + moe_loss.detach() + 1e-8)

            weighted_loss = (1 - moe_lambda) * base_loss + moe_lambda * moe_loss
       
            total_loss = weighted_loss.mean()  
        else:
            total_loss = base_loss.mean()
        
        return total_loss


    
    def take_actions(self, state, actual_excuted_action, target_return=None, target_ctg=None, pre_reward=None, pre_cost=None, cpa_constrain=None):
        self.eval()
        if self.eval_states is None:
            self.eval_states = torch.from_numpy(state).reshape(1, self.state_dim).to(self.device)
            ep_return = target_return.to(self.device) if target_return is not None else self.target_return
            self.eval_target_return = torch.tensor(ep_return, dtype=torch.float32).reshape(1, 1).to(self.device)
            self.eval_target_score_to_go = torch.tensor(ep_return, dtype=torch.float32).reshape(1,1).to(self.device)

            ep_ctg = target_ctg.to(self.device) if target_ctg is not None else self.target_ctg
            self.eval_target_ctg = torch.tensor(ep_ctg, dtype=torch.float32).reshape(1, 1).to(self.device)
        else:
            assert pre_reward is not None
            assert pre_cost is not None
            cur_state = torch.from_numpy(state).reshape(1, self.state_dim).to(self.device)
            self.eval_states = torch.cat([self.eval_states, cur_state], dim=0).to(self.device)
            
            self.eval_rewards[-1] = pre_reward
            self.eval_costs[-1] = pre_cost

            # Implementing different methods' condition to go
            pred_return = self.eval_target_return[0, -1] - (pre_reward / self.scale)
            self.eval_target_return = torch.cat([self.eval_target_return, pred_return.reshape(1, 1)], dim=1)

            # pred_ctg = self.eval_target_ctg[0, -1] - (pre_cost/ self.scale)
            pred_ctg = torch.ones_like(self.eval_target_ctg[0, -1]) # ctg is always set as 1 in the inference stage
            self.eval_target_ctg = torch.cat([self.eval_target_ctg, pred_ctg.reshape(1, 1)], dim=1)

            self.eval_timesteps = torch.cat(
                [self.eval_timesteps, torch.ones((1, 1), dtype=torch.long).to(self.device) * self.eval_timesteps[:, -1] + 1], dim=1)

        # If actual_executed_action has a value, the action actually executed should replace the placeholder action from the previous moment.
        if actual_excuted_action is None:
            self.eval_actions = torch.cat([self.eval_actions, torch.zeros(1, self.act_dim).to(self.device)], dim=0)
        else:
            self.eval_actions[-1] = torch.from_numpy(actual_excuted_action).reshape(1, self.act_dim).to(self.device)
            self.eval_actions = torch.cat([self.eval_actions, torch.zeros(1, self.act_dim).to(self.device)], dim=0)
        
        self.eval_rewards = torch.cat([self.eval_rewards, torch.zeros(1).to(self.device)])
        self.eval_costs = torch.cat([self.eval_costs, torch.zeros(1).to(self.device)])

        # states, actions, rewards, returns_to_go, ctg, score_to_go, timesteps
        x, action = self.get_action(
            (self.eval_states.to(dtype=torch.float32) - torch.tensor(self.state_mean).to(self.device)) / torch.tensor(self.state_std).to(self.device),
            self.eval_actions.to(dtype=torch.float32),
            self.eval_rewards.to(dtype=torch.float32),
            self.eval_target_return.to(dtype=torch.float32),
            self.eval_target_ctg.to(dtype=torch.float32),
            self.eval_target_score_to_go.to(dtype=torch.float32),
            self.eval_timesteps.to(dtype=torch.long),
        )
        self.eval_actions[-1] = action
        action = action.detach().cpu().numpy()
        return action

    def init_eval(self):
        self.eval_states = None
        self.eval_actions = torch.zeros((0, self.act_dim), dtype=torch.float32).to(self.device)
        self.eval_rewards = torch.zeros(0, dtype=torch.float32).to(self.device)
        self.eval_costs = torch.zeros(0, dtype=torch.float32).to(self.device)

        self.eval_target_return = None
        self.eval_target_ctg = None

        self.eval_timesteps = torch.tensor(0, dtype=torch.long).reshape(1, 1).to(self.device)

        self.eval_episode_return, self.eval_episode_length = 0, 0

    def save_net(self, save_path, name):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        file_path = os.path.join(save_path, name)
        torch.save(self.state_dict(), file_path)


    def save_jit(self, save_path):
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        jit_model = torch.jit.script(self.cpu())
        torch.jit.save(jit_model, f'{save_path}/grad.pth')

    def load_net(self, load_path="saved_model/grad/grad.pt", device='cpu'):
        file_path = load_path
        self.load_state_dict(torch.load(file_path, map_location=device), strict=False)

