import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from .modeling_pi0 import PI0Policy, PI0FlowMatching
from .paligemma_with_expert import PaliGemmaWithExpertModel
from .utils import create_sinusoidal_pos_embedding, make_att_2d_masks


class PI0PolicyWithGlobalHead(PI0Policy):
    """PI0 + 全局轨迹预测头架构"""
    
    def __init__(self, config, tokenizer_path: str = "google/paligemma-3b-pt-224"):
        super().__init__(config, tokenizer_path)
        # 替换原来的model为新的带全局头的版本
        paligemma_with_expert_config = self._create_paligemma_config()
        self.model = PI0FlowMatchingWithGlobalHead(config, paligemma_with_expert_config)
        self.reset()
    
    def _create_paligemma_config(self):
        """创建PaliGemma配置（复用原有逻辑）"""
        from .paligemma_with_expert import PaliGemmaWithExpertConfig
        return PaliGemmaWithExpertConfig(
            freeze_vision_encoder=self.config.freeze_vision_encoder,
            train_expert_only=self.config.train_expert_only,
            attention_implementation=self.config.attention_implementation,
            use_spatial_encoder=True,
            spatial_tower="cut3r",
            spatial_tower_select_feature="all",
            spatial_camera_config={
                "base_0_rgb": True,
                "left_wrist_0_rgb": False, 
                "right_wrist_0_rgb": False,
            },
            use_history_features=True,
            num_sampled_history_frames=5,
            history_sampling_method="uniform",
            history_camera_config={
                "base_0_rgb": True,
                "left_wrist_0_rgb": False,
                "right_wrist_0_rgb": False,
            },
            mm_projector_type="mlp2x_gelu",
            mm_hidden_size=768,
            fusion_block="cross_attention",
        )


class PI0FlowMatchingWithGlobalHead(PI0FlowMatching):
    """PI0 Flow Matching + 全局轨迹头"""
    
    def __init__(self, config, paligemma_with_expert_config):
        super().__init__(config, paligemma_with_expert_config)
        
        # 🔥 新增：全局轨迹预测头（与action expert完全同构）
        self.global_trajectory_expert = self._create_global_expert()
        
        # 🔥 修改原有action expert的输入投影（接收全局轨迹而非noise）
        self.global_trajectory_in_proj = nn.Linear(
            self.config.max_action_dim, self.config.proj_width
        )
        
        self.set_requires_grad()
    
    def _create_global_expert(self):
        """创建与action expert完全同构的全局轨迹头"""
        # 复用gemma_expert的配置创建新的expert
        from transformers import GemmaForCausalLM
        global_expert = GemmaForCausalLM(self.config.gemma_expert_config)
        # 移除embed_tokens（与原action expert保持一致）
        global_expert.model.embed_tokens = None
        return global_expert
    
    def set_requires_grad(self):
        """设置参数训练状态"""
        super().set_requires_grad()
        # 全局轨迹头始终可训练
        for params in self.global_trajectory_expert.parameters():
            params.requires_grad = True
        for params in self.global_trajectory_in_proj.parameters():
            params.requires_grad = True
    
    def embed_global_trajectory_suffix(self, state, noisy_actions, timestep):
        """为全局轨迹头嵌入后缀信息（复用原有逻辑）"""
        return self.embed_suffix(state, noisy_actions, timestep)
    
    def forward_global_trajectory(self, prefix_embs, prefix_pad_masks, prefix_att_masks,
                                  state, noisy_actions, timestep):
        """全局轨迹头前向传播"""
        
        # 1. 嵌入全局轨迹后缀
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_global_trajectory_suffix(
            state, noisy_actions, timestep
        )
        
        # 2. 拼接前缀和后缀
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        
        # 3. 全局轨迹expert前向传播（双向注意到paligemma主干）
        # 创建包含paligemma和global expert的inputs_embeds
        inputs_embeds = [prefix_embs, suffix_embs]
        models = [self.paligemma_with_expert.paligemma.language_model.model, 
                  self.global_trajectory_expert.model]
        
        # 使用类似paligemma_with_expert的forward逻辑
        global_outputs = self._forward_with_attention_bridge(
            models, inputs_embeds, att_2d_masks, position_ids
        )
        
        # 4. 输出全局轨迹
        global_suffix_out = global_outputs[1][:, -self.config.n_action_steps:]
        global_trajectory = self.action_out_proj(global_suffix_out)
        
        return global_trajectory
    
    def forward_action_expert_with_global_input(self, prefix_embs, prefix_pad_masks, 
                                                prefix_att_masks, state, global_trajectory, timestep):
        """实时action expert前向传播（输入改为全局轨迹）"""
        
        # 1. 使用全局轨迹作为输入（而非noise）
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(
            state, global_trajectory, timestep
        )
        
        # 2. 准备attention masks
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        
        # 3. action expert前向传播（能注意到paligemma + global expert）
        # 这里需要让action expert能看到全局轨迹头的输出
        # 复用原有的paligemma_with_expert forward逻辑
        (_, suffix_out), _ = self.paligemma_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )
        
        # 4. 输出最终动作
        suffix_out = suffix_out[:, -self.config.n_action_steps:]
        final_actions = self.action_out_proj(suffix_out)
        
        return final_actions
    
    def _forward_with_attention_bridge(self, models, inputs_embeds, attention_mask, position_ids):
        """实现模型间注意力桥接（简化版本，核心逻辑）"""
        # 这里实现全局头能双向注意到paligemma的逻辑
        # 复用paligemma_with_expert的attention机制
        
        num_layers = self.paligemma_with_expert.config.paligemma_config.text_config.num_hidden_layers
        
        for layer_idx in range(num_layers):
            # 计算Q, K, V
            query_states, key_states, value_states = [], [], []
            
            for i, hidden_states in enumerate(inputs_embeds):
                if hidden_states is None:
                    continue
                    
                layer = models[i].layers[layer_idx]
                hidden_states = layer.input_layernorm(hidden_states)
                hidden_shape = (*hidden_states.shape[:-1], -1, layer.self_attn.head_dim)
                
                query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape)
                key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape)
                value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape)
                
                query_states.append(query_state)
                key_states.append(key_state)
                value_states.append(value_state)
            
            # 拼接并应用attention
            query_states = torch.cat(query_states, dim=1)
            key_states = torch.cat(key_states, dim=1)
            value_states = torch.cat(value_states, dim=1)
            
            # 应用RoPE等（复用原有逻辑）
            from .utils import apply_rope
            query_states = apply_rope(query_states, position_ids)
            key_states = apply_rope(key_states, position_ids)
            
            # 执行attention
            att_output = self.paligemma_with_expert.attention_interface(
                query_states, key_states, value_states, attention_mask
            )
            
            # 分离输出并应用残差连接
            outputs_embeds = []
            start = 0
            for i, hidden_states in enumerate(inputs_embeds):
                if hidden_states is not None:
                    end = start + hidden_states.shape[1]
                    layer = models[i].layers[layer_idx]
                    
                    out_emb = layer.self_attn.o_proj(att_output[:, start:end])
                    out_emb += hidden_states  # 残差连接
                    after_first_residual = out_emb.clone()
                    
                    out_emb = layer.post_attention_layernorm(out_emb)
                    out_emb = layer.mlp(out_emb)
                    out_emb += after_first_residual  # 第二个残差连接
                    
                    outputs_embeds.append(out_emb)
                    start = end
                else:
                    outputs_embeds.append(None)
            
            inputs_embeds = outputs_embeds
        
        # 最终norm
        final_outputs = []
        for i, hidden_states in enumerate(inputs_embeds):
            if hidden_states is not None:
                out_emb = models[i].norm(hidden_states)
                final_outputs.append(out_emb)
            else:
                final_outputs.append(None)
        
        return final_outputs
    
    def forward(self, images, img_masks, lang_tokens, lang_masks, 
                state, actions, noise=None, time=None) -> Tensor:
        """训练时的前向传播"""
        bsize = state.shape[0]
        dtype = state.dtype
        device = state.device
        
        # 生成noise和time
        if noise is None:
            actions_shape = (bsize, self.config.n_action_steps, self.config.max_action_dim)
            noise = torch.randn(actions_shape, device=device, dtype=dtype)
        
        if time is None:
            time = self.sample_time(bsize, device).to(dtype)
        
        # 准备前缀嵌入
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        
        # 🔥 1. 全局轨迹头前向传播
        time_expanded = time[:, None, None]
        x_t_global = time_expanded * noise + (1 - time_expanded) * actions
        u_t_global = noise - actions
        
        global_trajectory = self.forward_global_trajectory(
            prefix_embs, prefix_pad_masks, prefix_att_masks,
            state, x_t_global, time
        )
        
        # 🔥 2. 实时action expert前向传播（使用全局轨迹作为输入）
        final_actions = self.forward_action_expert_with_global_input(
            prefix_embs, prefix_pad_masks, prefix_att_masks,
            state, global_trajectory, time
        )
        
        # 🔥 3. 计算损失（需要同时训练两个头）
        global_losses = F.mse_loss(u_t_global, global_trajectory, reduction="none")
        action_losses = F.mse_loss(actions, final_actions, reduction="none")
        
        # 组合损失
        total_losses = global_losses + action_losses
        
        return total_losses
    
    def sample_actions(self, images, img_masks, lang_tokens, lang_masks, 
                       state, noise=None) -> Tensor:
        """推理时的动作采样"""
        bsize = state.shape[0]
        device = state.device
        dtype = state.dtype
        
        if noise is None:
            actions_shape = (bsize, self.config.n_action_steps, self.config.max_action_dim)
            noise = torch.randn(actions_shape, device=device, dtype=dtype)
        
        # 准备前缀
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        
        # 🔥 采样过程：先全局规划，再实时细化
        dt = torch.tensor(-1.0 / self.config.num_steps, dtype=dtype, device=device)
        x_t = noise
        time = torch.tensor(1.0, dtype=dtype, device=device)
        
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            
            # 1. 全局轨迹预测
            global_trajectory = self.forward_global_trajectory(
                prefix_embs, prefix_pad_masks, prefix_att_masks,
                state, x_t, expanded_time
            )
            
            # 2. 基于全局轨迹的实时动作细化
            v_t = self.forward_action_expert_with_global_input(
                prefix_embs, prefix_pad_masks, prefix_att_masks,
                state, global_trajectory, expanded_time
            )
            
            # Euler步进
            x_t += dt * v_t
            time += dt
        
        return x_t