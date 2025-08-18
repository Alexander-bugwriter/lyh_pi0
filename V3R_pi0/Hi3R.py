import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import numpy as np
from torch import Tensor
from typing import Optional, Union, List
from transformers import AutoTokenizer, GemmaForCausalLM

from .paligemma_with_expert import PaliGemmaWithExpertConfig, PaliGemmaWithExpertModel
from .utils import (
    create_sinusoidal_pos_embedding,
    make_att_2d_masks,
    resize_with_pad,
    sample_beta,
)

# 🔥 新增：支持safetensors
try:
    from safetensors.torch import load_file as load_safetensors
    SAFETENSORS_AVAILABLE = True
except ImportError:
    print("⚠️  safetensors未安装，只支持.bin格式")
    SAFETENSORS_AVAILABLE = False

# 复制必要的常量
IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb", 
    "right_wrist_0_rgb",
)


def load_model_weights_hi3r(model_path: str, map_location='cpu'):
    """
    智能加载模型权重，支持.bin和.safetensors格式 (Hi3R版本)
    
    Args:
        model_path: 模型目录路径
        map_location: 加载位置
    
    Returns:
        state_dict: 模型权重字典
        file_format: 文件格式 ('bin' 或 'safetensors')
    """
    import os
    
    # 🔥 优先检查safetensors格式
    safetensors_file = os.path.join(model_path, "model.safetensors")
    bin_file = os.path.join(model_path, "pytorch_model.bin")
    
    if os.path.exists(safetensors_file) and SAFETENSORS_AVAILABLE:
        print(f"   📦 Hi3R加载safetensors格式: {safetensors_file}")
        state_dict = load_safetensors(safetensors_file)
        # safetensors加载后需要移动到指定设备
        if map_location != 'cpu':
            for key in state_dict:
                if isinstance(state_dict[key], torch.Tensor):
                    state_dict[key] = state_dict[key].to(map_location)
        return state_dict, 'safetensors'
    
    elif os.path.exists(bin_file):
        print(f"   📦 Hi3R加载bin格式: {bin_file}")
        state_dict = torch.load(bin_file, map_location=map_location)
        return state_dict, 'bin'
    
    else:
        # 🔥 更详细的错误信息
        available_files = []
        try:
            for file in os.listdir(model_path):
                if file.endswith(('.bin', '.safetensors', '.pth')):
                    file_size = os.path.getsize(os.path.join(model_path, file)) / (1024*1024)
                    available_files.append(f"{file} ({file_size:.1f}MB)")
        except:
            pass
            
        error_msg = f"Hi3R未找到支持的模型权重文件！\n"
        error_msg += f"检查路径: {model_path}\n"
        error_msg += f"期望文件: model.safetensors 或 pytorch_model.bin\n"
        if available_files:
            error_msg += f"可用文件: {available_files}"
        else:
            error_msg += "目录中没有找到权重文件"
            
        raise FileNotFoundError(error_msg)


class Hi3RConfig:
    """Hi3R模型配置类 - 完全独立，不依赖PI0Config"""
    
    def __init__(
        self,
        # 核心模型参数
        n_action_steps: int = 16,
        max_action_dim: int = 16,
        max_state_dim: int = 8,
        proj_width: int = 512,
        
        # 采样和训练参数
        num_steps: int = 50,
        use_cache: bool = True,
        
        # PaliGemma参数
        tokenizer_max_length: int = 256,
        resize_imgs_with_padding: tuple = (224, 224),
        
        # 训练控制
        freeze_vision_encoder: bool = True,
        train_expert_only: bool = True,
        train_state_proj: bool = False,
        attention_implementation: str = "eager",
        
        # 🔥 空间编码器控制 - 完全可选
        use_spatial_encoder: bool = False,
        spatial_tower: str = "cut3r",
        spatial_tower_select_feature: str = "all",
        spatial_camera_config: dict = None,
        
        # 历史特征控制
        use_history_features: bool = False,
        num_sampled_history_frames: int = 5,
        history_sampling_method: str = "uniform",
        history_camera_config: dict = None,
        
        # 融合和投影
        mm_projector_type: str = "mlp2x_gelu",
        mm_hidden_size: int = 768,
        fusion_block: str = "cross_attention",
        
        **kwargs
    ):
        self.n_action_steps = n_action_steps
        self.max_action_dim = max_action_dim  
        self.max_state_dim = max_state_dim
        self.proj_width = proj_width
        self.num_steps = num_steps
        self.use_cache = use_cache
        self.tokenizer_max_length = tokenizer_max_length
        self.resize_imgs_with_padding = resize_imgs_with_padding
        self.freeze_vision_encoder = freeze_vision_encoder
        self.train_expert_only = train_expert_only
        self.train_state_proj = train_state_proj
        self.attention_implementation = attention_implementation
        
        # 空间编码器配置
        self.use_spatial_encoder = use_spatial_encoder
        self.spatial_tower = spatial_tower
        self.spatial_tower_select_feature = spatial_tower_select_feature
        self.spatial_camera_config = spatial_camera_config or {
            "base_0_rgb": False,  # 🔥 默认禁用，避免意外加载
            "left_wrist_0_rgb": False,
            "right_wrist_0_rgb": False,
        }
        
        # 历史特征配置
        self.use_history_features = use_history_features
        self.num_sampled_history_frames = num_sampled_history_frames
        self.history_sampling_method = history_sampling_method
        self.history_camera_config = history_camera_config or {
            "base_0_rgb": False,  # 🔥 默认禁用
            "left_wrist_0_rgb": False,
            "right_wrist_0_rgb": False,
        }
        
        # 融合配置
        self.mm_projector_type = mm_projector_type
        self.mm_hidden_size = mm_hidden_size
        self.fusion_block = fusion_block
        
        # 创建Gemma Expert配置
        self.gemma_expert_config = self._create_gemma_expert_config()
    
    def _create_gemma_expert_config(self):
        """创建Gemma Expert配置"""
        from transformers import CONFIG_MAPPING
        return CONFIG_MAPPING["gemma"](
            attention_bias=False,
            attention_dropout=0.0,
            bos_token_id=2,
            eos_token_id=1,
            head_dim=256,
            hidden_act="gelu_pytorch_tanh",
            hidden_activation="gelu_pytorch_tanh",
            hidden_size=1024,
            initializer_range=0.02,
            intermediate_size=4096,
            max_position_embeddings=8192,
            model_type="gemma",
            num_attention_heads=8,
            num_hidden_layers=18,
            num_key_value_heads=1,
            pad_token_id=0,
            rms_norm_eps=1e-06,
            rope_theta=10000.0,
            torch_dtype="float32",
            transformers_version="4.48.1",
            use_cache=True,
            vocab_size=257152,
        )

    @classmethod
    def from_pi0_config(cls, pi0_config_dict, **overrides):
        """从PI0配置转换为Hi3R配置"""
        # 提取PI0配置中的关键参数
        hi3r_kwargs = {}
        
        # 映射PI0参数到Hi3R参数
        param_mapping = {
            'n_action_steps': 'n_action_steps',
            'max_action_dim': 'max_action_dim', 
            'max_state_dim': 'max_state_dim',
            'proj_width': 'proj_width',
            'num_steps': 'num_steps',
            'use_cache': 'use_cache',
            'tokenizer_max_length': 'tokenizer_max_length',
            'freeze_vision_encoder': 'freeze_vision_encoder',
            'train_expert_only': 'train_expert_only',
            'attention_implementation': 'attention_implementation',
        }
        
        for pi0_key, hi3r_key in param_mapping.items():
            if pi0_key in pi0_config_dict:
                hi3r_kwargs[hi3r_key] = pi0_config_dict[pi0_key]
        
        # 应用覆盖参数
        hi3r_kwargs.update(overrides)
        
        return cls(**hi3r_kwargs)


class Hi3RPolicy:
    """Hi3R策略类 - 完全独立实现，不继承PI0Policy"""
    
    def __init__(
        self, 
        config: Hi3RConfig,
        tokenizer_path: str = "google/paligemma-3b-pt-224"
    ):
        self.config = config
        self.language_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        # 核心模型
        self.model = Hi3RFlowMatching(config)
        
    def reset(self):
        """重置模型状态"""
        if hasattr(self.model, 'paligemma_with_expert'):
            paligemma_model = self.model.paligemma_with_expert
            
            # 重置CUT3R状态（如果使用）
            if (self.config.use_spatial_encoder and 
                hasattr(paligemma_model, 'reset_cut3r_state')):
                paligemma_model.reset_cut3r_state()
            
            # 重置历史特征缓存（如果使用）
            if (self.config.use_history_features and 
                hasattr(paligemma_model, 'history_buffer') and 
                paligemma_model.history_buffer):
                paligemma_model.history_buffer.clear()
                print("History buffer cleared.")
        
    def get_optim_params(self):
        """获取可优化参数"""
        return self.model.parameters()

    @torch.no_grad
    def select_action(self, observation: dict[str, Tensor], noise: Tensor = None):
        """动作选择"""
        self.model.eval()
        
        images, img_masks = self.prepare_images(observation)
        state = self.prepare_state(observation)
        lang_tokens, lang_masks = self.prepare_language(observation)
        
        actions = self.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state, noise=noise
        )
        return actions

    def forward(self, batch: dict[str, Tensor], noise=None, time=None):
        """训练时的前向传播"""
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens, lang_masks = self.prepare_language(batch)
        actions, action_dim = self.prepare_action(batch)
        noise = batch.get("noise", None)
        time = batch.get("time", None)

        losses = self.model.forward(
            images, img_masks, lang_tokens, lang_masks, state, actions, noise, time
        )

        # 处理padding
        actions_is_pad = batch.get("action_is_pad", None)
        if actions_is_pad is not None:
            in_episode_bound = ~actions_is_pad
            losses = losses * in_episode_bound.unsqueeze(-1)

        # 移除padding
        losses = losses[:, :, :action_dim]
        loss = losses.mean()
        
        loss_dict = {
            "l2_loss": loss.item(),
            "losses": losses.clone()
        }
        
        return loss, loss_dict

    def prepare_images(self, observation: dict[str, Tensor]):
        """图像预处理 - 复用PI0逻辑"""
        dtype = observation["state"].dtype
        bsize = observation["state"].shape[0]
        images, img_masks = [], []
        present_img_keys = [key for key in IMAGE_KEYS if key in observation["image"]]
        missing_img_keys = [key for key in IMAGE_KEYS if key not in present_img_keys]

        for key in present_img_keys:
            img = observation["image"][key]
            img = img.to(dtype) / 127.5 - 1.0
            img = resize_with_pad(
                img, *self.config.resize_imgs_with_padding, pad_value=-1.0
            )
            images.append(img)
            img_masks.append(torch.ones((bsize,), dtype=torch.bool, device=img.device))

        for key in missing_img_keys:
            img = torch.full_like(img, fill_value=-1.0)
            images.append(img)
            img_masks.append(torch.zeros((bsize,), dtype=torch.bool, device=img.device))

        images = torch.stack(images, dim=1)
        img_masks = torch.stack(img_masks, dim=1)
        return images, img_masks

    def prepare_state(self, observation: dict[str, Tensor]):
        """状态预处理"""
        state = observation["state"]
        state = F.pad(state, (0, self.config.max_state_dim - state.shape[1]))
        return state

    def prepare_action(self, observation: dict[str, Tensor]):
        """动作预处理"""
        action = observation["action"]
        action_dim = action.shape[-1]
        action = F.pad(action, (0, self.config.max_action_dim - action_dim))
        return action, action_dim

    def prepare_language(self, observation: dict[str, Tensor]):
        """语言预处理 - 复用PI0逻辑"""
        lang_tokens = observation.get("lang_tokens", None)
        lang_masks = observation.get("lang_masks", None)
        prompt = observation.get("prompt", None)

        if prompt is None and (lang_tokens is None or lang_masks is None):
            raise ValueError(
                "Either 'prompt' or ('lang_tokens', 'lang_masks') must be provided."
            )

        device = observation["state"].device
        if prompt is not None and (lang_tokens is None or lang_masks is None):
            prompt = [p if p.startswith("<bos>") else f"<bos>{p}" for p in prompt]
            prompt = [p if p.endswith("\n") else f"{p}\n" for p in prompt]
            tokenized_prompt = self.language_tokenizer(
                prompt,
                padding="max_length",
                padding_side="right",
                max_length=self.config.tokenizer_max_length,
                return_tensors="pt",
            )
            lang_tokens = tokenized_prompt["input_ids"].to(device=device)
            lang_masks = tokenized_prompt["attention_mask"].to(
                device=device, dtype=torch.bool
            )
        else:
            lang_tokens = observation["lang_tokens"].to(device=device)
            lang_masks = observation["lang_masks"].to(device=device, dtype=torch.bool)

        return lang_tokens, lang_masks

    @classmethod
    def from_pretrained(cls, model_path: str, config_overrides: dict = None):
        """从预训练模型加载 - 🔥 支持safetensors"""
        import json
        import os
        
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
            
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # 转换为Hi3R配置
        overrides = config_overrides or {}
        config = Hi3RConfig.from_pi0_config(config_dict, **overrides)
        
        # 创建模型
        policy = cls(config)
        
        # 🔥 使用新的智能加载函数
        try:
            state_dict, file_format = load_model_weights_hi3r(model_path, 'cpu')
            
            # 尝试加载权重，允许部分不匹配
            missing_keys, unexpected_keys = policy.model.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                print(f"⚠️  缺少的权重键: {len(missing_keys)} 个")
                for key in missing_keys[:5]:  # 只显示前5个
                    print(f"   - {key}")
                if len(missing_keys) > 5:
                    print(f"   ... 还有 {len(missing_keys) - 5} 个")
            
            if unexpected_keys:
                print(f"⚠️  意外的权重键: {len(unexpected_keys)} 个")
                for key in unexpected_keys[:5]:  # 只显示前5个
                    print(f"   - {key}")
                if len(unexpected_keys) > 5:
                    print(f"   ... 还有 {len(unexpected_keys) - 5} 个")
            
            print(f"✅ 从 {model_path} 加载模型权重完成 (格式: {file_format})")
            
        except Exception as e:
            print(f"⚠️  权重加载警告: {e}")
            print("   继续使用随机初始化的权重")
        
        return policy


class Hi3RFlowMatching(nn.Module):
    """Hi3R Flow Matching核心模型 - 双头架构"""
    
    def __init__(self, config: Hi3RConfig):
        super().__init__()
        self.config = config
        
        # 🔥 创建PaliGemma配置（可选空间编码器）
        paligemma_config = PaliGemmaWithExpertConfig(
            freeze_vision_encoder=config.freeze_vision_encoder,
            train_expert_only=config.train_expert_only,
            attention_implementation=config.attention_implementation,
            
            # 🔥 关键：完全受控的空间编码器配置
            use_spatial_encoder=config.use_spatial_encoder,
            spatial_tower=config.spatial_tower,
            spatial_tower_select_feature=config.spatial_tower_select_feature,
            spatial_camera_config=config.spatial_camera_config,
            
            # 历史特征配置
            use_history_features=config.use_history_features,
            num_sampled_history_frames=config.num_sampled_history_frames,
            history_sampling_method=config.history_sampling_method,
            history_camera_config=config.history_camera_config,
            
            # 融合配置
            mm_projector_type=config.mm_projector_type,
            mm_hidden_size=config.mm_hidden_size,
            fusion_block=config.fusion_block,
        )
        
        # PaliGemma + Action Expert
        self.paligemma_with_expert = PaliGemmaWithExpertModel(paligemma_config)
        
        # 🔥 全局轨迹专家（与action expert同构）
        self.global_trajectory_expert = GemmaForCausalLM(config.gemma_expert_config)
        self.global_trajectory_expert.model.embed_tokens = None  # 移除词嵌入
        
        # 投影层
        self.state_proj = nn.Linear(config.max_state_dim, config.proj_width)
        self.action_in_proj = nn.Linear(config.max_action_dim, config.proj_width)
        self.global_trajectory_in_proj = nn.Linear(config.max_action_dim, config.proj_width)
        self.action_out_proj = nn.Linear(config.proj_width, config.max_action_dim)
        
        # 时间MLP
        self.action_time_mlp_in = nn.Linear(config.proj_width * 2, config.proj_width)
        self.action_time_mlp_out = nn.Linear(config.proj_width, config.proj_width)
        
        self.set_requires_grad()

    def set_requires_grad(self):
        """设置参数训练状态"""
        # PaliGemma设置（继承原逻辑）
        self.paligemma_with_expert.set_requires_grad()
        
        # 全局轨迹专家始终可训练
        for param in self.global_trajectory_expert.parameters():
            param.requires_grad = True
        for param in self.global_trajectory_in_proj.parameters():
            param.requires_grad = True
            
        # 状态投影
        for param in self.state_proj.parameters():
            param.requires_grad = self.config.train_state_proj

    def sample_time(self, bsize, device):
        """采样时间步"""
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks):
        """嵌入前缀（图像+语言）"""
        bsize = images.shape[0]
        device = images.device
        dtype = images.dtype

        # 🔥 图像嵌入（支持可选空间编码器）
        img_emb = self.paligemma_with_expert.embed_image(images)
        
        if img_emb.dim() == 4:  # (B, N, L, D)
            num_patches_per_img = img_emb.shape[2]
            img_emb = img_emb.view(bsize, -1, img_emb.shape[-1])
            img_masks = einops.repeat(img_masks, "b n -> b (n l)", l=num_patches_per_img)
        else:
            num_patch = img_emb.shape[1] // images.shape[1]
            img_masks = einops.repeat(img_masks, "b n -> b (n l)", l=num_patch)

        img_emb = img_emb.to(dtype=dtype) * (img_emb.shape[-1] ** 0.5)
        num_img_embs = img_emb.shape[1]

        # 语言嵌入
        lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
        lang_emb = lang_emb.to(dtype=dtype) * np.sqrt(lang_emb.shape[-1])

        # 拼接
        embs = torch.cat([img_emb, lang_emb], dim=1)
        pad_masks = torch.cat([img_masks, lang_masks], dim=1)
        att_masks = torch.zeros(
            (bsize, num_img_embs + lang_emb.shape[1]), 
            device=device, dtype=torch.bool
        )
        
        return embs, pad_masks, att_masks

    def embed_suffix(self, state, noisy_actions, timestep, use_global_proj=False):
        """嵌入后缀（状态+噪声动作+时间）"""
        bsize = state.shape[0]
        device = state.device
        dtype = state.dtype

        # 状态嵌入
        state_emb = self.state_proj(state)

        # 时间嵌入
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.config.proj_width,
            min_period=4e-3, max_period=4.0, device=device,
        ).type(dtype=dtype)

        # 🔥 选择不同的动作投影
        if use_global_proj:
            action_emb = self.global_trajectory_in_proj(noisy_actions)
        else:
            action_emb = self.action_in_proj(noisy_actions)
            
        time_emb = einops.repeat(time_emb, "b d -> b n d", n=action_emb.shape[1])
        action_time_emb = torch.cat([action_emb, time_emb], dim=-1)

        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)
        action_time_emb = self.action_time_mlp_out(action_time_emb)
        action_time_dim = action_time_emb.shape[1]

        embs = torch.cat([state_emb[:, None], action_time_emb], dim=1)
        pad_masks = torch.ones(
            (bsize, action_time_dim + 1), device=device, dtype=torch.bool
        )
        att_masks = torch.zeros(
            (bsize, action_time_dim + 1), device=device, dtype=torch.bool
        )
        att_masks[:, :2] = True

        return embs, pad_masks, att_masks

    def forward_global_trajectory(self, prefix_embs, prefix_pad_masks, prefix_att_masks,
                                  state, noisy_actions, timestep):
        """全局轨迹头前向传播"""
        
        # 1. 嵌入全局轨迹后缀
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(
            state, noisy_actions, timestep, use_global_proj=True
        )
        
        # 2. 拼接前缀和后缀
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        
        # 3. 全局轨迹expert前向传播（双向注意到paligemma主干）
        inputs_embeds = [prefix_embs, suffix_embs]
        models = [self.paligemma_with_expert.paligemma.language_model.model, 
                  self.global_trajectory_expert.model]
        
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
            state, global_trajectory, timestep, use_global_proj=False
        )
        
        # 2. 准备attention masks
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        
        # 3. action expert前向传播（能注意到paligemma + global expert）
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
        # 复用paligemma_with_expert的attention机制
        # 这里实现全局头能双向注意到paligemma的逻辑
        
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
