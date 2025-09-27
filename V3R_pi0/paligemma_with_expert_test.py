# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from pathlib import Path
from typing import List, Optional, Union
import torchvision.transforms
import torch
from pytest import Cache
from transformers import (
    AutoConfig,
    GemmaForCausalLM,
    PaliGemmaForConditionalGeneration,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.models.auto import CONFIG_MAPPING
import torch.nn as nn
from .utils import (
    apply_rope,
    select_best_resolution,
    resize_and_pad_image,
    divide_to_patches,
    eager_attention_forward,
)
from PIL import Image
import PIL
import numpy as np

from .multimodal_spatial_encoder.builder import build_spatial_tower
from .multimodal_fusion_block.builder import build_multimodal_fusion_block
# from .multimodal_projector.builder import build_vision_projector
from .history_buffer import UnlimitedHistoryBuffer


class PaliGemmaWithExpertConfig(PretrainedConfig):
    model_type = "PaliGemmaWithExpertModel"
    sub_configs = {"paligemma_config": AutoConfig, "gemma_expert_config": AutoConfig}

    def __init__(
        self,
        paligemma_config: dict | None = None,
        gemma_expert_config: dict | None = None,
        freeze_vision_encoder: bool = True,
        train_expert_only: bool = True,
        attention_implementation: str = "eager",

        mode: str = "infer", 
        components_path: str = None,
        # 空间编码器参数

        use_spatial_encoder: bool = True,
        spatial_tower: str = "cut3r",  # "cut3r", "vggt", "spann3r"
        spatial_tower_select_feature: str = "all",  # "camera_tokens", "patch_tokens", "all"
        spatial_fusion_method: str = "concat",  # "residual", "concat"
        spatial_camera_config: dict = None,  # 关键：相机级别的空间编码配置
        # 新增历史特征参数
        use_history_features: bool = True,
        num_sampled_history_frames: int = 3,  # 从历史中采样的帧数 (3, 4, 5 等)
        history_sampling_method: str = "uniform",  # "uniform", "recent"
        history_camera_config: dict = None,  # 🔥 新增：控制哪些相机使用历史特征

        # 🔥 添加投影器和融合器配置
        fusion_block: str = "cross_attention",  # 融合块类型
        **kwargs,
    ):
        self.freeze_vision_encoder = freeze_vision_encoder
        self.train_expert_only = train_expert_only
        self.attention_implementation = attention_implementation

        if paligemma_config is None:
            # Default config from Pi0
            self.paligemma_config = CONFIG_MAPPING["paligemma"](
                transformers_version="4.48.1",
                _vocab_size=257152,
                bos_token_id=2,
                eos_token_id=1,
                hidden_size=2048,
                image_token_index=257152,
                model_type="paligemma",
                pad_token_id=0,
                projection_dim=2048,
                text_config={
                    "hidden_activation": "gelu_pytorch_tanh",
                    "hidden_size": 2048,
                    "intermediate_size": 16384,
                    "model_type": "gemma",
                    "num_attention_heads": 8,
                    "num_hidden_layers": 18,
                    "num_image_tokens": 256,
                    "num_key_value_heads": 1,
                    "torch_dtype": "float32",
                    "vocab_size": 257152,
                },
                vision_config={
                    "hidden_size": 1152,
                    "intermediate_size": 4304,
                    "model_type": "siglip_vision_model",
                    "num_attention_heads": 16,
                    "num_hidden_layers": 27,
                    "num_image_tokens": 256,
                    "patch_size": 14,
                    "projection_dim": 2048,
                    "projector_hidden_act": "gelu_fast",
                    "torch_dtype": "float32",
                    "vision_use_head": False,
                },
            )
        elif isinstance(self.paligemma_config, dict):
            # Override Pi0 default config for PaliGemma
            if "model_type" not in gemma_expert_config:
                paligemma_config["model_type"] = "paligemma"

            cfg_cls = CONFIG_MAPPING[paligemma_config["model_type"]]
            self.paligemma_config = cfg_cls(**paligemma_config)

        if gemma_expert_config is None:
            # Default config from Pi0
            self.gemma_expert_config = CONFIG_MAPPING["gemma"](
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
        elif isinstance(self.gemma_expert_config, dict):
            # Override Pi0 default config for Gemma Expert
            if "model_type" not in gemma_expert_config:
                gemma_expert_config["model_type"] = "gemma"

            cfg_cls = CONFIG_MAPPING[paligemma_config["model_type"]]
            self.gemma_expert_config = cfg_cls(**gemma_expert_config)
        else:
            self.gemma_expert_config=gemma_expert_config
        
        self.mode_type=mode#设置推理或者训练模式
        self.components_path = components_path
        #空间编码器配置
        self.use_spatial_encoder = use_spatial_encoder
        self.spatial_tower = spatial_tower
        self.mm_spatial_tower = spatial_tower  # VLM3R兼容性
        self.spatial_tower_select_feature = spatial_tower_select_feature
        self.spatial_fusion_method = spatial_fusion_method

        # 🔥 历史特征配置
        self.use_history_features = use_history_features
        self.num_sampled_history_frames = num_sampled_history_frames
        self.history_sampling_method = history_sampling_method
        #投影器和融合模块配置
        self.fusion_block = fusion_block
        self.mm_hidden_size = 768  # CUT3R输出维度，方便后续使用

        if spatial_camera_config is None:
            self.spatial_camera_config = {
                "base_0_rgb": True,        
                "left_wrist_0_rgb": True,
                "right_wrist_0_rgb": False,
            }
            print("spatial camera config:",self.spatial_camera_config)
        else:
            self.spatial_camera_config = spatial_camera_config
            print("spatial camera config:",self.spatial_camera_config)

        # 🔥 历史特征相机配置
        if history_camera_config is None:
            self.history_camera_config = {
                "base_0_rgb": False,        # 默认均不使用历史帧
                "left_wrist_0_rgb": False,
                "right_wrist_0_rgb": False,
            }
            print("history camera config:",self.history_camera_config)
        else:
            self.history_camera_config = history_camera_config
            print("history camera config:",self.history_camera_config)

        super().__init__(**kwargs)
    def __post_init__(self):
        super().__post_init__()
        if self.train_expert_only and not self.freeze_vision_encoder:
            raise ValueError(
                "You set `freeze_vision_encoder=False` and `train_expert_only=True` which are not compatible."
            )

        if self.attention_implementation not in ["eager", "fa2", "flex"]:
            raise ValueError(
                f"Wrong value provided for `attention_implementation` ({self.attention_implementation}). Expected 'eager', 'fa2' or 'flex'."
            )


class PaliGemmaWithExpertModel(PreTrainedModel):
    config_class = PaliGemmaWithExpertConfig

    def __init__(self, config: PaliGemmaWithExpertConfig):
        super().__init__(config=config)
        self.config = config
        
        self.paligemma = PaliGemmaForConditionalGeneration(
            config=config.paligemma_config
        )
        self.gemma_expert = GemmaForCausalLM(config=config.gemma_expert_config)
        # Remove unused embed_tokens
        self.gemma_expert.model.embed_tokens = None

        self.attention_interface = self.get_attention_interface()

        # self.to_bfloat16_like_physical_intelligence()
        self.set_requires_grad()

        # 🔥 获取正确的维度
        vision_hidden_size = config.paligemma_config.vision_config.hidden_size  # 1152 (SigLIP)
        paligemma_projection_dim = config.paligemma_config.projection_dim  # 2048
        if config.use_spatial_encoder and config.mode_type=="infer":
            self.spatial_tower = build_spatial_tower(config, delay_load=False)
        else:
            self.spatial_tower=None

            # 🔥 修正：融合配置使用原始视觉特征维度
        fusion_config = type('Config', (), {
            'fusion_block': config.fusion_block,
            'hidden_size': paligemma_projection_dim,  # 2048 (最终输出维度)
            'mm_hidden_size': vision_hidden_size,     # 1152 (SigLIP原始输出) ✅
            'spatial_feature_dim': config.mm_hidden_size,  # 768 (CUT3R输出) ✅
        })()
        self.fusion_block = build_multimodal_fusion_block(fusion_config)

        # 🔥 创建统一的投影器 (融合后使用)
        # mm_projector_config = type('Config', (), {
        #     'mm_hidden_size': vision_hidden_size,  # 1152 (融合后的特征维度)
        #     'hidden_size': paligemma_projection_dim,  # 2048 (目标维度)
        #     'mm_projector_type': config.mm_projector_type
        # })()
        # self.mm_projector = build_vision_projector(mm_projector_config)
        self.mm_projector = self.paligemma.multi_modal_projector

        # 历史特征间隔符保持不变
        embed_std = 1 / torch.sqrt(torch.tensor(config.paligemma_config.projection_dim, dtype=torch.float32))
        self.spatial_separator_token = nn.Parameter(
            torch.randn(1, config.paligemma_config.projection_dim) * embed_std
        )
        self._load_modular_components(config.components_path)

            # 🔥 历史特征缓存 (保持你的创新)
        if config.use_history_features and config.mode_type=="infer":
            self.history_buffer = UnlimitedHistoryBuffer(config)
        else:
            self.history_buffer = None

    def _load_modular_components(self, components_path=None):
        """模块化加载训练好的组件"""
        
        # 使用spatial_encoder_checkpoint目录
        # script_dir = os.path.dirname(os.path.abspath(__file__))
        # #project_root = os.path.abspath(os.path.join(script_dir, '..'))
        # base_path = Path(script_dir) / 'spatial_encoder_checkpoint'
        if components_path is None:
            print("组件路径未指定，使用随机初始化")
            return
        else:
            components_path = Path(components_path)
            lora_adapter_path = components_path / "lora_adapter"
            if lora_adapter_path.exists():
                print("lora存在于指定路径")
                self._merge_lora_adapter(lora_adapter_path)
            modules_config = {
                'fusion_block': {
                    'path': components_path / 'fusion_block.pth',
                    'component': self.fusion_block,
                    'name': '融合模块'
                },
                'mm_projector': {
                'path': components_path / 'mm_projector.pth',
                'component': self.mm_projector,
                'name': 'MM投影器'
                },
                'spatial_separator_token': {
                    'path': components_path / 'spatial_separator_token.pth',
                    'component': self.spatial_separator_token,
                    'name': '分隔符参数'
                }
            }
            
            for module_name, config in modules_config.items():
                if config['path'].exists():
                    try:
                        if hasattr(config['component'], 'load_state_dict'):
                            # 标准模块加载
                            state_dict = torch.load(config['path'], map_location='cpu')
                            config['component'].load_state_dict(state_dict)
                            print(f"加载{config['name']}: {config['path']}")
                        else:
                            # 单个参数加载
                            param_data = torch.load(config['path'], map_location='cpu')
                            config['component'].data = param_data
                            print(f"加载{config['name']}: {config['path']}")
                    except Exception as e:
                        print(f"加载{config['name']}失败: {e}，使用随机初始化")
            else:
                print(f"{config['name']}不存在，使用随机初始化")
    def _merge_lora_adapter(self, lora_adapter_path):
        """直接合并LoRA adapter"""
        print(f"合并LoRA adapter: {lora_adapter_path}")
    
        try:
            from peft import PeftModel
        
            language_model = self.paligemma.language_model
            peft_model = PeftModel.from_pretrained(language_model, lora_adapter_path)
            merged_model = peft_model.merge_and_unload()
            self.paligemma.language_model = merged_model
        
            print("  LoRA合并完成")
        
        except Exception as e:
            print(f"  LoRA合并失败: {e}")

    def save_modular_components(self, save_path=None):
        """保存训练好的模块组件"""
        if save_path is None:
            # 🔥 默认保存到spatial_encoder_checkpoint
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(script_dir, '..'))
            save_path = Path(project_root) / 'spatial_encoder_checkpoint'
        else:
            save_path = Path(save_path)
        
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 保存各模块
        torch.save(self.fusion_block.state_dict(), save_path / 'fusion_block.pth')
        torch.save(self.mm_projector.state_dict(), save_path / 'mm_projector.pth') 
        torch.save(self.spatial_separator_token.data, save_path / 'spatial_separator_token.pth')
        
        print(f"💾 模块组件保存完成: {save_path}")
    
    
    def set_requires_grad(self):
        """sets the requires_grad attribute of the model parameters based on the configuration.
        If `freeze_vision_encoder` is True, the vision tower parameters are frozen.
        If `train_expert_only` is True, the entire PaliGemma model is frozen.
        """
        if self.config.freeze_vision_encoder:
            self.paligemma.vision_tower.eval()
            for params in self.paligemma.vision_tower.parameters():
                params.requires_grad = False

        if self.config.train_expert_only:
            self.paligemma.eval()
            for params in self.paligemma.parameters():
                params.requires_grad = False

        if hasattr(self, 'spatial_projector') and self.spatial_projector is not None:
            for params in self.spatial_projector.parameters():
                params.requires_grad = True
        if hasattr(self, 'fusion_block') and self.fusion_block is not None:
            for params in self.fusion_block.parameters():
                params.requires_grad = True

        if hasattr(self, 'spatial_tower') and self.spatial_tower is not None:
            for param in self.spatial_tower.parameters():
                param.requires_grad = False  # CUT3R完全冻结

        # ✅ 新增组件：确保可训练
        #预定义的投影器不用训练
        if hasattr(self, 'mm_projector') and self.mm_projector is not None:
            for params in self.mm_projector.parameters():
                params.requires_grad = True  # 投影器需要训练

        if hasattr(self, 'fusion_block') and self.fusion_block is not None:
            for params in self.fusion_block.parameters():
                params.requires_grad = True  # 融合块需要训练

        if hasattr(self, 'spatial_separator_token'):
            self.spatial_separator_token.requires_grad = True  # 分隔符需要训练

    def train(self, mode: bool = True):
        super().train(mode)
        if self.config.freeze_vision_encoder:
            self.paligemma.vision_tower.eval()
        if self.config.train_expert_only:
            self.paligemma.eval()

    # def to_bfloat16_like_physical_intelligence(self):
    #     """casts the model to bfloat16.

    #     Modules not casted to bfloat16:
    #     - paligemma.language_model.model.embed_tokens.weight
    #     - paligemma.language_model.model.norm.weight
    #     - gemma_expert.model.norm.weight
    #     - gemma_expert.lm_head.weight
    #     """
    #     self.paligemma = self.paligemma.to(dtype=torch.bfloat16)

    #     params_to_change_dtype = [
    #         "language_model.model.layers",
    #         "gemma_expert.model.layers",
    #         "vision_tower",
    #         "multi_modal",
    #     ]
    #     for name, param in self.named_parameters():
    #         if any(selector in name for selector in params_to_change_dtype):
    #             param.data = param.data.to(dtype=torch.bfloat16)

    # def embed_image(self, image: torch.Tensor):
    #     return self.paligemma.get_image_features(image)

    def debug_spatial_tower(self):
        """调试spatial_tower的接口和方法"""
        print("🔍 调试spatial_tower接口:")
        print(f"Type: {type(self.spatial_tower)}")
        print(f"Methods: {[m for m in dir(self.spatial_tower) if not m.startswith('_')]}")
        
        # 检查forward方法的参数
        import inspect
        if hasattr(self.spatial_tower, 'forward'):
            sig = inspect.signature(self.spatial_tower.forward)
            print(f"Forward signature: {sig}")
        
        # 检查是否有特殊的单图像处理方法
        single_image_methods = [m for m in dir(self.spatial_tower) if 'single' in m.lower() or 'image' in m.lower()]
        print(f"可能的单图像方法: {single_image_methods}")
    

    def embed_image(self, image: torch.Tensor):
        """
        🔥 核心方法：选择性空间编码
        
        Args:
            image: (B, N, C, H, W) 输入图像，N对应IMAGE_KEYS的顺序
        
        Returns:
            enhanced_features: (B, N, L, D) 增强后的特征
        """
        batch_size, num_images = image.shape[:2]
        IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
        
        enhanced_features_list = []
        
        for img_idx in range(num_images):
            #print("embed_image里的图像尺寸：", image[:, img_idx].shape)
            current_img = image[:, img_idx]  # (B, C, H, W)
            camera_key = IMAGE_KEYS[img_idx] if img_idx < len(IMAGE_KEYS) else f"camera_{img_idx}"
            
            # 🔥 根据配置决定是否使用空间编码
            needs_spatial = (
                self.config.use_spatial_encoder 
                and self.spatial_tower is not None
                and self.config.spatial_camera_config.get(camera_key, False)
            )
            # 🔥 根据config判断是否使用历史特征
            needs_history = (
                self.config.use_history_features
                and self.history_buffer is not None
                and self.config.history_camera_config.get(camera_key, False)
            )
            
            if needs_spatial:
                # 🔥 使用CUT3R空间编码（CUT3R内部自动管理历史状态）
                # enhanced_features = self._encode_with_cut3r(current_img)
                #print(f"✅ {camera_key}: 使用CUT3R空间编码")
                vision_outputs = self.paligemma.vision_tower(image)
                raw_vision_features = vision_outputs.last_hidden_state  # (B, L, 1152)
                
                # 2. 🔥 关键：直接调用CUT3R，它内部会自动更新和利用历史状态
                with torch.no_grad():
                    image_fp16 = image.half()
                    camera_tokens, patch_tokens = self.spatial_tower(image_fp16)
                    camera_tokens = camera_tokens.to(raw_vision_features.dtype)
                    patch_tokens = patch_tokens.to(raw_vision_features.dtype)

                spatial_features = [{"camera_tokens": camera_tokens, "patch_tokens": patch_tokens}]
                enhanced_features=self.fuse_2D_with_cut3r(raw_vision_features,spatial_features)
            else:
                # 标准PaliGemma处理
                enhanced_features = self.paligemma.get_image_features(current_img)
                #print(f"📷 {camera_key}: 使用标准PaliGemma")

            # 🔥 历史特征处理
            if needs_history:
                final_features = self.history_buffer.get_enhanced_features_with_separators(
                    current_features=enhanced_features,separator_token=self.spatial_separator_token,max_frame_num=self.num_sampled_history_frames
                )
                self.history_buffer.append(enhanced_features, self.spatial_separator_token)
                #print(f"   📁 {camera_key} 启用历史特征增强")
            else:
                final_features = enhanced_features
                #print(f"   ⚡ {camera_key} 仅使用当前帧")
        
            enhanced_features_list.append(final_features)

        return enhanced_features_list
    

    def embed_image_with_preprocessing_feature(self, image: torch.Tensor, 
                                            precomputed_spatial_features=None,
                                            frame_index: int = 0):
        """
        🔥 支持历史特征的预计算特征处理
        """
        batch_size, num_images = image.shape[:2]
        IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
        enhanced_features_list = []
        # 收集原始图像特征
        # raw_images_feature = {}
        # spatial_tokens = {}
        
        for img_idx in range(num_images):
            camera_key = IMAGE_KEYS[img_idx] if img_idx < len(IMAGE_KEYS) else f"camera_{img_idx}"
            current_img = image[:, img_idx]
            needs_spatial = (
            self.config.use_spatial_encoder 
            and self.config.spatial_camera_config.get(camera_key, False)
            and precomputed_spatial_features is not None
            )
            if needs_spatial and camera_key in ["base_0_rgb", "left_wrist_0_rgb"]:
                # 使用空间特征增强
                vision_outputs = self.paligemma.vision_tower(current_img)
                raw_vision_features = vision_outputs.last_hidden_state
                
                if camera_key == "base_0_rgb":
                    spatial_tokens = {
                        'camera_tokens': precomputed_spatial_features['base_camera_tokens'].to(device=raw_vision_features.device, dtype=raw_vision_features.dtype).squeeze(1),
                        'patch_tokens': precomputed_spatial_features['base_patch_tokens'].to(device=raw_vision_features.device, dtype=raw_vision_features.dtype).squeeze(1)
                    }
                else:  # wrist cameras
                    spatial_tokens = {
                        'camera_tokens': precomputed_spatial_features['wrist_camera_tokens'].to(device=raw_vision_features.device, dtype=raw_vision_features.dtype).squeeze(1),
                        'patch_tokens': precomputed_spatial_features['wrist_patch_tokens'].to(device=raw_vision_features.device, dtype=raw_vision_features.dtype).squeeze(1)
                    }
                
                current_enhanced = self.fuse_2D_with_cut3r(raw_vision_features, spatial_tokens)
            else:
                # 标准PaliGemma处理
                current_enhanced = self.paligemma.get_image_features(current_img)
            
            needs_history = (
                self.config.use_history_features
                and self.config.history_camera_config.get(camera_key, False)
                and precomputed_spatial_features is not None
                and "history_info" in precomputed_spatial_features
                and camera_key == "base_0_rgb"
            )
            
            if needs_history:
                history_info_batch = precomputed_spatial_features["history_info"]
                batch_results = []
                
                for batch_idx in range(batch_size):
                    current_sample = current_enhanced[batch_idx:batch_idx+1]
                    device, dtype = current_sample.device, current_sample.dtype
                    
                    # 获取历史帧
                    history_features = []
                    if (batch_idx < len(history_info_batch) and 
                        history_info_batch[batch_idx] is not None and
                        'history_frames' in history_info_batch[batch_idx]):
                        
                        available_history = history_info_batch[batch_idx]['history_frames']
                        
                        # 🎯 循环处理每个历史帧 - 同样简单的if/else
                        for hist_frame in available_history:
                            hist_img = hist_frame['base_image_uint8'].to(device)
                            hist_img_normalized = hist_img.float() / 127.5 - 1.0
                            
                            if needs_spatial:
                                # 历史帧用空间特征增强
                                with torch.no_grad():
                                    hist_vision_outputs = self.paligemma.vision_tower(hist_img_normalized.unsqueeze(0))
                                    hist_raw_features = hist_vision_outputs.last_hidden_state.to(dtype)
                                
                                hist_spatial_tokens = {
                                    'camera_tokens': hist_frame['base_camera_tokens'].to(device=device, dtype=dtype).squeeze(1),
                                    'patch_tokens': hist_frame['base_patch_tokens'].to(device=device, dtype=dtype).squeeze(1)
                                }
                                #print("hist_frame['base_camera_tokens'] shape:", hist_frame['base_camera_tokens'].shape)
                                #print("hist_frame['base_patch_tokens'] shape:", hist_frame['base_patch_tokens'].shape)
                                
                                hist_enhanced = self.fuse_2D_with_cut3r(hist_raw_features, hist_spatial_tokens)
                            else:
                                # 历史帧用标准处理
                                with torch.no_grad():
                                    hist_enhanced = self.paligemma.get_image_features(hist_img_normalized.unsqueeze(0))
                            
                            history_features.append(hist_enhanced)
                    
                    # 🎯 拼接序列：[hist1] + [sep] + [hist2] + [sep] + ... + [current]
                    if history_features:
                        separator = self.spatial_separator_token.expand(1, -1, -1).to(device=device, dtype=dtype)
                        sequence_parts = []
                        
                        for hist_feat in history_features:
                            sequence_parts.append(hist_feat)
                            sequence_parts.append(separator)
                        
                        sequence_parts.append(current_sample)  # 当前帧后无分隔符
                        enhanced_sample = torch.cat(sequence_parts, dim=1)
                    else:
                        enhanced_sample = current_sample
                    
                    batch_results.append(enhanced_sample)
                
                # history buffer已保证固定长度，直接concat
                final_enhanced = torch.cat(batch_results, dim=0)
            else:
                final_enhanced = current_enhanced
            
            enhanced_features_list.append(final_enhanced)
        
        return enhanced_features_list
        
    
    def reset_cut3r_state(self):
        """🔥 重要：重置CUT3R内部状态 - 在episode边界调用"""
        if hasattr(self, 'spatial_tower') and self.spatial_tower is not None:
            # 🔥 直接调用spatial_tower的reset方法
            if hasattr(self.spatial_tower, 'reset_state'):
                self.spatial_tower.reset_state()
                # print("CUT3R is reset")
            else:
                print("警告：spatial_tower没有重置方法")
                print("可用方法:", [m for m in dir(self.spatial_tower) if not m.startswith('_')])
        
        # 重置历史缓存（如果使用的话）
        if hasattr(self, 'history_buffer') and self.history_buffer:
            self.history_buffer.clear()
            # print("History buffer cleared.")

    
              
    def fuse_2D_with_cut3r(self, raw_vision_features, spatial_features):
        """
        🔥 VLM3R方式：内部处理融合+投影，根据fusion类型决定投影时机
        现在两个输入维度正确：
        - raw_vision_features: (B, L, 1152) ✅ 原始SigLIP特征
        - spatial_features: (B, 730, 768) ✅ CUT3R特征
        """
        # print(f"🔍 spatial_features 类型: {type(spatial_features)}")
        # if isinstance(spatial_features, (list, tuple)):
        #     print(f"🔍 spatial_features 长度: {len(spatial_features)}")
        #     if len(spatial_features) > 0:
        #         print(f"🔍 spatial_features[0] 类型: {type(spatial_features[0])}")
        #         if isinstance(spatial_features[0], dict):
        #             print(f"🔍 spatial_features[0] 键: {spatial_features[0].keys()}")
        #         else:
        #             print(f"🔍 spatial_features[0] 形状: {spatial_features[0].shape}")
        # elif isinstance(spatial_features, dict):
        #     print(f"🔍 spatial_features 键: {spatial_features.keys()}")
        # else:
        #     print(f"🔍 spatial_features 形状: {spatial_features.shape if hasattr(spatial_features, 'shape') else 'no shape'}")
        try:
            camera_tokens = spatial_features[0]["camera_tokens"]  # (B, 1, 768)
            patch_tokens = spatial_features[0]["patch_tokens"]    # (B, 729, 768)
            # print("using single fusion")
        except:
            camera_tokens = spatial_features["camera_tokens"]  # (B, 1, 768)
            patch_tokens = spatial_features["patch_tokens"]    # (B, 729, 768)
            # print("using batch fusion")
        #print("camera_tokens shape:", camera_tokens.shape)
        #print("patch_tokens shape:", patch_tokens.shape)
        # VLM3R的特征选择逻辑
        spatial_tower_select_feature = getattr(self.config, "spatial_tower_select_feature", "all")
        spatial_tower_select_feature_list = spatial_tower_select_feature.split(",")
        
        final_image_features = []
        for feature_type in spatial_tower_select_feature_list:
            if feature_type == "camera_tokens":
                final_image_features.append(camera_tokens)
            elif feature_type == "patch_tokens":
                final_image_features.append(patch_tokens)
            elif feature_type == "all":
                final_image_features = [camera_tokens, patch_tokens]
                break
        
        # 拼接空间特征 (768维)
        final_image_features = torch.cat(final_image_features, dim=1).to(device=raw_vision_features.device, dtype=raw_vision_features.dtype)
        
        fusion_block_type = self.config.fusion_block
        
        if fusion_block_type == 'cross_attention':
            # VLM3R方式：先融合原始特征 (1152维 + 768维)，再投影到2048维
            enhanced_features, attn_weights = self.fusion_block(raw_vision_features, final_image_features)
            # 融合后投影
            enhanced_features = self.mm_projector(enhanced_features)
        
        elif fusion_block_type == 'cross_attention_with_mlp':
            # VLM3R方式：先融合原始特征，再投影
            enhanced_features, attn_weights = self.fusion_block(raw_vision_features, patch_tokens)
            # 融合后投影
            enhanced_features = self.mm_projector(enhanced_features)

        elif fusion_block_type == 'transformer':
            # VLM3R方式：先融合原始特征，再投影
            enhanced_features = self.fusion_block(raw_vision_features, final_image_features)
            # 融合后投影
            enhanced_features = self.mm_projector(enhanced_features)

        elif (fusion_block_type == 'mlp_after_clip_proj' 
            or fusion_block_type == 'concat_mlp'
            or fusion_block_type == 'concat_self_attention'):
            # 🔥 这些类型：先投影视觉特征，再融合，无需额外投影
            projected_features = self.mm_projector(raw_vision_features)  # 1152 -> 2048
            enhanced_features = self.fusion_block(projected_features, patch_tokens)
            # 已经是2048维，无需再投影
        
        else:
            raise ValueError(f"Unsupported fusion_block type: {fusion_block_type}")
        
        return enhanced_features  # 统一返回2048维的最终特征
        
        # # 展平patches维度
        # B, num_patches, feature_dim = final_image_features.shape
        # final_image_features = final_image_features.view(B, num_patches * feature_dim)
        
        # # 投影
        # projected_features = self.spatial_projector(final_image_features)
        
        # # 融合
        # if self.config.fusion_block == "cross_attention":
        #     enhanced_features, _ = self.fusion_block(base_features, projected_features)
        # else:
        #     enhanced_features = self.fusion_block(base_features, projected_features)
        
        # return enhanced_features


    def embed_language_tokens(self, tokens: torch.Tensor):
        #return self.paligemma.language_model.model.embed_tokens(tokens)
        # return self.paligemma.language_model.embed_tokens(tokens)
        return self.paligemma.language_model.get_input_embeddings()(tokens)
    def handle_kv_cache(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        past_key_values: Optional[Union[List[torch.FloatTensor], Cache]] = None,
        use_cache: Optional[bool] = None,
        fill_kv_cache: Optional[bool] = None,
    ):
        if use_cache:
            if past_key_values is None:
                past_key_values = {}

            if fill_kv_cache:
                past_key_values[layer_idx] = {
                    "key_states": key_states,
                    "value_states": value_states,
                }
            else:
                key_states = torch.cat(
                    [past_key_values[layer_idx]["key_states"], key_states], dim=1
                )
                value_states = torch.cat(
                    [past_key_values[layer_idx]["value_states"], value_states],
                    dim=1,
                )
        return key_states, value_states, past_key_values

    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[List[torch.FloatTensor], Cache]] = None,
        inputs_embeds: List[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        fill_kv_cache: Optional[bool] = None,
    ):
        """
        Args:
            attention_mask (Optional[torch.Tensor], optional):
                Attention mask with shape (b, seq_len, seq_len). Defaults to None.
            position_ids (Optional[torch.LongTensor], optional):
                Position indices for applying RoPE. Defaults to None.
            past_key_values (Optional[Union[List[torch.FloatTensor], Cache]], optional):
                Optional kv cache. Defaults to None.
            inputs_embeds (List[torch.FloatTensor], optional):
                Input embeddings. Defaults to None.
            use_cache (Optional[bool], optional):
                Whether to use kv cache. Defaults to None.
            fill_kv_cache (Optional[bool], optional):
                Whether to return kv tensors in this forward pass as cache. Defaults to None.

        Returns:
            outputs_embeds (torch.Tensor): Output embeddings.
            past_key_values (Optional[Union[List[torch.FloatTensor], Cache]]):
                Optional kv cache.
        """
        models = [self.paligemma.language_model.model, self.gemma_expert.model]
        #print(f"Model 0 type: {type(models[0])}")
        #print(f"Model 1 type: {type(models[1])}")
        #print(f"Model 0 has layers: {hasattr(models[0], 'layers')}")
        #print(f"Model 1 has layers: {hasattr(models[1], 'layers')}")
        # RMSNorm
        num_layers = self.paligemma.config.text_config.num_hidden_layers
        for layer_idx in range(num_layers):
            query_states = []
            key_states = []
            value_states = []
            for i, hidden_states in enumerate(inputs_embeds):
                if hidden_states is None:
                    continue

                #layer = models[i].layers[layer_idx]
                # 修复：根据模型类型使用正确的访问路径
                try:
                    layer = models[i].model.layers[layer_idx]
                except:
                    layer = models[i].layers[layer_idx]
                hidden_states = layer.input_layernorm(hidden_states)
                hidden_shape = (*hidden_states.shape[:-1], -1, layer.self_attn.head_dim)

                query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape)
                key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape)
                value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape)

                query_states.append(query_state)
                key_states.append(key_state)
                value_states.append(value_state)

            # B,L,H,D with L sequence length, H number of heads, D head dim
            # concatenate on the number of embeddings/tokens
            query_states = torch.cat(query_states, dim=1)
            key_states = torch.cat(key_states, dim=1)
            value_states = torch.cat(value_states, dim=1)

            query_states = apply_rope(query_states, position_ids)
            key_states = apply_rope(key_states, position_ids)

            key_states, value_states, past_key_values = self.handle_kv_cache(
                key_states,
                value_states,
                layer_idx,
                past_key_values=past_key_values,
                use_cache=use_cache,
                fill_kv_cache=fill_kv_cache,
            )

            att_output = self.attention_interface(
                query_states, key_states, value_states, attention_mask
            )

            # first part of att_output is prefix (up to sequence length, [:, 0:prefix_seq_len])
            outputs_embeds = []
            start = 0
            for i, hidden_states in enumerate(inputs_embeds):
                #layer = models[i].layers[layer_idx]
                # 修复：根据模型类型使用正确的访问路径
                try:
                    layer = models[i].model.layers[layer_idx]
                except:
                    layer = models[i].layers[layer_idx]

                if hidden_states is not None:
                    end = start + hidden_states.shape[1]

                    if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                        att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
                    out_emb = layer.self_attn.o_proj(att_output[:, start:end])

                    # first residual
                    out_emb += hidden_states
                    after_first_residual = out_emb.clone()

                    out_emb = layer.post_attention_layernorm(out_emb)
                    out_emb = layer.mlp(out_emb)

                    # second residual
                    out_emb += after_first_residual
                    outputs_embeds.append(out_emb)

                    start = end
                else:
                    outputs_embeds.append(None)

            inputs_embeds = outputs_embeds

        # final norm
        outputs_embeds = []
        for i, hidden_states in enumerate(inputs_embeds):
            if hidden_states is not None:
                #out_emb = models[i].norm(hidden_states)
                try:  # Model 0: GemmaForCausalLM
                    out_emb = models[i].model.norm(hidden_states)
                except:  # Model 1: GemmaModel
                    out_emb = models[i].norm(hidden_states)
                outputs_embeds.append(out_emb)
            else:
                outputs_embeds.append(None)

        return outputs_embeds, past_key_values
                

    def get_attention_interface(self):
        if self.config.attention_implementation == "fa2":
            raise NotImplementedError("FA2 is not implemented (yet)")
        elif self.config.attention_implementation == "flex":
            # attention_interface = flex_attention_forward
            raise NotImplementedError("Flex attention is not implemented (yet)")
        elif self.config.attention_implementation == "eager":
            attention_interface = eager_attention_forward
        elif self.config.attention_implementation == "xformer":
            # attention_interface = xformer_attention_forward
            raise NotImplementedError("Xformer attention is not implemented (yet)")
        else:
            raise ValueError(
                f"Invalid attention implementation: {self.config.attention_implementation}. "
                "Expected one of ['fa2', 'flex', 'eager', 'xformer']."
            )
        return attention_interface


