#!/usr/bin/env python3
"""
简化版PI0训练器 - 支持组件化训练

核心功能:
1. fusion_only: 只训练cross attention融合模块
2. fusion_and_vlm: 训练fusion + VLM LoRA微调

使用方法:
1. 只训练融合模块:
   python simple_pi0_trainer.py --mode fusion_only --base_model_path /path/to/pi0 --data_repo_id dataset_id
   
2. 融合模块+VLM联合训练:
   python simple_pi0_trainer.py --mode fusion_and_vlm --base_model_path /path/to/pi0 --data_repo_id dataset_id --components_path /path/to/pretrained_fusion
"""

# 复用原有的设备检查修复
def patched_is_torch_device_available(device: str) -> bool:
    """修复的设备可用性检查函数 - 支持带索引的设备名称"""
    import torch
    
    if ":" in device:
        device_type = device.split(":")[0]
    else:
        device_type = device
    
    try_device = device_type.lower()
    
    if try_device == "cuda":
        return torch.cuda.is_available()
    elif try_device == "mps":
        return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    elif try_device == "cpu":
        return True
    else:
        raise ValueError(f"Unknown device {try_device}. Supported devices are: cuda, mps or cpu.")

import lerobot.common.utils.utils
lerobot.common.utils.utils.is_torch_device_available = patched_is_torch_device_available
print("设备检查函数已修复")

import os
os.environ['HF_HOME'] = '/opt/liblibai-models/user-workspace2/dataset/.huggingface_cache'
os.environ['HF_DATASETS_CACHE'] = '/opt/liblibai-models/user-workspace2/dataset/.huggingface_cache/datasets'
import argparse
import torch
import pytorch_lightning as L
import json
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from pathlib import Path
from safetensors.torch import save_model
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.configs.policies import PreTrainedConfig
from V3R_pi0.modeling_pi0 import PI0Policy
#from utils.spatiotemporal_lerobot_dataset import Enhanced_LerobotPI0Dataset, enhanced_collate_fn
from utils.spatiotemporal_lerobot_dataset_indexed import Enhanced_LerobotPI0Dataset, enhanced_collate_fn
from peft import PeftModel,get_peft_model, LoraConfig, TaskType
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR, OneCycleLR

# 🎯 训练模式定义
class TrainingMode:
    FUSION_ONLY = "fusion_only"        # 只训练cross attention
    FUSION_AND_VLM = "fusion_and_vlm"  # fusion + VLM LoRA
    SPATIAL_SEPARATOR_TOKEN_ONLY = "token_only"
    HISTORY_FUSION = "history_fusion"  # ✅ 
    # 后续扩展:
    # MULTIFRAME = "multiframe"         # 多帧历史训练 (fusion冻结)
    # ADAPTIVE_SEP = "adaptive_sep"     # 自适应分隔符训练

# 🎯 LoRA配置 (硬编码，经验值)
LORA_CONFIG_2B = {
    "r": 16,
    "lora_alpha": 16,
    "lora_dropout": 0.1,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
}

LORA_CONFIG_300M = {
    "r": 32,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
}
class Lerobot_Trainer(L.LightningModule):
    """简化版PI0 Lightning训练器 - 支持精确的组件训练控制"""
    
    def __init__(self, policy: PI0Policy, training_mode: str, learning_rate: float = 2e-5):
        super().__init__()
        self.policy = policy
        self.training_mode = training_mode
        self.learning_rate = learning_rate
        self.last_episode_idx = None
        
        # 🎯 根据模式配置可训练组件
        self._setup_training_components()
        
        self.save_hyperparameters(ignore=['policy'])
    
    def _setup_training_components(self):
        """🎯 核心方法：根据训练模式配置可训练组件"""
        print(f"🎯 配置训练模式: {self.training_mode}")
        
        # 🔒 默认冻结所有参数
        for param in self.policy.parameters():
            param.requires_grad = False
        
        if self.training_mode == TrainingMode.FUSION_ONLY:
            self._setup_fusion_only()
        elif self.training_mode == TrainingMode.FUSION_AND_VLM:
            self._setup_fusion_and_vlm()
        elif self.training_mode == TrainingMode.SPATIAL_SEPARATOR_TOKEN_ONLY:
            self._setup_separator_token_only()
        elif self.training_mode == TrainingMode.HISTORY_FUSION:  # ✅ 新增
            self._setup_history_fusion()
        else:
            raise ValueError(f"不支持的训练模式: {self.training_mode}")
        
        self._print_trainable_stats()
    def _setup_separator_token_only(self):
        print("模式3: 只训练spatial_separator_token")
        paligemma_model = self.policy.model.paligemma_with_expert
        for param in paligemma_model.parameters():
            param.requires_grad = False
        # if hasattr(paligemma_model, 'spatial_separator_token'):
        #     paligemma_model.spatial_separator_token.requires_grad = True
        #     print("  ✅ spatial_separator_token -> 可训练")
        # else:
        #     print(" spatial_separator_token -> 不存在")
        if hasattr(paligemma_model, 'spatial_separator_token'):
            for param in paligemma_model.spatial_separator_token.parameters():
                param.requires_grad = True
            print("  ✅ spatial_separator_token (时序感知) -> 可训练")
        else:
            print(" spatial_separator_token -> 不存在") 
    def _setup_fusion_only(self):
        """模式1: 只训练Cross Attention融合模块"""
        print("模式1: 只训练Cross Attention融合模块")
        
        paligemma_model = self.policy.model.paligemma_with_expert
        
        # 只启用fusion_block (移除separator_token)
        if hasattr(paligemma_model, 'fusion_block') and paligemma_model.fusion_block is not None:
            for param in paligemma_model.fusion_block.parameters():
                param.requires_grad = True
            print("fusion_block -> 可训练")
        else:
            print("fusion_block -> 不存在")

        if hasattr(paligemma_model, 'mm_projector') and paligemma_model.mm_projector is not None:
            for param in paligemma_model.mm_projector.parameters():
                param.requires_grad = True
            print("mm_projector -> 可训练")
            
        # 🔒 明确冻结不需要的组件
        frozen_components = [
            'spatial_tower', 'paligemma', 
            'spatial_separator_token'  # 明确冻结separator_token
        ]
        for comp_name in frozen_components:
            print(f"{comp_name} -> 冻结")
    
    def _setup_fusion_and_vlm(self):
        """模式2: 训练Cross Attention + VLM LoRA"""
        print("🔥 模式2: 训练Cross Attention + VLM LoRA")
        
        # 🎯 首先启用融合组件 (复用fusion_only逻辑)
        self._setup_fusion_only()
        
        # 🎯 然后配置LoRA
        self._setup_vlm_lora()
        
        print("  ✅ Cross Attention + VLM LoRA -> 可训练")
    
    def _setup_history_fusion(self):
        """模式3: 训练历史特征融合 (VLM + separator)"""
        print("🔥 模式4: 训练历史特征融合")
        self._setup_separator_token_only()
        self._setup_vlm_lora()
        # ✅ 额外启用separator_token训练
        print("  ✅ Spatial_separator_token + VLM LoRA -> 可训练")
    
    def _setup_vlm_lora(self):
        """对 PaliGemma (2B) 和 Gemma Expert (300M) 应用不同的 LoRA"""
        paligemma_model = self.policy.model.paligemma_with_expert
    
        # PaliGemma Language Model (2B) - rank=16
        language_model = paligemma_model.paligemma.language_model
        if not (hasattr(language_model, 'peft_config') and language_model.peft_config):
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=LORA_CONFIG_2B["r"],
                lora_alpha=LORA_CONFIG_2B["lora_alpha"],
                lora_dropout=LORA_CONFIG_2B["lora_dropout"],
                target_modules=LORA_CONFIG_2B["target_modules"]
            )
            paligemma_model.paligemma.language_model = get_peft_model(language_model, lora_config)
            print(f"PaliGemma LoRA (2B): r={LORA_CONFIG_2B['r']}, alpha={LORA_CONFIG_2B['lora_alpha']}")
        else:
            for name, param in language_model.named_parameters():
                if 'lora_' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
    
        # Gemma Expert (300M) - rank=32
        gemma_expert = paligemma_model.gemma_expert
        if not (hasattr(gemma_expert, 'peft_config') and gemma_expert.peft_config):
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=LORA_CONFIG_300M["r"],
                lora_alpha=LORA_CONFIG_300M["lora_alpha"],
                lora_dropout=LORA_CONFIG_300M["lora_dropout"],
                target_modules=LORA_CONFIG_300M["target_modules"]
            )
            paligemma_model.gemma_expert = get_peft_model(gemma_expert, lora_config)
            print(f"Gemma Expert LoRA (300M): r={LORA_CONFIG_300M['r']}, alpha={LORA_CONFIG_300M['lora_alpha']}")
        else:
            for name, param in gemma_expert.named_parameters():
                if 'lora_' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        
    
    def _print_trainable_stats(self):
        """打印可训练参数统计"""
        total = sum(p.numel() for p in self.policy.parameters())
        trainable = sum(p.numel() for p in self.policy.parameters() if p.requires_grad)
        print(f"📊 参数统计: {trainable:,}/{total:,} ({trainable/total*100:.1f}% 可训练)")
    
    def training_step(self, batch, batch_idx):
        """训练步骤 (复用原有逻辑)"""

        loss, loss_dict = self.policy(batch)
        
        # 记录损失
        self.log('train_loss', loss.detach(), prog_bar=True, sync_dist=True)
        
        for key, value in loss_dict.items():
            if isinstance(value, (int, float)):
                self.log(f'train_{key}', value, sync_dist=True)
            elif isinstance(value, torch.Tensor):
                detached_value = value.detach()
                if detached_value.numel() == 1:
                    self.log(f'train_{key}', detached_value.item(), sync_dist=True)
                else:
                    self.log(f'train_{key}', detached_value.mean().item(), sync_dist=True)

        return loss
      
    def configure_optimizers(self):
        """配置优化器 - 统一学习率版本"""
        # 统一学习率，收集所有可训练参数
        trainable_params = [p for p in self.policy.parameters() if p.requires_grad]
        if not trainable_params:
            raise RuntimeError("没有找到可训练参数！")
        
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=1e-10,
            eps=1e-6
        )
        
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = 1000
        
        # 阶段1: Warmup (0 → lr)
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_steps
        )
        
        # 阶段2: 余弦衰减 (lr → 2.5e-6)
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=2.5e-6
        )
        
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        }
    
    def save_components(self, save_dir):
        """保存训练后的组件 (简化版，只保存必要的)"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"💾 保存组件到: {save_dir}")
        
        # 1. 保存完整模型 (复用原有逻辑)
        try:
            save_model(self.policy, save_dir / "model.safetensors")
            print(f"  ✅ model.safetensors")
        except Exception as e:
            print(f"  ❌ model.safetensors 保存失败: {e}")
        
        # 2. 保存配置
        config_dict = self.policy.config.__dict__.copy()
        config_dict['type'] = 'pi0'
        config_dict['training_mode'] = self.training_mode  # 记录训练模式
        
        with open(save_dir / 'config.json', 'w') as f:
            json.dump(config_dict, f, indent=2)
        print(f"  ✅ config.json")
        
        # 3. 保存组件 (使用原有方法)
        try:
            self.policy.model.paligemma_with_expert.save_modular_components(save_dir)
            print(f"  ✅ 融合组件")
        except Exception as e:
            print(f"  ❌ 融合组件保存失败: {e}")
        
        # 4. 保存训练信息
        training_info = {
            "training_mode": self.training_mode,
            "learning_rate": self.learning_rate,
            "lora_config": LORA_CONFIG if self.training_mode == TrainingMode.FUSION_AND_VLM else None,
            "trainable_params": sum(p.numel() for p in self.policy.parameters() if p.requires_grad),
            "total_params": sum(p.numel() for p in self.policy.parameters()),
        }
        
        with open(save_dir / "training_info.json", "w") as f:
            json.dump(training_info, f, indent=2)
        print(f"  ✅ training_info.json")
        
        return save_dir
    
    def save_lora_checkpoint(self, save_dir):
        """保存LoRA checkpoint（只保存adapter，节省空间）"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"💾 保存LoRA checkpoint到: {save_dir}")
        
        paligemma_model = self.policy.model.paligemma_with_expert

        # 1. 保存LoRA adapter（如果存在）
        backbone_has_lora = False    
        language_model = paligemma_model.paligemma.language_model
        gemma_expert = paligemma_model.gemma_expert
        if hasattr(language_model, 'save_pretrained'):  # 是PEFT模型
            language_model.save_pretrained(save_dir / "paligemma_lora_adapter")
            backbone_has_lora = True
        expert_has_lora = False
        if hasattr(gemma_expert, 'save_pretrained'):
            gemma_expert.save_pretrained(save_dir / "gemma_expert_lora_adapter")
            expert_has_lora=True
        
        # 2. 保存其他训练组件（fusion_block等）
        try:
            paligemma_model.save_modular_components(save_dir)
            print(f"  ✅ 其他组件")
        except Exception as e:
            print(f"  ❌ 其他组件保存失败: {e}")
        
        # 3. 保存训练元信息
        checkpoint_info = {
            "training_mode": self.training_mode,
            "learning_rate": self.learning_rate,
            "epoch": self.current_epoch,
            "backbone_has_lora": backbone_has_lora,
            "expert_has_lora":expert_has_lora,
            "checkpoint_type": "lora_only",  # 标记这是LoRA-only checkpoint
            "base_model_needed": True,  # 标记需要原始模型来加载
        }
        
        with open(save_dir / "checkpoint_info.json", "w") as f:
            json.dump(checkpoint_info, f, indent=2)
        print(f"  ✅ checkpoint_info.json")
        
        return save_dir
    
    def save_merged_final_model(self, save_dir):
        """保存合并后的最终模型（完整可用）"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🎯 保存最终合并模型到: {save_dir}")
        
        paligemma_model = self.policy.model.paligemma_with_expert
        
        # 🔥 关键：合并LoRA到原模型
        language_model = paligemma_model.paligemma.language_model
        if hasattr(language_model, 'merge_and_unload'):
            print("合并LoRA参数到主干模型...")
            merged_model = language_model.merge_and_unload()
            paligemma_model.paligemma.language_model = merged_model
            print("主干模型LoRA合并完成")
        gemma_expert = paligemma_model.gemma_expert
        if hasattr(gemma_expert, 'merge_and_unload'):
            print("合并LoRA参数到动作模型...")
            paligemma_model.gemma_expert = gemma_expert.merge_and_unload()
            print("动作模型LoRA合并完成")
        
        # 保存完整模型
        try:
            save_model(self.policy, save_dir / "model.safetensors")
            print(f"  ✅ model.safetensors")
        except Exception as e:
            print(f"  ❌ model.safetensors 保存失败: {e}")
            # 降级到torch保存
            torch.save(self.policy.state_dict(), save_dir / "model.pth")
            print(f"  ✅ model.pth (降级保存)")
        
        # 保存配置
        config_dict = self.policy.config.__dict__.copy()
        config_dict['type'] = 'pi0'
        config_dict['training_mode'] = self.training_mode
        
        with open(save_dir / 'config.json', 'w') as f:
            json.dump(config_dict, f, indent=2)
        print(f"  ✅ config.json")
        
        # 保存其他组件
        try:
            paligemma_model.save_modular_components(save_dir)
            print(f"  ✅ 其他组件")
        except Exception as e:
            print(f"  ❌ 其他组件保存失败: {e}")
        
        # 保存最终模型信息
        final_info = {
            "training_mode": self.training_mode,
            "learning_rate": self.learning_rate,
            "final_epoch": self.current_epoch,
            "checkpoint_type": "merged_final",  # 标记这是合并后的最终模型
            "lora_merged": True,
            "standalone": True,  # 标记可以独立使用
        }
        
        with open(save_dir / "model_info.json", "w") as f:
            json.dump(final_info, f, indent=2)
        print(f"  ✅ model_info.json")
        
        return save_dir


# 🎯 简化的保存回调
class ModelCheckpointCallback(L.Callback):
    """简化的检查点保存回调"""
    
    def __init__(self, save_dir, every_n_epochs=5):
        self.save_dir = Path(save_dir)
        self.every_n_epochs = every_n_epochs
        
    def on_train_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.every_n_epochs == 0:
            epoch_save_dir = self.save_dir / f"epoch_{trainer.current_epoch}"
            # pl_module.save_components(epoch_save_dir)
            pl_module.save_lora_checkpoint(epoch_save_dir)
    
    def on_train_end(self, trainer, pl_module):
        # 最终保存
        final_save_dir = self.save_dir / "final"
        # pl_module.save_components(final_save_dir)
        pl_module.save_merged_final_model(final_save_dir)

class SimpleDataModule(L.LightningDataModule):
    def __init__(self, repo_id, root, spatial_features_dir, debug_episodes,
                 batch_size, num_workers, use_history):
        super().__init__()
        
        # Enhanced_LerobotPI0Dataset 的参数
        self.repo_id = repo_id
        self.root = root
        self.image_size = 224
        self.action_horizon = 50
        self.dataset_fps = 10.0
        self.debug_episodes = debug_episodes
        self.spatial_features_dir = spatial_features_dir
        self.use_history=use_history
        # DataLoader 的参数
        self.batch_size = batch_size
        self.num_workers = num_workers
        
    def setup(self, stage=None):
        self.dataset = Enhanced_LerobotPI0Dataset(
            repo_id=self.repo_id,
            root=self.root,
            image_size=self.image_size,
            action_horizon=self.action_horizon,
            dataset_fps=self.dataset_fps,
            debug_episodes=self.debug_episodes,
            spatial_features_dir=self.spatial_features_dir,
            use_history=self.use_history
        )
        
    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            pin_memory=True,
            collate_fn=enhanced_collate_fn,
        )

def train_with_mode(args):
    """🎯 核心训练函数 - 根据模式训练"""
    print("="*60)
    print(f"🎯 开始训练: {args.mode}")
    print("="*60)
    torch.set_float32_matmul_precision('medium')  # 或 'high' 
    # 🔧 加载基础模型
    print("📂 加载基础模型...")
    config = PreTrainedConfig.from_pretrained(args.base_model_path)
    config.freeze_vision_encoder = True
    config.train_expert_only = True
    
    # 🔧 创建Policy (传入组件路径)
    # policy = PI0Policy(
    #     config,
    #     components_path=args.components_path,
    #     mode="train",
    #     use_spatial_encoder=True,
    #     use_history_features=True,
    #     history_camera_config={
    #         "base_0_rgb": True,
    #         "left_wrist_0_rgb": False,
    #         "right_wrist_0_rgb": False,
    #     },              
    # )
    # 🔥 显存预占用 - 加在这里


    if args.use_history_features:
        history_camera_config = {"base_0_rgb": True, "left_wrist_0_rgb": True, "right_wrist_0_rgb": False}
        use_history=True
    else:
        history_camera_config = {"base_0_rgb": False, "left_wrist_0_rgb": False, "right_wrist_0_rgb": False}
        use_history=False
    policy = PI0Policy(
        config,
        components_path=args.components_path,
        mode="train",
        
        # 🔥 从args获取空间编码配置
        use_spatial_encoder=getattr(args, 'use_spatial_encoder',False),  # 默认True
        # spatial_tower=getattr(args, 'spatial_tower', 'cut3r'),
        # spatial_tower_select_feature=getattr(args, 'spatial_tower_select_feature', 'all'),
        spatial_camera_config=getattr(args, 'spatial_camera_config', {
            "base_0_rgb": True,
            "left_wrist_0_rgb": True,
            "right_wrist_0_rgb": False,
        }),
        
        # 🔥 从args获取历史特征配置
        use_history_features=getattr(args, 'use_history_features', False),  # 默认False
        num_sampled_history_frames=getattr(args, 'num_sampled_history_frames', 5),
        # history_sampling_method=getattr(args, 'history_sampling_method', 'uniform'),
        history_camera_config=history_camera_config,
        
        # 🔥 融合配置
        # fusion_block=getattr(args, 'fusion_block', 'cross_attention'),
    )
    
    # 🔧 创建训练器
    lightning_module = Lerobot_Trainer(
        policy=policy,
        training_mode=args.mode,
        learning_rate=args.learning_rate
    )
    
    # 🔧 准备数据 (复用原有逻辑)
    print("📊 准备数据...")
    datamodule = SimpleDataModule(
        repo_id=args.data_repo_id,
        root=args.data_root,
        spatial_features_dir=args.spatial_features_dir,
        debug_episodes=args.debug_episodes,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_history=use_history
    )
    # dataset = Enhanced_LerobotPI0Dataset(
    #     repo_id=args.data_repo_id,
    #     root=args.data_root,
    #     image_size=224,
    #     action_horizon=50,
    #     dataset_fps=10.0,
    #     debug_episodes=args.debug_episodes,
    #     spatial_features_dir=args.spatial_features_dir,
    # )
    #
    # dataloader = DataLoader(
    #     dataset,
    #     batch_size=args.batch_size,
    #     shuffle=False,
    #     num_workers=args.num_workers,
    #     persistent_workers=True if args.num_workers > 0 else False,
    #     pin_memory=True,
    #     collate_fn=enhanced_collate_fn,
    # )
    
    # 🔧 设置保存和回调
    save_dir = Path(args.output_dir) / args.mode
    save_dir.mkdir(parents=True, exist_ok=True)
    
    callbacks = [
        ModelCheckpoint(
            dirpath=save_dir / "lightning_checkpoints",
            filename="epoch_{epoch}-step_{step}",
            monitor="train_loss",  # 添加这行
            mode="min",           # 添加这行
            save_top_k=3,
            every_n_epochs=args.save_every_n_epochs,
        ),
        ModelCheckpointCallback(
            save_dir=save_dir,
            every_n_epochs=args.save_every_n_epochs
        )
    ]
    
    # 🔧 创建Lightning训练器 (复用原有配置)
    trainer = L.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=args.devices,
        strategy="ddp_find_unused_parameters_true" if args.devices > 1 else "auto",
        max_epochs=args.max_epochs,
        precision=args.precision,
        gradient_clip_val=1.0,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=callbacks,
        enable_progress_bar=True,
        log_every_n_steps=10,
    )

    # trainer.fit(lightning_module, dataloader)
    #trainer.fit(lightning_module, datamodule)
    if args.ckpt_path:
        print(f"从checkpoint恢复训练: {args.ckpt_path}")
        trainer.fit(lightning_module, datamodule, ckpt_path=args.ckpt_path)
    else:
        trainer.fit(lightning_module, datamodule)
    print(f"✅ 训练完成: {args.mode}")
    final_path = save_dir / "final"
    print(f"📁 最终模型: {final_path}")
    
    return final_path


def main():
    parser = argparse.ArgumentParser(description="简化版PI0训练器")
    
    # 🎯 核心参数
    parser.add_argument("--mode", type=str, required=True,
                   choices=[TrainingMode.FUSION_ONLY, TrainingMode.FUSION_AND_VLM, TrainingMode.SPATIAL_SEPARATOR_TOKEN_ONLY,
                           TrainingMode.HISTORY_FUSION],  # ✅ 添加新选项
                   help="训练模式: fusion_only | fusion_and_vlm | token_only | history_fusion")
    parser.add_argument("--use_spatial_encoder", action="store_true", default=False,
                       help="是否使用空间编码器")
    parser.add_argument("--use_history_features", action="store_true", default=False,
                       help="是否使用历史特征")
    parser.add_argument("--num_sampled_history_frames", type=int, default=3,
                       help="历史帧采样数量")
    
    parser.add_argument("--base_model_path", type=str, required=True,
                       help="Pi0基础模型路径")
    parser.add_argument("--components_path", type=str, default=None,
                       help="组件加载路径 (用于加载预训练的fusion组件)")
    parser.add_argument("--ckpt_path", type=str, default=None,
                   help="完整恢复训练状态的checkpoint路径（用于中断续训）")    
    # 数据参数
    parser.add_argument("--data_repo_id", type=str, default=None,
                       help="LeRobot数据集ID") 
    parser.add_argument("--data_root", type=str, default=None,
                       help="数据集根目录")
    parser.add_argument("--spatial_features_dir", type=str, default=None,
                       help="预计算特征目录")
    parser.add_argument("--output_dir", type=str, default="./experiments",
                       help="输出目录")
    
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=16,
                       help="批次大小")
    parser.add_argument("--learning_rate", type=float, default=None,
                       help="学习率 (不指定则使用模式默认值)")
    parser.add_argument("--max_epochs", type=int, default=15,
                       help="训练轮数")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="数据加载器工作进程数")
    parser.add_argument("--devices", type=int, default=1,
                       help="GPU设备数量")
    parser.add_argument("--precision", type=str, default="16-mixed",
                       help="训练精度")
    parser.add_argument("--accumulate_grad_batches", type=int, default=2,
                       help="梯度累积批次数")
    parser.add_argument("--save_every_n_epochs", type=int, default=5,
                       help="每N个epoch保存一次")
    parser.add_argument("--debug_episodes", type=int, default=None,
                       help="调试用：限制样本数量")
    
    args = parser.parse_args()
    
    # 🎯 根据模式设置默认学习率
    if args.learning_rate is None:
        if args.mode == TrainingMode.FUSION_ONLY:
            args.learning_rate = 5e-4  # 融合模块用较大学习率
        elif args.mode == TrainingMode.FUSION_AND_VLM:
            args.learning_rate = 2e-5  # VLM微调用小学习率
        print(f"🎯 使用默认学习率: {args.learning_rate}")
    
    # 数据验证
    if args.data_repo_id == "None":
        args.data_repo_id = None
    
    # 🎯 组件路径提示
    if args.components_path is not None:
        components_path = Path(args.components_path)
        if components_path.exists():
            fusion_file = components_path / 'fusion_block.pth'
            if fusion_file.exists():
                print(f"🔍 将加载预训练融合组件: {fusion_file}")
            else:
                print(f"⚠️  未找到fusion_block.pth，将使用随机初始化")
        else:
            print(f"⚠️  组件路径不存在: {components_path}")
    else:
        print("ℹ️  未指定组件路径，使用随机初始化")
    
    # 🚀 开始训练
    final_path = train_with_mode(args)
    
    # 🎯 训练完成提示
    print("\n" + "="*60)
    print("🎉 训练完成!")
    print(f"📁 模型路径: {final_path}")
    if args.mode == TrainingMode.FUSION_ONLY:
        print("💡 下一步: 可以用 --mode fusion_and_vlm 加载这个fusion组件进行联合训练")
        print(f"   python simple_pi0_trainer.py --mode fusion_and_vlm --components_path {final_path}")
    print("="*60)


if __name__ == "__main__":
    main()
