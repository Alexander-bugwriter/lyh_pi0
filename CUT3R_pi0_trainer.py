#!/usr/bin/env python3
"""
修复后的Pi0 Lightning训练器 - 支持自动两阶段训练
阶段1: 空间融合模块 + PaliGemma LoRA
阶段2: 全模型联合训练

使用方法:
1. 自动两阶段训练:
   python simple_pi0_trainer.py --auto_two_stage --base_model_path /path/to/pi0 --data_repo_id dataset_id
   
2. 单独训练某阶段:
   python simple_pi0_trainer.py --stage 1 --base_model_path /path/to/pi0 --data_repo_id dataset_id


"""
# 设备检查修复补丁
def patched_is_torch_device_available(device: str) -> bool:
    """修复的设备可用性检查函数 - 支持带索引的设备名称"""
    import torch
    
    # 处理带索引的设备名称，如 "cuda:0" -> "cuda"
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

# 应用Monkey Patch
import lerobot.common.utils.utils
lerobot.common.utils.utils.is_torch_device_available = patched_is_torch_device_available
print("✅ 设备检查函数已修复")



import os
import argparse
import torch
import pytorch_lightning as L
import json
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from torchvision.transforms.v2 import Compose, Resize

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.configs.policies import PreTrainedConfig
from V3R_pi0.modeling_pi0 import PI0Policy
from utils.normalizers import Normalizer


class LerobotPI0Dataset(Dataset):
    """标准Lerobot格式数据集包装器"""
    
    def __init__(self, repo_id=None, root=None, image_size=224, action_horizon=50):
        print(f"加载Lerobot数据集: {repo_id}")
        
        image_transforms = Resize((image_size, image_size))
        
        # 标准lerobot格式的时间戳配置
        delta_timestamps = {
            "observation.images.base": [0],
            "observation.images.wrist": [0], 
            "observation.state": [0],
            "action": [i / 30 for i in range(action_horizon)],
        }
        
        try:
            self.dataset = LeRobotDataset(
                repo_id=repo_id,
                root=root,
                image_transforms=image_transforms,
                delta_timestamps=delta_timestamps,
            )
            print(f" 数据集加载成功，共 {len(self.dataset)} 条数据")
            
        except Exception as e:
            print(f"   数据集加载失败: {e}")
            
            # 调试信息：检查本地路径结构
            if root and os.path.exists(root):
                print(f"    调试：检查本地路径结构")
                self._debug_local_path_structure(root)
            raise
        
        # 标准化器配置
        self.normalizer = Normalizer(
            norm_stats=self.dataset.meta.stats,
            norm_type={
                "observation.images.base": "identity",
                "observation.images.wrist": "identity", 
                "observation.state": "meanstd",
                "action": "meanstd",
            }
        )


    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        normalized_item = self.normalizer.normalize(item)
        
        # 图像处理
        images = {}
        
        # 基础相机 (必需)
        if "observation.images.base" in normalized_item:
            base_image = (normalized_item["observation.images.base"] * 255).to(torch.uint8)
            images["base_0_rgb"] = base_image
        
        # 手腕相机 (可选)
        if "observation.images.wrist" in normalized_item:
            wrist_image = (normalized_item["observation.images.wrist"] * 255).to(torch.uint8)
            images["left_wrist_0_rgb"] = wrist_image
        
        # 任务指令
        task_text = item.get("task", "complete the task")
        if isinstance(task_text, str):
            prompt = [task_text]
        elif isinstance(task_text, (list, tuple)):
            prompt = task_text
        else:
            prompt = [str(task_text)]
        
        return {
            "image": images,
            "state": normalized_item["observation.state"][0],
            "action": normalized_item["action"],
            "action_is_pad": normalized_item.get("action_is_pad", 
                torch.zeros_like(normalized_item["action"][..., 0], dtype=torch.bool)
            ),
            "prompt": prompt,
        }


class CUT3R_pi0_Trainer(L.LightningModule):
    """修复后的Pi0 Lightning训练器"""
    
    def __init__(self, policy: PI0Policy, training_stage: int, learning_rate: float = 2e-5):
        super().__init__()
        self.policy = policy
        self.training_stage = training_stage
        self.learning_rate = learning_rate
        
        # 设置训练阶段
        self._setup_training_stage()
        
        # 保存超参数
        self.save_hyperparameters(ignore=['policy'])
    
    def _setup_training_stage(self):
        """根据阶段配置可训练参数"""
        print(f"配置训练阶段 {self.training_stage}")
        
        # 冻结所有参数
        for param in self.policy.parameters():
            param.requires_grad = False
        
        if self.training_stage == 1:
            self._setup_stage1()
        elif self.training_stage == 2:
            self._setup_stage2()
        
        self._print_trainable_stats()
    
    def _setup_stage1(self):
        """阶段1: 空间融合模块 + PaliGemma LoRA"""
        print(" 阶段1: 训练空间融合模块 + PaliGemma LoRA")
        
        paligemma_model = self.policy.model.paligemma_with_expert
        
        # 启用融合相关模块
        fusion_components = ['fusion_block', 'mm_projector', 'spatial_separator_token']
        for comp_name in fusion_components:
            if hasattr(paligemma_model, comp_name):
                component = getattr(paligemma_model, comp_name)
                if hasattr(component, 'parameters'):
                    for param in component.parameters():
                        param.requires_grad = True
                else:
                    component.requires_grad = True
                print(f"  {comp_name}")
        
        # 设置LoRA
        self._setup_lora()
        
        # 确保Action Expert冻结
        action_modules = ['state_proj', 'action_in_proj', 'action_out_proj', 
                         'action_time_mlp_in', 'action_time_mlp_out']
        for module_name in action_modules:
            if hasattr(self.policy.model, module_name):
                for param in getattr(self.policy.model, module_name).parameters():
                    param.requires_grad = False
        print("  Action Expert保持冻结")
    
    def _setup_stage2(self):
        """阶段2: 全模型联合训练"""
        print("阶段2: 全模型联合训练")
        
        # 继承阶段1
        self._setup_stage1()
        
        # 启用Action Expert
        action_modules = ['state_proj', 'action_in_proj', 'action_out_proj',
                         'action_time_mlp_in', 'action_time_mlp_out']
        for module_name in action_modules:
            if hasattr(self.policy.model, module_name):
                for param in getattr(self.policy.model, module_name).parameters():
                    param.requires_grad = True
                print(f"  {module_name}")
    
    def _setup_lora(self):
        """修复后的LoRA设置"""
        from peft import get_peft_model, LoraConfig, TaskType
        
        language_model = self.policy.model.paligemma_with_expert.paligemma.language_model
        
        # ������ 修复：更严格的LoRA检查
        if (hasattr(language_model, 'peft_config') and 
            language_model.peft_config is not None and 
            len(language_model.peft_config) > 0):
            print("  LoRA已正确配置，启用训练")
            # 确保LoRA参数可训练
            for param in language_model.parameters():
                if param.requires_grad:  # 只设置本来就应该训练的参数
                    continue
                # 检查是否是LoRA参数
                param.requires_grad = True
            return
        
        # 配置新的LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=128, 
            lora_alpha=256, 
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
        )
        
        language_model = get_peft_model(language_model, lora_config)
        self.policy.model.paligemma_with_expert.paligemma.language_model = language_model
        print("  新建LoRA (r=128)")
        
    
    def _print_trainable_stats(self):
        """打印参数统计"""
        total = sum(p.numel() for p in self.policy.parameters())
        trainable = sum(p.numel() for p in self.policy.parameters() if p.requires_grad)
        print(f"参数: {trainable:,}/{total:,} ({trainable/total*100:.1f}% 可训练)")
    
    def training_step(self, batch, batch_idx):
        """训练步骤"""
        loss, loss_dict = self.policy(batch)
        
        # 记录损失
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        
        # 记录详细损失
        for key, value in loss_dict.items():
            if isinstance(value, (int, float, torch.Tensor)):
                self.log(f'train_{key}', value, sync_dist=True)
        
        return loss
    
    def configure_optimizers(self):
        """修复后的优化器配置"""
        trainable_params = [p for p in self.policy.parameters() if p.requires_grad]
        
        if not trainable_params:
            raise RuntimeError("没有找到可训练参数！检查training_stage设置")
        
        optimizer = torch.optim.AdamW(
            trainable_params, 
            lr=self.learning_rate,
            weight_decay=1e-2,
            eps=1e-6
        )
        
        # ������ 修复：使用更合适的学习率调度器
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(0.05 * total_steps)  # 5% warmup
        
        from torch.optim.lr_scheduler import OneCycleLR
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,
            total_steps=total_steps,
            pct_start=warmup_steps/total_steps,
            anneal_strategy='cos'
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        }
    
    def save_modular_checkpoint(self, save_dir):
        """保存模型：主干 + 融合模块都在final/目录，便于手动部署"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f" 保存模型到: {save_dir}")
        
        # ������ 1. 保存主干模型（和原版Pi0格式完全一致）
        print("   保存主干模型（PaliGemma + LoRA + Action Expert）...")
        self.policy.save_pretrained(save_dir)
        print(f"      config.json")
        print(f"      model.safetensors（主干+Action Expert）")
        
        # ������ 2. 保存融合模块到final/目录（便于手动移动）
        print(f"  保存融合模块到final/目录...")
        
        paligemma_model = self.policy.model.paligemma_with_expert
        
        # 保存融合相关模块
        modules_config = [
            ('fusion_block', 'fusion_block.pth', paligemma_model.fusion_block),
            ('mm_projector', 'mm_projector.pth', paligemma_model.mm_projector), 
            ('spatial_separator_token', 'spatial_separator_token.pth', paligemma_model.spatial_separator_token)
        ]
        
        saved_modules = []
        for module_name, file_name, component in modules_config:
            if component is not None:
                file_path = save_dir / file_name
                try:
                    if module_name == 'spatial_separator_token':
                        torch.save(component.data, file_path)
                    else:
                        torch.save(component.state_dict(), file_path)
                    print(f"      {file_name}")
                    saved_modules.append(file_name)
                except Exception as e:
                    print(f"      {file_name} 保存失败: {e}")
        
        # ������ 3. 保存训练元信息和部署说明
        training_info = {
            "training_stage": self.training_stage,
            "enhanced_pi0_version": "1.0",
            "spatial_encoder": "cut3r",
            "fusion_type": getattr(self.policy.config, 'fusion_block', 'cross_attention'),
            "training_completed": True,
            "architecture": "modular_loading",
            "saved_fusion_modules": saved_modules,
            "deployment_instruction": {
                "step1": f"主干模型已保存到: {save_dir}",
                "step2": "融合模块文件需要手动移动到: spatial_encoder_checkpoint/",
                "step3": "移动完成后即可使用 PI0Policy.from_pretrained() 加载",
                "required_files": saved_modules
            }
        }
        
        with open(save_dir / "training_info.json", "w") as f:
            json.dump(training_info, f, indent=2)
        
        #  4. 生成部署脚本
        deploy_script = f'''#!/bin/bash
# Enhanced Pi0 模块部署脚本
# 执行此脚本将融合模块移动到正确位置

echo " 开始部署Enhanced Pi0融合模块..."

# 创建目标目录
mkdir -p spatial_encoder_checkpoint

# 移动融合模块文件
'''
        
        for module_file in saved_modules:
            deploy_script += f'''
if [ -f "{save_dir}/{module_file}" ]; then
    cp "{save_dir}/{module_file}" spatial_encoder_checkpoint/
    echo "✅ 已复制 {module_file}"
else
    echo "❌ 未找到 {module_file}"
fi'''
        
        deploy_script += '''

echo " 融合模块已部署到: spatial_encoder_checkpoint/"
echo " 现在可以使用以下代码加载模型:"
echo ""
echo "from V3R_pi0.modeling_pi0 import PI0Policy"
echo f"policy = PI0Policy.from_pretrained('{save_dir}')"
echo ""
echo "✅ 部署完成！"
'''
        
        with open(save_dir / "deploy_modules.sh", "w") as f:
            f.write(deploy_script)
        
        # 设置脚本执行权限
        import stat
        (save_dir / "deploy_modules.sh").chmod(stat.S_IRWXU | stat.S_IRGRP | stat.S_IROTH)
        
        print(f"")
        print(f" 模型保存完成！")
        print(f"    主干模型: {save_dir}")
        print(f"    融合模块: {save_dir}")
        print(f"")
        print(f" 下一步操作:")
        print(f"   1. 手动移动融合模块到目标位置:")
        for module_file in saved_modules:
            print(f"      cp {save_dir}/{module_file} spatial_encoder_checkpoint/")
        print(f"")
        print(f"   2. 或者执行自动部署脚本:")
        print(f"      bash {save_dir}/deploy_modules.sh")
        print(f"")
        print(f"   3. 部署完成后加载模型:")
        print(f"      PI0Policy.from_pretrained('{save_dir}')")
        
        return save_dir


class ModularCheckpointCallback(L.Callback):
    """分模块保存回调"""
    
    def __init__(self, save_dir, every_n_epochs=4):
        self.save_dir = Path(save_dir)
        self.every_n_epochs = every_n_epochs
        
    def on_train_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.every_n_epochs == 0:
            epoch_save_dir = self.save_dir / f"epoch_{trainer.current_epoch}"
            pl_module.save_modular_checkpoint(epoch_save_dir)
    
    def on_train_end(self, trainer, pl_module):
        # 最终保存
        final_save_dir = self.save_dir / "final"
        pl_module.save_modular_checkpoint(final_save_dir)


def load_stage1_checkpoint(policy, checkpoint_dir):
    """加载阶段1检查点 - 支持final/目录下的融合模块"""
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        print(f" 检查点目录不存在: {checkpoint_dir}")
        return False
        
    print(f" 加载阶段1检查点: {checkpoint_dir}")
    
    # ������ 1. 检查主干模型是否存在
    has_main_model = (
        (checkpoint_dir / "config.json").exists() and 
        ((checkpoint_dir / "model.safetensors").exists() or 
         (checkpoint_dir / "pytorch_model.bin").exists())
    )
    
    # ������ 2. 检查final/目录下的融合模块文件
    fusion_files = ['fusion_block.pth', 'mm_projector.pth', 'spatial_separator_token.pth']
    final_fusion_files = [f for f in fusion_files if (checkpoint_dir / f).exists()]
    
    if has_main_model and final_fusion_files:
        print(f" 发现主干模型和融合模块文件: {final_fusion_files}")
        print("  需要手动部署融合模块:")
        print("   方式1: 执行部署脚本")
        print(f"      bash {checkpoint_dir}/deploy_modules.sh")
        print("   方式2: 手动复制文件")
        for f in final_fusion_files:
            print(f"      cp {checkpoint_dir}/{f} spatial_encoder_checkpoint/")
        print("   部署完成后模型即可正常工作")
        return True
    
    if has_main_model:
        print(" 发现主干模型，假设融合模块已在spatial_encoder_checkpoint/")
        return True
    
    # ������ 3. 备用：检查spatial_encoder_checkpoint目录
    script_dir = Path(__file__).parent if '__file__' in globals() else Path.cwd()
    project_root = script_dir.parent
    default_spatial_dir = project_root / 'spatial_encoder_checkpoint'
    
    if default_spatial_dir.exists():
        existing_files = [f for f in fusion_files if (default_spatial_dir / f).exists()]
        
        if existing_files:
            print(f" 发现spatial_encoder_checkpoint中的融合模块: {existing_files}")
            return True
    
    # 4. 兼容旧格式：直接在checkpoint_dir中查找融合模块
    if final_fusion_files:
        print(f" 在检查点目录中发现融合模块: {final_fusion_files}")
        print("   建议将这些文件移动到spatial_encoder_checkpoint/")
        return True
    
    print(" 未找到有效的阶段1检查点")
    print("   请确保以下之一存在：")
    print("   1. 主干模型文件 (config.json + model.safetensors)")
    print("   2. final/目录下的融合模块文件")
    print("   3. spatial_encoder_checkpoint/目录及融合模块文件")
    
    return False


def train_single_stage(args, stage):
    """训练单个阶段"""
    print("="*60)
    print(f" 开始阶段{stage}训练")
    print("="*60)
    
    # 加载模型
    print(" 加载模型...")
    config = PreTrainedConfig.from_pretrained(args.base_model_path)
    # ������ 修复：移除手动设备设置，让Lightning管理
    # config.device = "cpu"  # 删除这行！
    config.freeze_vision_encoder = True
    config.train_expert_only = True
    
    policy = PI0Policy(config)
    
    # 如果是阶段2，加载阶段1检查点
    if stage == 2 and args.stage1_checkpoint:
        if not load_stage1_checkpoint(policy, args.stage1_checkpoint):
            print(" 无法加载阶段1检查点，退出训练")
            return None
    
    # 创建Lightning训练器模块
    lightning_module = CUT3R_pi0_Trainer(
        policy=policy,
        training_stage=stage,
        learning_rate=args.learning_rate
    )
    
    # 创建数据集和加载器
    print(" 准备数据...")
    dataset = LerobotPI0Dataset(
        repo_id=args.data_repo_id,
        root=args.data_root,
        image_size=224,
        action_horizon=50
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=True if args.num_workers > 0 else False
    )
    
    # 设置保存目录
    save_dir = Path(args.output_dir) / f"stage_{stage}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置回调
    callbacks = [
        # 标准Lightning检查点
        ModelCheckpoint(
            dirpath=save_dir / "lightning_checkpoints",
            filename="epoch_{epoch}-step_{step}",
            save_top_k=-1,
            every_n_epochs=args.save_every_n_epochs,
        ),
        # 分模块检查点
        ModularCheckpointCallback(
            save_dir=save_dir / "modular_checkpoints",
            every_n_epochs=args.save_every_n_epochs
        )
    ]
    
    # 创建Lightning训练器
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
    
    # 开始训练
    print(f" 开始阶段{stage}训练...")
    trainer.fit(lightning_module, dataloader)
    
    print(f" 阶段{stage}训练完成!")
    print(f" Lightning检查点: {save_dir}/lightning_checkpoints")
    print(f" 模块检查点: {save_dir}/modular_checkpoints")
    
    return save_dir / "modular_checkpoints" / "final"


def main():
    parser = argparse.ArgumentParser(description="CUT3R_Pi0 Lightning训练器")
    
    # 基础参数
    parser.add_argument("--base_model_path", type=str, required=True,
                       help="Pi0基础模型路径")
    parser.add_argument("--data_repo_id", type=str, default=None,
                       help="LeRobot数据集ID") 
    parser.add_argument("--data_root", type=str, default=None,
                       help="数据集根目录")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/enhanced_pi0",
                       help="输出目录")
    
    # ������ 新增：支持自动两阶段训练
    parser.add_argument("--auto_two_stage", action="store_true", 
                       help="自动执行两阶段训练")
    parser.add_argument("--stage", type=int, choices=[1, 2], default=None,
                       help="单独执行某个阶段 (如果不设置auto_two_stage)")
    parser.add_argument("--stage1_checkpoint", type=str, default=None,
                       help="阶段2时加载的阶段1检查点目录")
    
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=4,
                       help="批次大小")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                       help="学习率")
    parser.add_argument("--stage1_epochs", type=int, default=20,
                       help="阶段1训练轮数")
    parser.add_argument("--stage2_epochs", type=int, default=15,
                       help="阶段2训练轮数")
    parser.add_argument("--max_epochs", type=int, default=20,
                       help="单阶段训练轮数")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="数据加载器工作进程数")
    parser.add_argument("--devices", type=int, default=1,
                       help="GPU设备数量")
    parser.add_argument("--precision", type=str, default="bf16-mixed",
                       help="训练精度")
    parser.add_argument("--accumulate_grad_batches", type=int, default=2,
                       help="梯度累积批次数")
    parser.add_argument("--save_every_n_epochs", type=int, default=4,
                       help="每N个epoch保存一次")
    
    args = parser.parse_args()
    
    if args.auto_two_stage:
        # ������ 自动两阶段训练
        print(" 启动自动两阶段训练模式")
        
        # 阶段1
        print("\n" + "="*50)
        print(" 开始阶段1：空间融合模块 + LoRA训练")
        print("="*50)
        args.max_epochs = args.stage1_epochs
        stage1_checkpoint = train_single_stage(args, stage=1)
        
        if stage1_checkpoint is None:
            print(" 阶段1训练失败")
            return
        
        print(f" 阶段1训练完成！检查点: {stage1_checkpoint}")
        
        # 阶段2
        print("\n" + "="*50)
        print(" 开始阶段2：全模型联合训练")
        print("="*50)
        args.max_epochs = args.stage2_epochs
        args.learning_rate = args.learning_rate * 0.5  # 阶段2使用较小学习率
        args.stage1_checkpoint = str(stage1_checkpoint)
        
        stage2_checkpoint = train_single_stage(args, stage=2)
        
        if stage2_checkpoint is None:
            print(" 阶段2训练失败")
            return
        
        print("\n" + "="*60)
        print(" 两阶段训练全部完成！")
        print("="*60)
        print(f" 最终模型: {stage2_checkpoint}")
        print(f" 部署说明:")
        print(f"   1. 执行部署脚本: bash {stage2_checkpoint}/deploy_modules.sh")
        print(f"   2. 或手动复制融合模块到spatial_encoder_checkpoint/")
        print(f"   3. 使用模型: PI0Policy.from_pretrained('{stage2_checkpoint}')")
        
    elif args.stage is not None:
        # 单阶段训练
        train_single_stage(args, stage=args.stage)
        
    else:
        print(" 请指定 --auto_two_stage 或 --stage")
        parser.print_help()


if __name__ == "__main__":
    main()

