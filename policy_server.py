#!/usr/bin/env python3
"""
VLA服务器脚本 - 通过WebSocket提供端到端推理服务
"""
import os
import sys

# 🔥 关键: 在所有导入之前就修复设备检测函数
def setup_device_patch():
    """修复设备检测函数 - 必须在导入lerobot之前调用"""
    def patched_is_torch_device_available(device: str) -> bool:
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
            raise ValueError(f"Unknown device {try_device}")
    
    # 🔥 立即应用补丁
    import lerobot.common.utils.utils
    lerobot.common.utils.utils.is_torch_device_available = patched_is_torch_device_available
    print("设备检测函数已修复")

# 🔥 在这里立即调用,在导入其他模块之前
setup_device_patch()

import os
import time
import torch
import numpy as np
from websocket_tool.websocket_server_tool import (
    start_websocket_server, 
    wait_for_observation, 
    send_action_response, 
    wait_for_client_connection
)
import argparse
from pathlib import Path
import json
from V3R_pi0.modeling_pi0 import PI0Policy
from lerobot.configs.policies import PreTrainedConfig

# 设置环境变量
os.environ["LEROBOT_DEVICE"] = "cuda" if torch.cuda.is_available() else "cpu"
from utils.spatiotemporal_lerobot_dataset import LerobotPI0Dataset

def load_normalization_stats_from_dataset(dataset_path):
    """
    🔥 从数据集快速加载归一化参数（使用缓存）
    """
    print(f"从数据集加载归一化参数: {dataset_path}")
    
    #只需要 root 参数，debug_episodes=1 快速加载
    dataset = LerobotPI0Dataset(
            root=dataset_path,
            debug_episodes=1  # 只加载1个episode，够获取meta就行
    )
        
        # 🔥 从 dataset.meta.stats 获取归一化参数
    stats = dataset.dataset.meta.stats
        
    print(f"归一化参数加载成功")    
        # 转换为 numpy 格式
    norm_stats = {
        'state_mean': np.array(stats["state"]["mean"], dtype=np.float32),
        'state_std': np.array(stats["state"]["std"], dtype=np.float32),
        'action_mean': np.array(stats["actions"]["mean"], dtype=np.float32),
        'action_std': np.array(stats["actions"]["std"], dtype=np.float32)
    }
        
    print(f"   State: mean={norm_stats['state_mean'][:3]}, std={norm_stats['state_std'][:3]}")
    print(f"   Action: mean={norm_stats['action_mean'][:3]}, std={norm_stats['action_std'][:3]}")
        
    return norm_stats
        
def convert_observation(raw_data: dict, device, norm_stats) -> dict:
    """
    转换原始数据为模型期望的格式
    输入: {"observation/image": np.array, "observation/state": np.array, "prompt": str, "reset": bool}
    输出: {"image": {"base_0_rgb": tensor}, "state": tensor, "prompt": [str]}
    """
    observation = {
        "image": {},
        "state": None,
        "prompt": [""]
    }
    
    # 处理主摄像头图像
    if "observation/image" in raw_data:
        img = raw_data["observation/image"]
        if isinstance(img, np.ndarray) and len(img.shape) == 3:
            img_tensor = torch.from_numpy(img.copy()).permute(2, 0, 1).unsqueeze(0)
            observation["image"]["base_0_rgb"] = img_tensor.to(dtype=torch.uint8, device=device)
    
    # 处理手腕摄像头图像
    if "observation/wrist_image" in raw_data:
        wrist = raw_data["observation/wrist_image"]
        if isinstance(wrist, np.ndarray) and len(wrist.shape) == 3:
            wrist_tensor = torch.from_numpy(wrist.copy()).permute(2, 0, 1).unsqueeze(0)
            observation["image"]["left_wrist_0_rgb"] = wrist_tensor.to(dtype=torch.uint8, device=device)
    
    # 处理状态数据
    if "observation/state" in raw_data:
        state = raw_data["observation/state"]
        if isinstance(state, np.ndarray):
            normalized_state = (state - norm_stats['state_mean']) / (norm_stats['state_std'] + 1e-6)
            state_tensor = torch.from_numpy(normalized_state.copy()).unsqueeze(0)
            observation["state"] = state_tensor.to(dtype=torch.float32, device=device)
    
    # 处理prompt
    if "prompt" in raw_data:
        prompt = raw_data["prompt"]
        observation["prompt"] = [prompt] if isinstance(prompt, str) else prompt
    
    return observation

def denormalize_action(action, norm_stats, current_state):
    """动作反归一化"""
    # action: [batch, time, 7] 的tensor
    # current_state: [8] 的numpy array (未归一化的状态)
    
    if hasattr(action, 'cpu'):
        action_np = action.cpu().numpy()
    else:
        action_np = np.array(action)

    # 反归一化动作
    denorm_action = action_np * (norm_stats['action_std'] + 1e-6) + norm_stats['action_mean']
        
    # 增量控制：前6维加上当前状态的前6维
    denorm_action[:, :6] += current_state[None, :6]

    return denorm_action

def run_server(host="0.0.0.0", port=8000, model_type="pi0"):
    """运行VLA推理服务器"""
    
    # 加载模型
    # policy = PI0Policy.from_pretrained()
    config_path = Path(args.base_model_path) / "config.json"

    if config_path.exists():
        print(f"检查并修复config文件: {config_path}")

        # 读取原始config
        with open(config_path, 'r') as f:
            config_dict = json.load(f)

        # 移除训练特定字段
        fields_to_remove = ['training_mode', 'pretrained_path']
        removed_fields = []

        for field in fields_to_remove:
            if field in config_dict:
                config_dict.pop(field)
                removed_fields.append(field)

        if removed_fields:
            print(f"移除非标准字段: {removed_fields}")

            # 备份原文件
            backup_path = config_path.with_suffix('.json.backup')
            if not backup_path.exists():
                import shutil
                shutil.copy(config_path, backup_path)
                print(f"原文件已备份到: {backup_path}")

            # 写回修复后的config
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
            print(f"config.json 已修复")

    # 现在可以正常加载了
    print("📂 加载基础模型...")
    config = PreTrainedConfig.from_pretrained(args.base_model_path)
    

    if args.use_history_features:
        history_camera_config = {"base_0_rgb": True, "left_wrist_0_rgb": True, "right_wrist_0_rgb": False}
    else:
        history_camera_config = {"base_0_rgb": False, "left_wrist_0_rgb": False, "right_wrist_0_rgb": False}
    policy = PI0Policy(
        config,
        components_path=args.components_path,
        mode="infer",
        
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
        num_sampled_history_frames=getattr(args, 'num_sampled_history_frames', 3),
        # history_sampling_method=getattr(args, 'history_sampling_method', 'uniform'),
        history_camera_config=history_camera_config,
        
        # 🔥 融合配置
        # fusion_block=getattr(args, 'fusion_block', 'cross_attention'),
    )
    #device = policy.config.device
    device="cuda:0"
    #norm_stats = load_normalization_stats()
    norm_stats = load_normalization_stats_from_dataset(args.dataset_path)
    # 启动WebSocket服务器
    print(f"启动服务器 {host}:{port}")
    start_websocket_server(host=host, port=port, device=device)
    
    print("⏳ 等待客户端连接...")
    wait_for_client_connection()
    print("✅ 客户端已连接，开始处理请求...")
    
    # 推理循环
    episode_count = 0
    step_count = 0
    
    while True:
        try:
            # 接收观测数据
            raw_data = wait_for_observation()
            
            # 检查是否需要重置
            need_reset = raw_data.get("reset", False)
            
            if need_reset:
                episode_count += 1
                step_count = 0
                print(f"RESET - 开始第 {episode_count} 个episode")
                if hasattr(policy, 'reset'):
                    policy.reset()
                    print("模型内部状态已重置（包括CUT3R和历史缓存）")
                # 这里可以添加模型重置逻辑
                # policy.reset() if hasattr(policy, 'reset') else None
            
            step_count += 1
            raw_state = raw_data.get("observation/state", np.zeros(8))
            # 转换数据格式
            observation = convert_observation(raw_data, device, norm_stats)
            
            # 执行推理
            start_time = time.perf_counter()
            action = policy.select_action(observation)[0, :, :7]  # 取前7个动作维度
            end_time = time.perf_counter()
            denormalized_action = denormalize_action(action, norm_stats, raw_state)
            # 发送动作响应
           # send_action_response(action)
            send_action_response(denormalized_action)
            
            # 日志输出
            inference_time = (end_time - start_time) * 1000
            prompt_text = observation['prompt'][0][:50] + "..." if len(observation['prompt'][0]) > 50 else observation['prompt'][0]
            
            status = "RESET" if need_reset else "STEP"
            print(f"[{status}] E{episode_count:03d}-S{step_count:03d} | "
                  f"推理时间: {inference_time:.1f}ms | Prompt: {prompt_text}")
                  
        except KeyboardInterrupt:
            print("\n收到中断信号，正在关闭服务器...")
            break

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="VLA推理服务器")
    parser.add_argument("--host", default="0.0.0.0", help="服务器地址")
    parser.add_argument("--port", type=int, default=8000, help="服务器端口")
    parser.add_argument("--use_spatial_encoder", action="store_true", default=False,
                       help="是否使用空间编码器")
    parser.add_argument("--use_history_features", action="store_true", default=False,
                       help="是否使用历史特征")
    parser.add_argument("--num_sampled_history_frames", type=int, default=3,
                       help="历史帧采样数量")
    parser.add_argument("--base_model_path", type=str, required=True,
                       help="基座模型路径")
    parser.add_argument("--components_path", type=str, default=None,
                       help="组件加载路径 (用于加载预训练的增强组件)")
    parser.add_argument("--dataset_path", type=str, required=True,
                       help="数据集路径（用于加载归一化参数）")
    args = parser.parse_args()
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
    
    print("启动VLA推理服务器")
    print(f"地址: {args.host}:{args.port}")
    print(f"使用空间编码器: {args.use_spatial_encoder}")  # 🔥 修复
    print(f"使用历史特征: {args.use_history_features}")    # 🔥 修复
    print("-" * 50)

    run_server(host=args.host, port=args.port, model_type="CUT3R-pi0")  # 🔥 修复
