#!/usr/bin/env python3
"""
VLA服务器脚本 - 通过WebSocket提供端到端推理服务
"""

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

# 设置环境变量
os.environ["LEROBOT_DEVICE"] = "cuda" if torch.cuda.is_available() else "cpu"

# 模型路径配置
PATH_TO_PI_MODEL = "/opt/liblibai-models/user-workspace2/users/lyh/model_checkpoint/pi0/pytorch/pi0_libero"
PATH_TO_PI_FAST_MODEL = "/opt/liblibai-models/user-workspace2/users/lyh/model_checkpoint/pi0/pytorch/pi0_fast_base"
PATH_TO_JAX_PI_MODEL = "/opt/liblibai-models/user-workspace2/users/lyh/model_checkpoint/pi0/jax/pi0_libero/pi0_libero"  # 归一化参数路径
def load_normalization_stats():
    """加载归一化参数"""
    norm_stats_path = Path(PATH_TO_JAX_PI_MODEL) / "assets/physical-intelligence/libero/norm_stats.json"

    if not norm_stats_path.exists():
        print(f"⚠️  警告: 归一化参数文件不存在: {norm_stats_path}")
    with open(norm_stats_path) as f:
        norm_stats = json.load(f)

    return {
        'state_mean': np.array(norm_stats["norm_stats"]["state"]["mean"][:8], dtype=np.float32),
        'state_std': np.array(norm_stats["norm_stats"]["state"]["std"][:8], dtype=np.float32),
        'action_mean': np.array(norm_stats["norm_stats"]["actions"]["mean"][:7], dtype=np.float32),
        'action_std': np.array(norm_stats["norm_stats"]["actions"]["std"][:7], dtype=np.float32)
    }
def setup_device_patch():
    """修复设备检测函数"""
    def patched_is_torch_device_available(device: str) -> bool:
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
    
    import lerobot.common.utils.utils
    lerobot.common.utils.utils.is_torch_device_available = patched_is_torch_device_available

def load_model(model_type="pi0"):
    """加载模型"""
    setup_device_patch()
    
    from pi0 import PI0Policy, PI0FASTPolicy
    
    print(f" 正在加载 {model_type} 模型...")
    try:
        if model_type == "pi0":
            policy = PI0Policy.from_pretrained(PATH_TO_PI_MODEL)
        else:
            policy = PI0FASTPolicy.from_pretrained(PATH_TO_PI_FAST_MODEL)
        
        print("✅ 模型加载完成")
        return policy
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        raise

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
    policy = load_model(model_type)
    device = policy.config.device
    norm_stats = load_normalization_stats()
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
        except Exception as e:
            print(f"❌ 处理请求时发生错误: {e}")
            continue

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="VLA推理服务器")
    parser.add_argument("--host", default="0.0.0.0", help="服务器地址")
    parser.add_argument("--port", type=int, default=8000, help="服务器端口")
    parser.add_argument("--model", choices=["pi0", "pi0fast"], default="pi0", help="模型类型")
    
    args = parser.parse_args()
    
    print("启动VLA推理服务器")
    print(f"地址: {args.host}:{args.port}")
    print(f"模型: {args.model}")
    print("-" * 50)
    
    run_server(host=args.host, port=args.port, model_type=args.model)
