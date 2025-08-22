import torch
import os
import draccus
import time
import numpy as np
from draccus.parsers import encoding
from enum import Enum
from websocket_tool.websocket_server_tool import (
    start_websocket_server, 
    wait_for_observation, 
    send_action_response, 
    wait_for_client_connection
)
# 定义枚举（从lerobot库复制）
class NormalizationMode(Enum):
    IDENTITY = "identity"
    MEAN_STD = "mean_std" 
    MIN_MAX = "min_max"

# 注册枚举编码器
@encoding.encode.register
def encode_normalization_mode(obj: NormalizationMode, declared_type):
    return obj.value

# Monkey patch the device availability function before importing lerobot
def patched_is_torch_device_available(device: str) -> bool:
    """Check if a torch device is available."""
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

# Apply the monkey patch
import lerobot.common.utils.utils
lerobot.common.utils.utils.is_torch_device_available = patched_is_torch_device_available

# 强制设置设备为 "cuda" 而不是 "cuda:0"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 使用GPU 1

from pi0 import PI0FASTPolicy, PI0Policy
from lerobot.common.policies.pi0.configuration_pi0 import PI0Config
# from V3R_pi0 import PI0FASTPolicy, PI0Policy

PATH_TO_PI_MODEL = (
    "/opt/liblibai-models/user-workspace2/users/lyh/model_checkpoint/pi0/pytorch/pi0_base"
)
PATH_TO_PI_FAST_MODEL = (
    "/opt/liblibai-models/user-workspace2/users/lyh/model_checkpoint/pi0/pytorch/pi0_fast_base"
)
model_type = "pi0"  # or "pi0fast"


# load model
try:
    # 你的主要代码
    if model_type == "pi0":
        policy = PI0Policy.from_pretrained(PATH_TO_PI_MODEL)
    else:
        policy = PI0FASTPolicy.from_pretrained(PATH_TO_PI_FAST_MODEL)
    print("✅ 模型加载完成")
except Exception as e:
    print(f"❌ 发生错误: {e}")
    import traceback
    traceback.print_exc()
# create pseudo observation
# check the comment in `PI0Policy.select_action` for the expected observation format
# let's assume we have the following observation
# policy.config.device = "cuda"
device = policy.config.device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
observation = {
    "image": {
        "base_0_rgb": torch.randint(
            0, 256, (1, 3, 224, 224), dtype=torch.uint8, device=device
        ),
        # "left_wrist_0_rgb": ...,   Suppose we don't have this view
        # "right_wrist_0_rgb": ...,  Suppose we don't have this view
    },
    "state": torch.randn(1, 8, device=device) * 0.2,
    "prompt": ["do something"],
}

# select action
# let's assume the `action_dim` is 7
start_time = time.perf_counter()
action_0 = policy.select_action(observation)[0, :, :7]
end_time = time.perf_counter()
print(f"Action_0 selection time: {(end_time - start_time) * 1000:.2f} ms.It's an inital step for the model")

start_websocket_server(host="0.0.0.0", port=8000, device=device)

print("Server ready, waiting for client...")
wait_for_client_connection()
print("Client connected, processing messages...")

def convert_raw_to_clean_observation(raw_data: dict, device) -> dict:
    """
    将原始数据转换为模型期望的观测格式，并移除reset字段
    输入: {"observation/image": np.array, "observation/state": np.array, "prompt": str, "reset": bool}
    输出: {"image": {"base_0_rgb": tensor}, "state": tensor, "prompt": [str]}
    """
    
    observation = {
        "image": {},
        "state": None,
        "prompt": [""]
    }
    
    # 处理图像数据 observation/image -> image/base_0_rgb
    if "observation/image" in raw_data:
        img = raw_data["observation/image"]
        if isinstance(img, np.ndarray) and len(img.shape) == 3:  # (H,W,C)
            # numpy (H,W,C) -> torch (1,C,H,W)
            img_tensor = torch.from_numpy(img.copy()).permute(2, 0, 1).unsqueeze(0)
            observation["image"]["base_0_rgb"] = img_tensor.to(dtype=torch.uint8, device=device)
    
    # 处理手腕相机（如果有）observation/wrist_image -> image/wrist_0_rgb  
    if "observation/wrist_image" in raw_data:
        wrist = raw_data["observation/wrist_image"]
        if isinstance(wrist, np.ndarray) and len(wrist.shape) == 3:
            wrist_tensor = torch.from_numpy(wrist).permute(2, 0, 1).unsqueeze(0)
            observation["image"]["left_wrist_0_rgb"] = wrist_tensor.to(dtype=torch.uint8, device=device)
    
    # 处理状态数据 observation/state -> state
    if "observation/state" in raw_data:
        state = raw_data["observation/state"]
        if isinstance(state, np.ndarray):
            # numpy (N,) -> torch (1,N)
            state_tensor = torch.from_numpy(state).unsqueeze(0)
            observation["state"] = state_tensor.to(dtype=torch.float32, device=device)
    
    # 处理prompt
    if "prompt" in raw_data:
        prompt = raw_data["prompt"]
        observation["prompt"] = [prompt] if isinstance(prompt, str) else prompt
    
    return observation


episode_count = 0
step_count = 0
while True:
    # 从WebSocket服务器获取观测数据（现在包含reset字段）
    raw_data = wait_for_observation()
    
    #  新增：检查reset字段
    need_reset = raw_data.get("reset", False)
    
    if need_reset:
        # 需要reset - 执行模型重置
        episode_count += 1
        step_count = 0
        print(f"RESET - 开始 Episode {episode_count}")
        # 在这里执行模型重置逻辑
        # policy.reset() # 如果模型有reset方法
        
        # reset时仍然需要处理这一步的推理，因为环境期待响应
        step_count += 1
        observation_data = convert_raw_to_clean_observation(raw_data, device)
        
    else:
        # 不需要reset - 正常推理
        step_count += 1
        # 转换数据并去除reset字段
        observation_data = convert_raw_to_clean_observation(raw_data, device)
    
    
    # 原有的策略推理逻辑保持不变
    action = policy.select_action(observation_data)[0, :, :7]
    send_action_response(action)
    
    # 日志输出
    print(f"Episode {episode_count} Step {step_count} | Prompt: {observation_data['prompt'][0]}")
