import torch
import os
os.environ['TORCH_USE_CUDA_DSA'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
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

# from pi0 import PI0FASTPolicy, PI0Policy
from V3R_pi0 import PI0FASTPolicy, PI0Policy

PATH_TO_PI_MODEL = (
    "/home/pi0_model_checkpoint/pytorch/pi0_libero"
)
PATH_TO_PI_FAST_MODEL = (
    "/home/pi0_model_checkpoint/pytorch/pi0_fast_libero"
)
model_type = "pi0"  # or "pi0fast"


# load model
if model_type == "pi0":
    policy = PI0Policy.from_pretrained(PATH_TO_PI_MODEL)
else:
    policy = PI0FASTPolicy.from_pretrained(PATH_TO_PI_FAST_MODEL)

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
        "left_wrist_0_rgb": torch.randint(
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
action = policy.select_action(observation)[0, :, :7]
print(action)
