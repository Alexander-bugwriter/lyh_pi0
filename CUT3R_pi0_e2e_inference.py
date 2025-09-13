import torch
import os
import sys
import time
os.environ['TORCH_USE_CUDA_DSA'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# 设置PyTorch库路径
torch_lib_path = os.path.join(os.path.dirname(torch.__file__), 'lib')
current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
new_ld_path = torch_lib_path + ':' + current_ld_path
os.environ['LD_LIBRARY_PATH'] = new_ld_path

print(f"设置LD_LIBRARY_PATH: {torch_lib_path}")
print(f"当前LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', '')[:100]}...")

# 添加Python模块路径
script_dir = os.path.dirname(os.path.abspath(__file__))
paths_to_add = [
    os.path.join(script_dir, "V3R_pi0", "CUT3R", "src", "croco", "models", "curope"),
    os.path.join(script_dir, "V3R_pi0", "CUT3R", "src"),
    os.path.join(script_dir, "V3R_pi0", "CUT3R"),
]

for path in paths_to_add:
    if os.path.exists(path) and path not in sys.path:
        sys.path.insert(0, path)
        print(f"Added: {os.path.relpath(path, script_dir)}")

# 详细测试curope导入
print("\n=== 尝试导入curope ===")
try:
    import curope
    print("CUDA curope loaded successfully!")
    print(f"Available functions: {[attr for attr in dir(curope) if not attr.startswith('_')]}")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    
    # 检查.so文件的当前依赖状态
    curope_path = os.path.join(script_dir, "V3R_pi0", "CUT3R", "src", "croco", "models", "curope")
    so_file = os.path.join(curope_path, "curope.cpython-310-x86_64-linux-gnu.so")
    
    if os.path.exists(so_file):
        print("\n=== 检查.so文件依赖 ===")
        import subprocess
        try:
            result = subprocess.run(['ldd', so_file], capture_output=True, text=True)
            missing_libs = [line for line in result.stdout.split('\n') if 'not found' in line]
            if missing_libs:
                print("找不到的库:")
                for lib in missing_libs:
                    print(f"  {lib.strip()}")
            else:
                print("所有依赖库都找到了")
        except Exception as e:
            print(f"无法检查依赖: {e}")

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
from V3R_pi0 import PI0Policy

PATH_TO_PI_MODEL = (
    #"/opt/liblibai-models/user-workspace2/users/lyh/model_checkpoint/pi0/pytorch/pi0_base"
    "/opt/liblibai-models/user-workspace2/users/lyh/model_checkpoint/pi0/pytorch/CUT3R_pi0_test"
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
start_time = time.perf_counter()
action_0 = policy.select_action(observation)[0, :, :7]
end_time = time.perf_counter()
print(f"Action_0 selection time: {(end_time - start_time) * 1000:.2f} ms")
print(action_0)

start_time = time.perf_counter()
action_1 = policy.select_action(observation)[0, :, :7]
end_time = time.perf_counter()
print(f"Action_1 selection time: {(end_time - start_time) * 1000:.2f} ms")
print(action_1)
