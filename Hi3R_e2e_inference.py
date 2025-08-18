import torch
import os
import sys
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
    print(f"导入失败: {e}")
    
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
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 使用GPU 0

# 更新导入：使用Hi3R而不是PI0
from V3R_pi0.Hi3R import Hi3RPolicy

# 更新模型路径：使用转换后的Hi3R模型
PATH_TO_HI3R_MODEL = (
    "/opt/liblibai-models/user-workspace2/users/lyh/model_checkpoint/Hi3R/pytorch/Hi3R_base"
)

print("=== 加载Hi3R模型 ===")

# 加载Hi3R模型
# 可以选择性地禁用空间编码器以加快加载速度（如果不需要的话）
config_overrides = {
    # 如果想要启用空间编码器，设置为True并配置相应的camera
    'use_spatial_encoder': True,  # 设置为True来启用空间编码器
    'use_history_features': True,  # 设置为True来启用历史特征
    'spatial_camera_config': {
        "base_0_rgb": True,          # 根据需要启用
        "left_wrist_0_rgb": True,    # 根据需要启用
        "right_wrist_0_rgb": False,   # 根据需要启用
    },
    #下面的设置不要变，变了会报错。但是我也懒得debug了
    'history_camera_config': {
        "base_0_rgb": True,
        "left_wrist_0_rgb": False,
        "right_wrist_0_rgb": False,
    },
}

try:
    policy = Hi3RPolicy.from_pretrained(
        PATH_TO_HI3R_MODEL,
        config_overrides=config_overrides
    )
    print("Hi3R模型加载成功！")
    print(f"模型设备: {policy.config.__dict__.get('device', 'not specified')}")
    
    # 将模型移动到GPU（如果可用）
    if torch.cuda.is_available():
        device = torch.device('cuda')
        policy.model = policy.model.to(device)
        print(f"模型已移动到: {device}")
    else:
        device = torch.device('cpu')
        print("使用CPU进行推理")
        
except Exception as e:
    print(f"模型加载失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n=== 创建测试观测数据 ===")

# 创建伪观测数据（与PI0格式相同，Hi3R兼容）
# 检查 `Hi3RPolicy.select_action` 中的预期观测格式
observation = {
    "image": {
        "base_0_rgb": torch.randint(
            0, 256, (1, 3, 224, 224), dtype=torch.uint8, device=device
        ),
        "left_wrist_0_rgb": torch.randint(
            0, 256, (1, 3, 224, 224), dtype=torch.uint8, device=device
        ),
        # 如果有更多视角，可以添加：
        # "right_wrist_0_rgb": torch.randint(
        #     0, 256, (1, 3, 224, 224), dtype=torch.uint8, device=device
        # ),
    },
    "state": torch.randn(1, 8, device=device) * 0.2,
    "prompt": ["pick up the red cube and place it in the box"],
}

print("观测数据创建完成:")
print(f"  图像形状: {observation['image']['base_0_rgb'].shape}")
print(f"  状态形状: {observation['state'].shape}")
print(f"  任务提示: {observation['prompt'][0]}")

print("\n=== 执行Hi3R推理 ===")

try:
    # 使用Hi3R进行动作选择
    # Hi3R会先进行全局轨迹规划，然后生成实时动作
    with torch.no_grad():
        # 让action_dim是7（例如7DOF机械臂）
        action = policy.select_action(observation)[0, :, :7]
        
    print("Hi3R推理成功！")
    print(f"输出动作形状: {action.shape}")
    print(f"生成的动作序列:")
    for i, step_action in enumerate(action):
        print(f"  Step {i+1}: {step_action.cpu().numpy()}")
        
    # 额外信息
    print(f"\n动作统计:")
    print(f"  最大值: {action.max().item():.4f}")
    print(f"  最小值: {action.min().item():.4f}")
    print(f"  平均值: {action.mean().item():.4f}")
    print(f"  标准差: {action.std().item():.4f}")
    
except Exception as e:
    print(f"推理失败: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Hi3R推理完成 ===")

# 可选：展示Hi3R的特殊功能
print("\n=== Hi3R模型信息 ===")
try:
    # 显示模型配置
    config = policy.config
    print(f"动作步数: {config.n_action_steps}")
    print(f"最大动作维度: {config.max_action_dim}")
    print(f"最大状态维度: {config.max_state_dim}")
    print(f"投影宽度: {config.proj_width}")
    print(f"扩散步数: {config.num_steps}")
    print(f"空间编码器启用: {config.use_spatial_encoder}")
    print(f"历史特征启用: {config.use_history_features}")
    
    # 显示模型参数统计
    total_params = sum(p.numel() for p in policy.model.parameters())
    trainable_params = sum(p.numel() for p in policy.model.parameters() if p.requires_grad)
    print(f"\n模型参数统计:")
    print(f"  总参数数: {total_params:,}")
    print(f"  可训练参数数: {trainable_params:,}")
    
    # 检查是否有全局轨迹专家
    if hasattr(policy.model, 'global_trajectory_expert'):
        global_params = sum(p.numel() for p in policy.model.global_trajectory_expert.parameters())
        print(f"  全局轨迹专家参数数: {global_params:,}")
    
except Exception as e:
    print(f"无法获取模型信息: {e}")

print("\nHi3R端到端推理测试完成！")
