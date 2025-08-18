import torch
import os
import json
import gc
import shutil
import tempfile
import hashlib
from pathlib import Path
from V3R_pi0 import PI0Policy
from V3R_pi0.Hi3R import Hi3RPolicy

# 新增：支持safetensors
try:
    from safetensors.torch import load_file as load_safetensors
    from safetensors.torch import save_file as save_safetensors
    SAFETENSORS_AVAILABLE = True
except ImportError:
    print("WARNING: safetensors not installed, only .bin format supported")
    SAFETENSORS_AVAILABLE = False


def load_model_weights(model_path: str, map_location='cpu'):
    """
    智能加载模型权重，支持.bin和.safetensors格式
    """
    # 优先检查safetensors格式
    safetensors_file = os.path.join(model_path, "model.safetensors")
    bin_file = os.path.join(model_path, "pytorch_model.bin")
    
    if os.path.exists(safetensors_file) and SAFETENSORS_AVAILABLE:
        print(f"   Loading safetensors format: {safetensors_file}")
        state_dict = load_safetensors(safetensors_file)
        # safetensors加载后需要移动到指定设备
        if map_location != 'cpu':
            for key in state_dict:
                if isinstance(state_dict[key], torch.Tensor):
                    state_dict[key] = state_dict[key].to(map_location)
        return state_dict, 'safetensors'
    
    elif os.path.exists(bin_file):
        print(f"   Loading bin format: {bin_file}")
        state_dict = torch.load(bin_file, map_location=map_location)
        return state_dict, 'bin'
    
    else:
        # 更详细的错误信息
        available_files = []
        try:
            for file in os.listdir(model_path):
                if file.endswith(('.bin', '.safetensors', '.pth')):
                    file_size = os.path.getsize(os.path.join(model_path, file)) / (1024*1024)
                    available_files.append(f"{file} ({file_size:.1f}MB)")
        except:
            pass
            
        error_msg = f"No supported model weight file found!\n"
        error_msg += f"Checked path: {model_path}\n"
        error_msg += f"Expected files: model.safetensors or pytorch_model.bin\n"
        if available_files:
            error_msg += f"Available files: {available_files}"
        else:
            error_msg += "No weight files found in directory"
            
        raise FileNotFoundError(error_msg)


def save_model_weights(state_dict: dict, output_path: str, file_format: str = 'safetensors'):
    """
    保存模型权重，支持选择格式
    """
    os.makedirs(output_path, exist_ok=True)
    
    if file_format == 'safetensors' and SAFETENSORS_AVAILABLE:
        output_file = os.path.join(output_path, "model.safetensors")
        # 确保所有张量在CPU上且为float32
        cpu_state_dict = {}
        for key, value in state_dict.items():
            if isinstance(value, torch.Tensor):
                cpu_state_dict[key] = value.cpu().float()
            else:
                cpu_state_dict[key] = value
        save_safetensors(cpu_state_dict, output_file)
        print(f"   Saved as safetensors format: {output_file}")
    else:
        output_file = os.path.join(output_path, "pytorch_model.bin")
        # 确保所有张量在CPU上
        cpu_state_dict = {}
        for key, value in state_dict.items():
            if isinstance(value, torch.Tensor):
                cpu_state_dict[key] = value.cpu().float()
            else:
                cpu_state_dict[key] = value
        torch.save(cpu_state_dict, output_file)
        print(f"   Saved as bin format: {output_file}")


def create_isolated_model_copy(original_model_path: str) -> str:
    """
    创建完全隔离的模型副本，不影响原始文件
    
    Returns:
        isolated_model_path: 隔离的模型副本路径
    """
    print("Creating isolated model copy...")
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp(prefix="pi0_conversion_isolated_")
    isolated_model_path = os.path.join(temp_dir, "isolated_model")
    
    print(f"   Isolated model directory: {isolated_model_path}")
    
    # 复制整个模型目录到临时位置
    try:
        shutil.copytree(original_model_path, isolated_model_path)
        print(f"   Model copied to isolated directory")
        
        # 验证关键文件存在
        config_file = os.path.join(isolated_model_path, "config.json")
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Config file not found in isolated copy: {config_file}")
            
        # 检查权重文件
        weight_files = [
            os.path.join(isolated_model_path, "model.safetensors"),
            os.path.join(isolated_model_path, "pytorch_model.bin")
        ]
        
        weight_exists = any(os.path.exists(f) for f in weight_files)
        if not weight_exists:
            raise FileNotFoundError("No weight file found in isolated copy")
            
        print(f"   Isolated copy verification passed")
        return isolated_model_path
        
    except Exception as e:
        # 清理失败的临时目录
        try:
            shutil.rmtree(temp_dir)
        except:
            pass
        raise Exception(f"Failed to create isolated model copy: {e}")


def verify_original_model_integrity(original_model_path: str, backup_checksums: dict) -> bool:
    """
    验证原始模型文件完整性
    
    Args:
        original_model_path: 原始模型路径
        backup_checksums: 备份的文件校验和
    
    Returns:
        完整性验证是否通过
    """
    print("Verifying original model integrity...")
    
    try:
        for file_name, expected_checksum in backup_checksums.items():
            file_path = os.path.join(original_model_path, file_name)
            
            if not os.path.exists(file_path):
                print(f"   File missing: {file_name}")
                return False
                
            with open(file_path, 'rb') as f:
                current_checksum = hashlib.md5(f.read()).hexdigest()
                
            if current_checksum != expected_checksum:
                print(f"   File modified: {file_name}")
                return False
            else:
                print(f"   File intact: {file_name}")
                
        print("   ✅ All original files verified intact")
        return True
        
    except Exception as e:
        print(f"   ❌ Verification error: {e}")
        return False


def calculate_directory_checksums(directory_path: str) -> dict:
    """
    计算目录中关键文件的校验和
    """
    checksums = {}
    
    # 需要监控的关键文件
    key_files = [
        "config.json",
        "model.safetensors", 
        "pytorch_model.bin",
        "tokenizer.json",
        "tokenizer_config.json"
    ]
    
    for file_name in key_files:
        file_path = os.path.join(directory_path, file_name)
        if os.path.exists(file_path):
            try:
                with open(file_path, 'rb') as f:
                    checksums[file_name] = hashlib.md5(f.read()).hexdigest()
                print(f"   Calculated checksum for: {file_name}")
            except Exception as e:
                print(f"   Warning: Could not calculate checksum for {file_name}: {e}")
                
    return checksums


def convert_pi0_to_global_head(original_model_path: str, output_model_path: str):
    """
    将原始PI0模型转换为带全局轨迹头的模型 - 完全隔离版本
    
    Args:
        original_model_path: 原始PI0模型路径
        output_model_path: 输出模型路径
    """
    print(f"Starting ISOLATED model conversion: {original_model_path} -> {output_model_path}")
    print("=" * 80)
    
    # 强制使用CPU进行转换，避免显存问题
    torch.cuda.empty_cache()
    
    # 第一步：计算原始文件校验和，用于后续验证
    print("Step 1: Calculating original file checksums for integrity verification...")
    original_checksums = calculate_directory_checksums(original_model_path)
    
    isolated_model_path = None
    temp_dir = None
    
    try:
        # 第二步：创建完全隔离的模型副本
        print("\nStep 2: Creating isolated model copy...")
        isolated_model_path = create_isolated_model_copy(original_model_path)
        temp_dir = os.path.dirname(isolated_model_path)
        
        # 第三步：在隔离副本上加载原始配置
        print("\nStep 3: Loading configuration from isolated copy...")
        isolated_config_path = os.path.join(isolated_model_path, "config.json")
        
        with open(isolated_config_path, 'r') as f:
            original_config_dict = json.load(f)
        
        print(f"   Original config loaded from isolated copy")
        
        # 第四步：加载原始模型权重（从隔离副本）
        print("\nStep 4: Loading original model weights from isolated copy...")
        
        with torch.no_grad():
            # 从隔离副本加载权重
            original_state_dict, original_format = load_model_weights(isolated_model_path, 'cpu')
            
            # 转换所有张量到CPU float32
            for key in original_state_dict:
                if isinstance(original_state_dict[key], torch.Tensor):
                    original_state_dict[key] = original_state_dict[key].cpu().float()
        
        print(f"   Original model weights loaded. Parameters: {len(original_state_dict)}, Format: {original_format}")
        
        # 第五步：修改隔离副本的配置文件以创建Hi3R模型
        print("\nStep 5: Modifying isolated copy config for Hi3R...")
        
        # 在隔离副本上修改配置（不影响原始）
        hi3r_config = original_config_dict.copy()
        hi3r_config.update({
            'type': 'hi3r',
            'use_spatial_encoder': False,
            'use_history_features': False,
            'device': 'cpu',
            'spatial_camera_config': {
                "base_0_rgb": False,
                "left_wrist_0_rgb": False,
                "right_wrist_0_rgb": False,
            },
            'history_camera_config': {
                "base_0_rgb": False,
                "left_wrist_0_rgb": False,
                "right_wrist_0_rgb": False,
            }
        })
        
        # 修改隔离副本的配置文件
        with open(isolated_config_path, 'w') as f:
            json.dump(hi3r_config, f, indent=2)
        
        print(f"   Isolated config modified for Hi3R")
        
        # 第六步：从修改后的隔离副本创建Hi3R模型
        print("\nStep 6: Creating Hi3R model from isolated copy...")
        
        config_overrides = {
            'use_spatial_encoder': False,
            'use_history_features': False,
            'spatial_camera_config': {
                "base_0_rgb": False,
                "left_wrist_0_rgb": False,
                "right_wrist_0_rgb": False,
            },
            'history_camera_config': {
                "base_0_rgb": False,
                "left_wrist_0_rgb": False,
                "right_wrist_0_rgb": False,
            }
        }
        
        with torch.no_grad():
            # 从隔离副本加载Hi3R模型
            new_policy = Hi3RPolicy.from_pretrained(
                isolated_model_path,  # 使用隔离副本路径
                config_overrides=config_overrides
            )
            
            new_policy.model = new_policy.model.cpu().float()
            new_state_dict = new_policy.model.state_dict()
            
            # 确保所有参数在CPU上
            for key in new_state_dict:
                if isinstance(new_state_dict[key], torch.Tensor):
                    new_state_dict[key] = new_state_dict[key].cpu().float()
        
        print(f"   Hi3R model created. Parameters: {len(new_state_dict)}")
        
        # 释放Hi3R模型引用
        del new_policy
        torch.cuda.empty_cache()
        gc.collect()
        
        # 第七步：复制参数（全部在CPU上进行）
        print("\nStep 7: Copying model parameters...")
        print(f"   Original model parameters: {len(original_state_dict)}")
        print(f"   New model parameters: {len(new_state_dict)}")
        
        # 检查参数命名模式
        orig_has_model_prefix = any(key.startswith("model.") for key in original_state_dict.keys())
        print(f"   Original model has 'model.' prefix: {orig_has_model_prefix}")
        
        # 创建键名映射函数
        def map_key(orig_key, target_keys):
            """智能映射键名，处理model前缀"""
            # 直接匹配
            if orig_key in target_keys:
                return orig_key
            
            # 去掉model前缀匹配
            if orig_key.startswith("model."):
                no_prefix_key = orig_key[6:]  # 去掉"model."
                if no_prefix_key in target_keys:
                    return no_prefix_key
            
            # 添加model前缀匹配
            prefixed_key = f"model.{orig_key}"
            if prefixed_key in target_keys:
                return prefixed_key
                
            return None
        
        # 7.1 复制完全相同的参数
        print("   Copying identical parameters...")
        copied_direct = 0
        
        for orig_key in original_state_dict.keys():
            target_key = map_key(orig_key, new_state_dict.keys())
            if target_key:
                new_state_dict[target_key] = original_state_dict[orig_key].clone()
                print(f"     Copied: {orig_key} -> {target_key}")
                copied_direct += 1
        
        print(f"   Directly copied {copied_direct} parameters")
        
        # 7.2 复制action expert参数到global trajectory expert
        print("   Copying action expert to global trajectory expert...")
        
        # 适配不同的参数路径格式
        possible_action_prefixes = [
            "paligemma_with_expert.gemma_expert.",
            "model.paligemma_with_expert.gemma_expert.",
            "gemma_expert.",
            "model.gemma_expert."
        ]
        
        global_expert_prefix = "global_trajectory_expert."
        copied_expert = 0
        
        for orig_key in original_state_dict.keys():
            for action_prefix in possible_action_prefixes:
                if orig_key.startswith(action_prefix):
                    # 构建目标键
                    suffix = orig_key[len(action_prefix):]
                    global_key = global_expert_prefix + suffix
                    
                    if global_key in new_state_dict:
                        new_state_dict[global_key] = original_state_dict[orig_key].clone()
                        print(f"     Copied expert: {orig_key} -> {global_key}")
                        copied_expert += 1
                        break
        
        print(f"   Copied {copied_expert} action expert parameters to global expert")
        
        # 7.3 复制action投影参数到global trajectory投影
        print("   Copying projection parameters...")
        projection_mappings = [
            ("action_in_proj.weight", "global_trajectory_in_proj.weight"),
            ("action_in_proj.bias", "global_trajectory_in_proj.bias"),
            ("model.action_in_proj.weight", "global_trajectory_in_proj.weight"),
            ("model.action_in_proj.bias", "global_trajectory_in_proj.bias"),
        ]
        
        copied_proj = 0
        for orig_suffix, target_suffix in projection_mappings:
            if orig_suffix in original_state_dict and target_suffix in new_state_dict:
                new_state_dict[target_suffix] = original_state_dict[orig_suffix].clone()
                print(f"     Copied projection: {orig_suffix} -> {target_suffix}")
                copied_proj += 1
        
        print(f"   Copied {copied_proj} projection parameters")
        
        # 释放原始state_dict内存
        del original_state_dict
        gc.collect()
        
        # 第八步：保存转换后的模型到目标目录
        print(f"\nStep 8: Saving converted model to: {output_model_path}")
        
        # 创建输出目录
        os.makedirs(output_model_path, exist_ok=True)
        
        # 保存权重文件
        save_model_weights(new_state_dict, output_model_path, original_format)
        
        # 保存Hi3R配置文件（基于原始配置创建）
        output_config_dict = original_config_dict.copy()
        output_config_dict.update({
            'type': 'hi3r',
            'use_spatial_encoder': False,
            'use_history_features': False,
            'spatial_camera_config': {
                "base_0_rgb": False,
                "left_wrist_0_rgb": False,
                "right_wrist_0_rgb": False,
            },
            'history_camera_config': {
                "base_0_rgb": False,
                "left_wrist_0_rgb": False,
                "right_wrist_0_rgb": False,
            }
        })
        
        output_config_path = os.path.join(output_model_path, "config.json")
        with open(output_config_path, 'w') as f:
            json.dump(output_config_dict, f, indent=2)
        print("   Hi3R config file saved")
        
        # 第九步：复制其他必要文件（从原始目录，不是隔离副本）
        print("\nStep 9: Copying additional files from original directory...")
        
        # 复制tokenizer相关文件
        tokenizer_files = [
            "tokenizer.json", "tokenizer_config.json", 
            "special_tokens_map.json", "vocab.txt"
        ]
        
        for file_name in tokenizer_files:
            src_path = os.path.join(original_model_path, file_name)  # 从原始目录复制
            if os.path.exists(src_path):
                dst_path = os.path.join(output_model_path, file_name)
                shutil.copy2(src_path, dst_path)
                print(f"   Tokenizer file copied: {file_name}")
        
        # 复制空间编码器相关文件和目录
        print("   Copying spatial encoder related files...")
        spatial_encoder_items = [
            "spatial_encoder",
            "cut3r",
            "spatial_tower",
            "cut3r_weights.pth",
            "spatial_encoder.pth",
        ]
        
        copied_spatial_items = []
        for item_name in spatial_encoder_items:
            src_path = os.path.join(original_model_path, item_name)  # 从原始目录复制
            dst_path = os.path.join(output_model_path, item_name)
            
            if os.path.exists(src_path):
                try:
                    if os.path.isdir(src_path):
                        shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
                        copied_spatial_items.append(f"Directory: {item_name}")
                    else:
                        shutil.copy2(src_path, dst_path)
                        copied_spatial_items.append(f"File: {item_name}")
                except Exception as e:
                    print(f"     Warning: Failed to copy {item_name}: {e}")
        
        if copied_spatial_items:
            print(f"   Copied spatial encoder items: {copied_spatial_items}")
        else:
            print("   No spatial encoder files found")
        
        # 第十步：验证转换结果
        print("\nStep 10: Verifying conversion results...")
        try:
            saved_state_dict, saved_format = load_model_weights(output_model_path, 'cpu')
            print(f"   Converted model loads successfully. Parameters: {len(saved_state_dict)}, Format: {saved_format}")
            
            # 检查关键参数是否存在
            key_params = [
                "global_trajectory_expert.model.layers.0.self_attn.q_proj.weight",
                "global_trajectory_in_proj.weight"
            ]
            
            missing_key_params = [key for key in key_params if key not in saved_state_dict]
            if missing_key_params:
                print(f"   Missing key parameters: {missing_key_params}")
            else:
                print("   Key parameters verification passed")
                
            del saved_state_dict
            
        except Exception as e:
            print(f"   Verification failed: {e}")
        
        # 最终清理
        del new_state_dict
        torch.cuda.empty_cache()
        gc.collect()
        
        print("\n" + "=" * 80)
        print("Model conversion completed successfully!")
        
        # 统计信息
        total_copied = copied_direct + copied_expert + copied_proj
        print(f"Conversion Statistics:")
        print(f"   Direct parameters copied: {copied_direct}")
        print(f"   Expert parameters copied: {copied_expert}")
        print(f"   Projection parameters copied: {copied_proj}")
        print(f"   Total parameters copied: {total_copied}")
        
        return True
        
    except Exception as e:
        print(f"\nError during conversion: {e}")
        torch.cuda.empty_cache()
        gc.collect()
        raise e
        
    finally:
        # 第十一步：验证原始文件完整性
        print(f"\nStep 11: Verifying original model integrity...")
        integrity_ok = verify_original_model_integrity(original_model_path, original_checksums)
        
        if integrity_ok:
            print("Original model files remain completely intact!")
        else:
            print("WARNING: Original model files may have been modified!")
        
        # 清理隔离的临时目录
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                print(f"Isolated temporary directory cleaned up: {temp_dir}")
            except Exception as e:
                print(f"Warning: Could not clean up temporary directory {temp_dir}: {e}")


def main():
    """主函数"""
    # 修改为你实际的路径
    original_model_path = "/opt/liblibai-models/user-workspace2/users/lyh/model_checkpoint/pi0/pytorch/pi0_base"
    output_model_path = "/opt/liblibai-models/user-workspace2/users/lyh/model_checkpoint/Hi3R/pytorch/Hi3R_base"
    
    print("PI0 to Hi3R Model Converter - ISOLATED VERSION")
    print("=" * 80)
    print("This version creates a complete isolated copy to prevent any")
    print("modification of original model files.")
    print("=" * 80)
    
    # 检查路径和磁盘空间
    if not os.path.exists(original_model_path):
        print(f"ERROR: Original model path does not exist: {original_model_path}")
        return
    
    # 检查磁盘空间（需要更多空间因为要创建完整副本）
    try:
        _, _, free_space = shutil.disk_usage(os.path.dirname(output_model_path))
        free_gb = free_space // (1024**3)
        print(f"Available space at output path: {free_gb} GB")
        
        if free_gb < 20:  # 隔离版本需要更多空间
            print("WARNING: Disk space may be insufficient! (Need at least 20GB for isolated conversion)")
        else:
            print("Sufficient disk space available")
    except Exception as e:
        print(f"Could not check disk space: {e}")
    
    # 检查safetensors支持
    if SAFETENSORS_AVAILABLE:
        print("Safetensors support enabled")
    else:
        print("Safetensors not installed, only .bin format supported")
        print("   Install command: pip install safetensors")
    
    # 设置环境变量减少内存使用
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    
    print(f"\nStarting isolated conversion process...")
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory // (1024**3)
        print(f"Current available VRAM: {vram_gb} GB")
    else:
        print("Running in CPU mode")
    
    try:
        success = convert_pi0_to_global_head(original_model_path, output_model_path)
        if success:
            print("\nISOLATED CONVERSION COMPLETED SUCCESSFULLY!")
            print(f"Output saved to: {output_model_path}")
            print("Original model files remain completely untouched!")
        else:
            print("\nConversion failed!")
    except Exception as e:
        print(f"\nConversion error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
