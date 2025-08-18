import torch
import os
import json
import gc
from pathlib import Path
from V3R_pi0 import PI0Policy
from V3R_pi0.Hi3R import Hi3RPolicy  # 使用新的解耦版Hi3R

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
    
    Args:
        model_path: 模型目录路径
        map_location: 加载位置
    
    Returns:
        state_dict: 模型权重字典
        file_format: 文件格式 ('bin' 或 'safetensors')
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
    
    Args:
        state_dict: 模型权重字典
        output_path: 输出目录路径
        file_format: 文件格式 ('bin' 或 'safetensors')
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


def convert_pi0_to_global_head(original_model_path: str, output_model_path: str):
    """
    将原始PI0模型转换为带全局轨迹头的模型
    
    Args:
        original_model_path: 原始PI0模型路径
        output_model_path: 输出模型路径
    """
    print(f"Starting model conversion: {original_model_path} -> {output_model_path}")
    
    # 关键修复1：强制使用CPU进行转换，避免显存问题
    original_device = torch.cuda.current_device() if torch.cuda.is_available() else None
    torch.cuda.empty_cache()  # 清空显存
    
    try:
        # 1. 加载原始模型配置
        print("Loading original PI0 model config...")
        config_path = os.path.join(original_model_path, "config.json")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            original_config_dict = json.load(f)
        
        # 备份原始配置
        original_config_backup = original_config_dict.copy()
        
        # 为转换过程创建工作配置（不修改原始文件）
        working_config_dict = original_config_dict.copy()
        working_config_dict['type'] = 'pi0'
        working_config_dict['device'] = 'cpu'
        
        # 关键修复3：使用新的智能加载函数
        print("Loading original PI0 model on CPU (skipping spatial encoder)...")
        
        with torch.no_grad():
            # 使用新的智能加载函数
            original_state_dict, original_format = load_model_weights(original_model_path, 'cpu')
            
            # 转换所有张量到CPU float32
            for key in original_state_dict:
                if isinstance(original_state_dict[key], torch.Tensor):
                    original_state_dict[key] = original_state_dict[key].cpu().float()
        
        print(f"Original model weights loaded. Parameters: {len(original_state_dict)}, Format: {original_format}")
        
        # 关键修复4：创建新模型时使用临时配置文件，不修改原始配置
        print("Creating new model with global head (skipping spatial encoder)...")
        
        # 创建临时配置文件用于Hi3R加载
        temp_config_path = os.path.join(original_model_path, "config_temp_hi3r.json")
        temp_hi3r_config = working_config_dict.copy()
        
        # 为Hi3R添加必要的配置，但不影响原始配置
        temp_hi3r_config.update({
            'use_spatial_encoder': False,
            'use_history_features': False,
            'device': 'cpu'
        })
        
        with open(temp_config_path, 'w') as f:
            json.dump(temp_hi3r_config, f, indent=2)
        
        try:
            # 使用解耦版Hi3R，完全禁用空间编码器
            config_overrides = {
                'use_spatial_encoder': False,  # 强制禁用空间编码器
                'use_history_features': False,  # 强制禁用历史特征
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
                new_policy = Hi3RPolicy.from_pretrained(
                    original_model_path, 
                    config_overrides=config_overrides
                )
                # 修复：对model调用cpu()而不是policy
                new_policy.model = new_policy.model.cpu().float()
                new_state_dict = new_policy.model.state_dict()
                
                # 确保所有参数在CPU上
                for key in new_state_dict:
                    if isinstance(new_state_dict[key], torch.Tensor):
                        new_state_dict[key] = new_state_dict[key].cpu().float()
        finally:
            # 清理临时配置文件，确保不影响原始配置
            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)
                print(f"   Cleaned up temporary config: {temp_config_path}")
            
            # 确保原始配置文件完全恢复
            with open(config_path, 'w') as f:
                json.dump(original_config_backup, f, indent=2)
                print("   Original config file restored")
        
        print(f"New model created. Parameters: {len(new_state_dict)}")
        
        # 关键修复5：删除新模型引用，只保留state_dict
        del new_policy
        torch.cuda.empty_cache()
        gc.collect()
        
        # 3. 复制参数（全部在CPU上进行）
        print("Copying model parameters...")
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
        
        # 3.1 复制完全相同的参数
        print("Copying identical parameters...")
        copied_direct = 0
        
        for orig_key in original_state_dict.keys():
            target_key = map_key(orig_key, new_state_dict.keys())
            if target_key:
                new_state_dict[target_key] = original_state_dict[orig_key].clone()
                print(f"   Copied: {orig_key} -> {target_key}")
                copied_direct += 1
        
        print(f"   Directly copied {copied_direct} parameters")
        
        # 3.2 复制action expert参数到global trajectory expert
        print("Copying action expert to global trajectory expert...")
        
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
                        print(f"   Copied expert: {orig_key} -> {global_key}")
                        copied_expert += 1
                        break
        
        print(f"   Copied {copied_expert} action expert parameters to global expert")
        
        # 3.3 复制action投影参数到global trajectory投影
        print("Copying projection parameters...")
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
                print(f"   Copied projection: {orig_suffix} -> {target_suffix}")
                copied_proj += 1
        
        print(f"   Copied {copied_proj} projection parameters")
        
        # 关键修复7：释放原始state_dict内存
        del original_state_dict
        gc.collect()
        
        # 4. 检查未初始化的参数
        print("Checking uninitialized parameters...")
        
        # 统计哪些参数被成功复制了
        total_copied = copied_direct + copied_expert + copied_proj
        total_new_params = len(new_state_dict)
        
        print(f"   Total copied: {total_copied} parameters")
        print(f"   New model total: {total_new_params} parameters")
        
        # 找出所有未复制的参数并分类
        all_copied_keys = set()
        
        # 重新定义映射函数（确保一致性）
        def map_key_check(orig_key, target_keys):
            """智能映射键名，处理model前缀 - 检查版本"""
            if orig_key in target_keys:
                return orig_key
            if orig_key.startswith("model."):
                no_prefix_key = orig_key[6:]
                if no_prefix_key in target_keys:
                    return no_prefix_key
            prefixed_key = f"model.{orig_key}"
            if prefixed_key in target_keys:
                return prefixed_key
            return None
        
        for orig_key in original_state_dict.keys() if 'original_state_dict' in locals() else []:
            target_key = map_key_check(orig_key, new_state_dict.keys())
            if target_key:
                all_copied_keys.add(target_key)
        
        # 添加expert和投影参数到已复制集合
        possible_action_prefixes_check = [
            "paligemma_with_expert.gemma_expert.",
            "model.paligemma_with_expert.gemma_expert.",
            "gemma_expert.",
            "model.gemma_expert."
        ]
        global_expert_prefix_check = "global_trajectory_expert."
        
        # 注意：这里original_state_dict已经被删除，所以跳过这个检查
        print("   Note: Detailed parameter analysis skipped due to memory optimization")
        
        # 检查global trajectory expert相关参数
        global_params = [k for k in new_state_dict.keys() if k.startswith("global_trajectory_expert.")]
        print(f"   Global trajectory expert parameters: {len(global_params)} total")
        
        # 检查PaliGemma相关参数（这些可能需要预训练权重）
        paligemma_params = [k for k in new_state_dict.keys() if "paligemma_with_expert.paligemma" in k]
        print(f"   PaliGemma model parameters: {len(paligemma_params)} total")
        
        # 总结
        print(f"   Conversion completed with {total_copied} copied parameters")
        
        # 5. 保存转换后的模型
        print(f"Saving converted model to: {output_model_path}")
        
        # 关键修复8：使用新的智能保存函数，保持原格式
        save_model_weights(new_state_dict, output_model_path, original_format)
        
        # 保存配置文件（基于原始配置创建Hi3R配置）
        output_config_dict = original_config_backup.copy()
        # 为Hi3R添加必要的配置字段
        output_config_dict.update({
            'type': 'hi3r',  # 标记为Hi3R模型
            'use_spatial_encoder': False,  # 可以根据需要修改
            'use_history_features': False,  # 可以根据需要修改
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
        
        # 保存tokenizer相关文件
        tokenizer_files = [
            "tokenizer.json", "tokenizer_config.json", 
            "special_tokens_map.json", "vocab.txt"
        ]
        
        for file_name in tokenizer_files:
            src_path = os.path.join(original_model_path, file_name)
            if os.path.exists(src_path):
                dst_path = os.path.join(output_model_path, file_name)
                shutil.copy2(src_path, dst_path)
                print(f"   Tokenizer file copied: {file_name}")
        
        # 关键修复：复制空间编码器相关文件和目录
        print("Copying spatial encoder related files...")
        spatial_encoder_items = [
            "spatial_encoder",     # 空间编码器目录
            "cut3r",              # 可能的CUT3R目录
            "spatial_tower",      # 可能的空间塔目录
            "cut3r_weights.pth",  # 可能的权重文件
            "spatial_encoder.pth", # 可能的权重文件
        ]
        
        copied_spatial_items = []
        for item_name in spatial_encoder_items:
            src_path = os.path.join(original_model_path, item_name)
            dst_path = os.path.join(output_model_path, item_name)
            
            if os.path.exists(src_path):
                if os.path.isdir(src_path):
                    # 复制整个目录
                    shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
                    copied_spatial_items.append(f"Directory: {item_name}")
                else:
                    # 复制文件
                    shutil.copy2(src_path, dst_path)
                    copied_spatial_items.append(f"File: {item_name}")
        
        if copied_spatial_items:
            print(f"   Copied spatial encoder items: {copied_spatial_items}")
        else:
            print("   No spatial encoder files found, please verify manually")
            
        # 额外检查：列出原始目录中的所有文件，帮助识别遗漏的空间编码器文件
        print("Original model directory content check:")
        try:
            all_items = os.listdir(original_model_path)
            print(f"   Model directory contains {len(all_items)} items:")
            for item in all_items:
                item_path = os.path.join(original_model_path, item)
                if os.path.isdir(item_path):
                    print(f"   Directory: {item}")
                else:
                    file_size = os.path.getsize(item_path) / (1024*1024)  # MB
                    print(f"   File: {item} ({file_size:.1f} MB)")
        except Exception as e:
            print(f"   Cannot list directory contents: {e}")
        
        print("Model conversion completed!")
        
        # 关键修复9：简化验证，避免重新加载大模型
        print("Verifying conversion results...")
        try:
            # 验证保存的文件
            saved_state_dict, saved_format = load_model_weights(output_model_path, 'cpu')
            print(f"   Converted model can be loaded normally. Parameters: {len(saved_state_dict)}, Format: {saved_format}")
            
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
        
        # 关键修复10：最终清理
        del new_state_dict
        torch.cuda.empty_cache()
        gc.collect()
        
        return True
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        # 清理内存
        torch.cuda.empty_cache()
        gc.collect()
        raise e


def main():
    """主函数"""
    # 修改为你实际的路径
    original_model_path = "/opt/liblibai-models/user-workspace2/users/lyh/model_checkpoint/pi0/pytorch/pi0_base"
    output_model_path = "/opt/liblibai-models/user-workspace2/users/lyh/model_checkpoint/Hi3R/pytorch/Hi3R_base"
    
    # 关键修复11：检查路径和磁盘空间
    if not os.path.exists(original_model_path):
        print(f"Original model path does not exist: {original_model_path}")
        return
    
    # 检查磁盘空间（粗略估计）
    import shutil
    _, _, free_space = shutil.disk_usage(os.path.dirname(output_model_path))
    free_gb = free_space // (1024**3)
    print(f"Available space at output path: {free_gb} GB")
    
    if free_gb < 10:  # 至少需要10GB空间
        print("WARNING: Disk space may be insufficient!")
    
    # 检查safetensors支持
    if SAFETENSORS_AVAILABLE:
        print("Safetensors support enabled")
    else:
        print("Safetensors not installed, only .bin format supported")
        print("   Install command: pip install safetensors")
    
    # 关键修复12：设置环境变量减少内存使用
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    
    print("Starting model conversion...")
    print(f"Current available VRAM: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB" if torch.cuda.is_available() else "CPU mode")
    
    try:
        success = convert_pi0_to_global_head(original_model_path, output_model_path)
        if success:
            print("Conversion completed successfully!")
        else:
            print("Conversion failed!")
    except Exception as e:
        print(f"Conversion error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
