import torch
import os
import json
import gc
from pathlib import Path
from V3R_pi0 import PI0Policy
from V3R_pi0.Hi3R import Hi3RPolicy  # 🔥 使用新的解耦版Hi3R


def convert_pi0_to_global_head(original_model_path: str, output_model_path: str):
    """
    将原始PI0模型转换为带全局轨迹头的模型
    
    Args:
        original_model_path: 原始PI0模型路径
        output_model_path: 输出模型路径
    """
    print(f"🔄 开始转换模型: {original_model_path} -> {output_model_path}")
    
    # 🔥 关键修复1：强制使用CPU进行转换，避免显存问题
    original_device = torch.cuda.current_device() if torch.cuda.is_available() else None
    torch.cuda.empty_cache()  # 清空显存
    
    try:
        # 1. 加载原始模型配置
        print("📥 准备加载原始PI0模型配置...")
        config_path = os.path.join(original_model_path, "config.json")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
            
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # 修复配置
        config_dict['type'] = 'pi0'
        # 🔥 关键修复2：强制使用CPU进行转换
        config_dict['device'] = 'cpu'
        
        # 保存修改后的配置
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        # 🔥 关键修复3：避免加载空间编码器，只加载纯PI0权重
        print("📥 在CPU上加载原始PI0模型（跳过空间编码器）...")
        
        # 方案1：直接加载state_dict而不初始化完整模型
        model_file = os.path.join(original_model_path, "pytorch_model.bin")
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"模型权重文件不存在: {model_file}")
            
        with torch.no_grad():
            # 直接加载state_dict，避免模型初始化时加载CUT3R
            print("   📦 直接加载state_dict，避免空间编码器加载...")
            original_state_dict = torch.load(model_file, map_location='cpu')
            
            # 转换所有张量到CPU float32
            for key in original_state_dict:
                if isinstance(original_state_dict[key], torch.Tensor):
                    original_state_dict[key] = original_state_dict[key].cpu().float()
        
        print(f"✅ 原始模型权重加载完成，参数数量: {len(original_state_dict)}")
        
        # 🔥 关键修复4：创建新模型时也要小心，避免空间编码器加载
        print("🔨 创建带全局头的新模型（跳过空间编码器）...")
        
        # 临时修改配置，禁用空间编码器
        temp_config_path = os.path.join(original_model_path, "config_temp.json")
        with open(config_path, 'r') as f:
            temp_config = json.load(f)
        
        # 临时禁用空间编码器相关功能，避免CUT3R加载
        original_spatial_config = temp_config.get('use_spatial_encoder', True)
        temp_config['use_spatial_encoder'] = False
        temp_config['device'] = 'cpu'
        
        with open(temp_config_path, 'w') as f:
            json.dump(temp_config, f, indent=2)
        
        try:
            # 使用解耦版Hi3R，完全禁用空间编码器
            config_overrides = {
                'use_spatial_encoder': False,  # 🔥 强制禁用空间编码器
                'use_history_features': False,  # 🔥 强制禁用历史特征
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
                new_policy = new_policy.cpu().float()
                new_state_dict = new_policy.model.state_dict()
                
                # 确保所有参数在CPU上
                for key in new_state_dict:
                    if isinstance(new_state_dict[key], torch.Tensor):
                        new_state_dict[key] = new_state_dict[key].cpu().float()
        finally:
            # 清理临时文件并恢复原始配置
            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)
            
            # 恢复原始配置中的空间编码器设置
            temp_config['use_spatial_encoder'] = original_spatial_config
            with open(config_path, 'w') as f:
                json.dump(temp_config, f, indent=2)
        
        print(f"✅ 新模型创建完成，参数数量: {len(new_state_dict)}")
        
        # 🔥 关键修复5：删除新模型引用，只保留state_dict
        del new_policy
        torch.cuda.empty_cache()
        gc.collect()
        
        # 3. 复制参数（全部在CPU上进行）
        print("📋 复制模型参数...")
        
        # 3.1 复制完全相同的参数
        common_keys = set(original_state_dict.keys()) & set(new_state_dict.keys())
        print(f"   共同参数数量: {len(common_keys)}")
        
        for key in common_keys:
            new_state_dict[key] = original_state_dict[key].clone()
            print(f"   ✅ 复制: {key}")
        
        # 3.2 复制action expert参数到global trajectory expert
        print("🔄 复制action expert到global trajectory expert...")
        
        # 🔥 更新：适配新的Hi3R架构参数路径
        action_expert_prefix = "paligemma_with_expert.gemma_expert."
        global_expert_prefix = "global_trajectory_expert."
        
        copied_count = 0
        for key in original_state_dict.keys():
            if key.startswith(action_expert_prefix):
                global_key = key.replace(action_expert_prefix, global_expert_prefix)
                if global_key in new_state_dict:
                    new_state_dict[global_key] = original_state_dict[key].clone()
                    print(f"   ✅ 复制: {key} -> {global_key}")
                    copied_count += 1
        
        print(f"   复制了 {copied_count} 个action expert参数到global expert")
        
        # 3.3 复制action投影参数到global trajectory投影
        action_proj_keys = [
            "action_in_proj.weight",
            "action_in_proj.bias"
        ]
        global_proj_keys = [
            "global_trajectory_in_proj.weight", 
            "global_trajectory_in_proj.bias"
        ]
        
        for orig_key, new_key in zip(action_proj_keys, global_proj_keys):
            if orig_key in original_state_dict and new_key in new_state_dict:
                new_state_dict[new_key] = original_state_dict[orig_key].clone()
                print(f"   ✅ 复制投影参数: {orig_key} -> {new_key}")
            elif f"model.{orig_key}" in original_state_dict and new_key in new_state_dict:
                # 兼容带model前缀的键名
                new_state_dict[new_key] = original_state_dict[f"model.{orig_key}"].clone()
                print(f"   ✅ 复制投影参数: model.{orig_key} -> {new_key}")
        
        # 🔥 关键修复7：释放原始state_dict内存
        del original_state_dict
        gc.collect()
        
        # 4. 检查未初始化的参数
        print("🔍 检查未初始化的参数...")
        missing_keys = []
        for key in new_state_dict.keys():
            if key not in common_keys and not any(key.startswith(prefix) for prefix in [
                global_expert_prefix, "global_trajectory_in_proj."
            ]):
                missing_keys.append(key)
        
        if missing_keys:
            print(f"   ⚠️  以下参数将使用随机初始化: {missing_keys}")
        else:
            print("   ✅ 所有参数都已正确初始化")
        
        # 5. 保存转换后的模型
        print(f"💾 保存转换后的模型到: {output_model_path}")
        os.makedirs(output_model_path, exist_ok=True)
        
        # 🔥 关键修复8：直接保存state_dict，不创建新的模型实例
        torch.save(new_state_dict, os.path.join(output_model_path, "pytorch_model.bin"))
        
        # 保存配置文件
        import shutil
        shutil.copy2(config_path, os.path.join(output_model_path, "config.json"))
        print("   ✅ 复制配置文件")
        
        # 保存tokenizer相关文件
        tokenizer_files = [
            "tokenizer.json", "tokenizer_config.json", 
            "special_tokens_map.json", "vocab.txt"
        ]
        
        for file_name in tokenizer_files:
            src_path = os.path.join(original_model_path, file_name)
            if os.path.exists(src_path):
                shutil.copy2(src_path, os.path.join(output_model_path, file_name))
        
        # 🔥 关键修复：复制空间编码器相关文件和目录
        print("📁 复制空间编码器相关文件...")
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
                    copied_spatial_items.append(f"目录: {item_name}")
                else:
                    # 复制文件
                    shutil.copy2(src_path, dst_path)
                    copied_spatial_items.append(f"文件: {item_name}")
        
        if copied_spatial_items:
            print(f"   ✅ 复制了空间编码器相关项: {copied_spatial_items}")
        else:
            print("   ⚠️  未找到空间编码器相关文件，请手动确认")
            
        # 额外检查：列出原始目录中的所有文件，帮助识别遗漏的空间编码器文件
        print("📋 原始模型目录内容检查:")
        try:
            all_items = os.listdir(original_model_path)
            for item in all_items:
                item_path = os.path.join(original_model_path, item)
                if os.path.isdir(item_path):
                    print(f"   📁 目录: {item}")
                else:
                    file_size = os.path.getsize(item_path) / (1024*1024)  # MB
                    print(f"   📄 文件: {item} ({file_size:.1f} MB)")
        except Exception as e:
            print(f"   ❌ 无法列出目录内容: {e}")
        
        print("✅ 模型转换完成!")
        
        # 🔥 关键修复9：简化验证，避免重新加载大模型
        print("🔍 验证转换结果...")
        try:
            # 只验证文件是否存在和state_dict是否可加载
            saved_state_dict = torch.load(
                os.path.join(output_model_path, "pytorch_model.bin"), 
                map_location='cpu'
            )
            print(f"   ✅ 转换后的模型state_dict可以正常加载，参数数量: {len(saved_state_dict)}")
            
            # 检查关键参数是否存在
            key_params = [
                "global_trajectory_expert.model.layers.0.self_attn.q_proj.weight",
                "global_trajectory_in_proj.weight"
            ]
            
            missing_key_params = [key for key in key_params if key not in saved_state_dict]
            if missing_key_params:
                print(f"   ⚠️  缺少关键参数: {missing_key_params}")
            else:
                print("   ✅ 关键参数验证通过")
                
            del saved_state_dict
            
        except Exception as e:
            print(f"   ❌ 验证失败: {e}")
        
        # 🔥 关键修复10：最终清理
        del new_state_dict
        torch.cuda.empty_cache()
        gc.collect()
        
        return True
        
    except Exception as e:
        print(f"❌ 转换过程中出现错误: {e}")
        # 清理内存
        torch.cuda.empty_cache()
        gc.collect()
        raise e


def main():
    """主函数"""
    original_model_path = "/home/pi0_model_checkpoint/pytorch/pi0_base"
    output_model_path = "/home/pi0_model_checkpoint/pytorch/Hi3R_base"
    
    # 🔥 关键修复11：检查路径和磁盘空间
    if not os.path.exists(original_model_path):
        print(f"❌ 原始模型路径不存在: {original_model_path}")
        return
    
    # 检查磁盘空间（粗略估计）
    import shutil
    _, _, free_space = shutil.disk_usage(os.path.dirname(output_model_path))
    free_gb = free_space // (1024**3)
    print(f"📁 输出路径可用空间: {free_gb} GB")
    
    if free_gb < 10:  # 至少需要10GB空间
        print("⚠️  警告：磁盘空间可能不足！")
    
    # 🔥 关键修复12：设置环境变量减少内存使用
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    
    print("🚀 开始模型转换...")
    print(f"💾 当前可用显存: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB" if torch.cuda.is_available() else "CPU模式")
    
    try:
        success = convert_pi0_to_global_head(original_model_path, output_model_path)
        if success:
            print("🎉 转换成功完成！")
        else:
            print("❌ 转换失败！")
    except Exception as e:
        print(f"❌ 转换过程出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()