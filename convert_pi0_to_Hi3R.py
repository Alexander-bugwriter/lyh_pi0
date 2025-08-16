import torch
import os
import json
from pathlib import Path
from V3R_pi0 import PI0Policy
from V3R_pi0.Hi3R_VLA import PI0PolicyWithGlobalHead


def convert_pi0_to_global_head(original_model_path: str, output_model_path: str):
    """
    将原始PI0模型转换为带全局轨迹头的模型
    
    Args:
        original_model_path: 原始PI0模型路径
        output_model_path: 输出模型路径
    """
    print(f"🔄 开始转换模型: {original_model_path} -> {output_model_path}")
    
    # 1. 加载原始模型
    print("📥 加载原始PI0模型...")
    # original_policy = PI0Policy.from_pretrained(original_model_path)
  
    config_path = os.path.join(original_model_path, "config.json")
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    config_dict['type'] = 'pi0'  # 添加必需的type字段
    config_dict['device'] = 'cuda'  # 修复设备名称格式

    # 保存修改后的配置
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)

    # 然后加载模型
    original_policy = PI0Policy.from_pretrained(original_model_path)
    original_state_dict = original_policy.state_dict()
    
    # 2. 创建新的带全局头的模型
    print("🔨 创建带全局头的新模型...")
    config = original_policy.config
    new_policy = PI0PolicyWithGlobalHead(config)
    new_state_dict = new_policy.state_dict()
    
    # 3. 复制参数
    print("📋 复制模型参数...")
    
    # 3.1 复制完全相同的参数
    common_keys = set(original_state_dict.keys()) & set(new_state_dict.keys())
    print(f"   共同参数数量: {len(common_keys)}")
    
    for key in common_keys:
        new_state_dict[key] = original_state_dict[key].clone()
        print(f"   ✅ 复制: {key}")
    
    # 3.2 复制action expert参数到global trajectory expert
    print("🔄 复制action expert到global trajectory expert...")
    
    action_expert_prefix = "model.gemma_expert."
    global_expert_prefix = "model.global_trajectory_expert."
    
    copied_count = 0
    for key in original_state_dict.keys():
        if key.startswith(action_expert_prefix):
            # 将action expert的参数复制到global trajectory expert
            global_key = key.replace(action_expert_prefix, global_expert_prefix)
            if global_key in new_state_dict:
                new_state_dict[global_key] = original_state_dict[key].clone()
                print(f"   ✅ 复制: {key} -> {global_key}")
                copied_count += 1
    
    print(f"   复制了 {copied_count} 个action expert参数到global expert")
    
    # 3.3 复制action投影参数到global trajectory投影
    action_proj_keys = [
        "model.action_in_proj.weight",
        "model.action_in_proj.bias"
    ]
    global_proj_keys = [
        "model.global_trajectory_in_proj.weight", 
        "model.global_trajectory_in_proj.bias"
    ]
    
    for orig_key, new_key in zip(action_proj_keys, global_proj_keys):
        if orig_key in original_state_dict and new_key in new_state_dict:
            new_state_dict[new_key] = original_state_dict[orig_key].clone()
            print(f"   ✅ 复制投影参数: {orig_key} -> {new_key}")
    
    # 4. 检查未初始化的参数
    print("🔍 检查未初始化的参数...")
    missing_keys = []
    for key in new_state_dict.keys():
        # 检查哪些参数没有被复制
        if key not in common_keys and not any(key.startswith(prefix) for prefix in [
            global_expert_prefix, "model.global_trajectory_in_proj."
        ]):
            missing_keys.append(key)
    
    if missing_keys:
        print(f"   ⚠️  以下参数将使用随机初始化: {missing_keys}")
    else:
        print("   ✅ 所有参数都已正确初始化")
    
    # 5. 加载转换后的参数
    new_policy.load_state_dict(new_state_dict)
    
    # 6. 保存转换后的模型
    print(f"💾 保存转换后的模型到: {output_model_path}")
    os.makedirs(output_model_path, exist_ok=True)
    
    # 保存模型权重
    torch.save(new_state_dict, os.path.join(output_model_path, "pytorch_model.bin"))
    
    # 保存配置（复用原始配置）
    config_path = os.path.join(original_model_path, "config.json")
    if os.path.exists(config_path):
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
            import shutil
            shutil.copy2(src_path, os.path.join(output_model_path, file_name))
    
    print("✅ 模型转换完成!")
    
    # 7. 验证转换结果
    print("🔍 验证转换结果...")
    try:
        # 尝试加载转换后的模型
        converted_policy = PI0PolicyWithGlobalHead.from_pretrained(output_model_path)
        print("   ✅ 转换后的模型可以正常加载")
        
        # 简单的前向传播测试
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        test_observation = {
            "image": {
                "base_0_rgb": torch.randint(0, 256, (1, 3, 224, 224), dtype=torch.uint8, device=device),
            },
            "state": torch.randn(1, 8, device=device) * 0.2,
            "prompt": ["test prompt"],
        }
        
        with torch.no_grad():
            action = converted_policy.select_action(test_observation)
            print(f"   ✅ 模型推理测试通过，输出shape: {action.shape}")
            
    except Exception as e:
        print(f"   ❌ 验证失败: {e}")
    
    return new_policy


def main():
    """主函数"""
    original_model_path = "/home/pi0_model_checkpoint/pytorch/pi0_base"
    output_model_path = "/home/pi0_model_checkpoint/pytorch/Hi3R_base"
    
    convert_pi0_to_global_head(original_model_path, output_model_path)


if __name__ == "__main__":
    main()