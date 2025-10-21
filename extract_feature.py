# python extract_feature_1020.py \
#     --data_root /path/to/data \
#     --output_dir ./features \
#     --cut3r_weights_path /path/to/cut3r \
#     --save_history_features \
#     --num_history_frames 3


import argparse
import os
import sys
import torch
import pickle
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import gc
# LeRobot imports
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from utils.dataset_config import get_dataset_info, generate_delta_timestamps
from utils.normalizers import Normalizer  # 🔥 添加归一化器
from torchvision.transforms.v2 import Resize

# 你的项目导入
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from V3R_pi0.multimodal_spatial_encoder.cut3r_spatial_encoder import Cut3rSpatialTower, Cut3rSpatialConfig
from utils.spatiotemporal_lerobot_dataset import LerobotPI0Dataset

def compute_history_indices(current_t, num_history_frames=3):
    """
    🔥 核心函数：计算历史帧索引
    
    采样策略：
    - 如果 t >= 30: [t-30, t-20, t-10]
    - 如果 3 <= t < 30: [0, t//3, 2*t//3]
    - 如果 t < 3: [0, 1, t] 或者根据t的大小填充
    
    Args:
        current_t: 当前帧索引
        num_history_frames: 历史帧数量（默认3）
    
    Returns:
        history_indices: List[int] 历史帧的索引列表
    """
    if current_t >= 30:
        # 策略1: 固定间隔采样
        history_indices = [current_t - 30, current_t - 20, current_t - 10]
    elif current_t >= 3:
        # 策略2: 均匀分布采样
        history_indices = [0, current_t // 3, (2 * current_t) // 3]
    else:
        # 策略3: 填充策略
        if current_t == 0:
            history_indices = [0, 0, 0]
        elif current_t == 1:
            history_indices = [0, 1, 1]
        elif current_t == 2:
            history_indices = [0, 1, 2]
        else:
            history_indices = [0, 1, current_t]
    
    return history_indices

def group_by_episode(dataset):
    """按episode分组并按frame_index排序"""
    print("📊 按episode分组数据...")
    
    episodes = defaultdict(list)
    
    for idx in tqdm(range(len(dataset)), desc="分组数据"):
        item = dataset[idx]
        episode_idx = item['episode_index'].item()
        episodes[episode_idx].append((idx, item))  # 保存原始索引
    
    # 按frame_index排序每个episode
    for episode_idx in episodes:
        episodes[episode_idx].sort(key=lambda x: x[1]['frame_index'].item())
    
    print(f"分组完成：{len(episodes)} 个episodes")
    return dict(episodes)


def init_cut3r(cut3r_weights_path, device='cuda:0'):
    """初始化CUT3R模型"""
    print(f"初始化CUT3R模型...")
    
    spatial_config = Cut3rSpatialConfig(
        weights_path=cut3r_weights_path,
        spatial_tower_select_feature="all",
        spatial_tower_select_layer=-1,
        export_point_cloud=False
    )
    
    spatial_tower = Cut3rSpatialTower(
        spatial_tower='cut3r',
        spatial_tower_cfg=spatial_config,
        delay_load=False
    )
    
    spatial_tower.to(device=device, dtype=torch.float16)
    spatial_tower.eval()
    
    print(f"CUT3R初始化完成")
    return spatial_tower

def extract_episode_features_with_history_indices(spatial_tower, episode_data, episode_id, 
                                                   dataset, save_history, num_history_frames=3,
                                                   device='cuda:0'):
    """
    🔥 提取单个episode的特征 - 保存历史索引而非历史数据
    
    Args:
        spatial_tower: CUT3R模型
        episode_data: episode的所有帧数据
        episode_id: episode ID
        dataset: 数据集
        save_history: 是否保存历史信息
        num_history_frames: 历史帧数量
        device: 设备
    """
    print(f"处理Episode {episode_id}: {len(episode_data)} 帧")
    
    # 重置CUT3R状态
    if hasattr(spatial_tower, 'reset_state'):
        spatial_tower.reset_state(batch_size=2)
    
    episode_features = {
        'episode_id': episode_id,
        'num_frames': len(episode_data),
        'num_history_frames': num_history_frames,
        'features': []
    }
    
    with torch.no_grad():
        for frame_idx, (dataset_idx, frame_data) in enumerate(tqdm(episode_data, desc=f"Episode {episode_id}")):
            current_frame_index = frame_data['frame_index'].item()
            
            # 🔥 使用 LerobotPI0Dataset 的预处理（返回 uint8 [0, 255]）
            processed_item = dataset[dataset_idx]
            images_dict = processed_item["image"]
            
            # 🔥 归一化到 [-1, 1]
            base_image_uint8 = images_dict["base_0_rgb"]
            base_image = base_image_uint8.to(device=device, dtype=torch.float32)
            base_image = (base_image / 127.5) - 1.0
            
            wrist_image_uint8 = images_dict["left_wrist_0_rgb"]
            wrist_image = wrist_image_uint8.to(device=device, dtype=torch.float32)
            wrist_image = (wrist_image / 127.5) - 1.0
            
            # 🔥 拼接：(C,H,W) -> (1,1,C,H,W) base + (1,1,C,H,W) wrist -> (1,2,C,H,W)
            base_image = base_image.unsqueeze(0).unsqueeze(0)
            wrist_image = wrist_image.unsqueeze(0).unsqueeze(0)
            spatial_batch = torch.cat([base_image, wrist_image], dim=1)
            
            # 🔥 调用 CUT3R
            camera_tokens, patch_tokens = spatial_tower(spatial_batch)
            camera_tokens = camera_tokens.to(base_image.dtype)
            patch_tokens = patch_tokens.to(base_image.dtype)
            
            # 🔥 在 B 维度切分
            base_camera_tokens = camera_tokens[0:1]
            base_patch_tokens = patch_tokens[0:1]
            wrist_camera_tokens = camera_tokens[1:2]
            wrist_patch_tokens = patch_tokens[1:2]
            
            # 🔥 核心改动：计算历史帧索引而不是存储历史数据
            history_info = None
            if save_history:
                history_indices = compute_history_indices(current_frame_index, num_history_frames)
                history_info = {
                    'num_history_frames': num_history_frames,
                    'history_indices': history_indices,  # 🔥 只保存索引
                    'current_frame_index': current_frame_index,
                }
            
            # 保存帧特征
            frame_features = {
                # 基本信息
                'frame_index': current_frame_index,
                'episode_index': frame_data['episode_index'].item(),
                'timestamp': frame_data['timestamp'].item(),
                'dataset_idx': dataset_idx,
                
                # 🔥 当前帧的spatial tokens
                'base_camera_tokens': base_camera_tokens.cpu(),
                'base_patch_tokens': base_patch_tokens.cpu(),
                'wrist_camera_tokens': wrist_camera_tokens.cpu(),
                'wrist_patch_tokens': wrist_patch_tokens.cpu(),
            }
            
            # 🔥 只在需要时添加 history_info（仅包含索引）
            if history_info is not None:
                frame_features['history_info'] = history_info
            
            episode_features['features'].append(frame_features)
    
    gc.collect()
    
    return episode_features

def validate_history_indices():
    """🧪 测试历史索引计算逻辑"""
    print("🧪 测试历史索引计算...")
    
    test_cases = [
        (0, "t=0"),
        (1, "t=1"),
        (2, "t=2"),
        (5, "t=5"),
        (15, "t=15"),
        (29, "t=29"),
        (30, "t=30"),
        (50, "t=50"),
        (100, "t=100"),
    ]
    
    for t, desc in test_cases:
        indices = compute_history_indices(t, num_history_frames=3)
        print(f"  {desc}: 历史索引 = {indices}")
    
    print("✅ 历史索引测试完成")

def validate_preprocessing_with_history(dataset, device='cuda:0'):
    """验证预处理流程的正确性"""
    print("验证预处理流程...")
    
    sample_item = dataset[0]
    
    print("LerobotPI0Dataset 输出:")
    if "image" in sample_item:
        for key, img in sample_item["image"].items():
            print(f"  {key}: shape={img.shape}, dtype={img.dtype}, min={img.min()}, max={img.max()}")
    
    print("\n归一化到 [-1, 1]:")
    base_image_uint8 = sample_item["image"]["base_0_rgb"]
    base_image_normalized = (base_image_uint8.to(device=device, dtype=torch.float32) / 127.5) - 1.0
    print(f"  归一化后: min={base_image_normalized.min():.3f}, max={base_image_normalized.max():.3f}")
    print(f"  shape={base_image_normalized.shape}, dtype={base_image_normalized.dtype}")
    
    print("\n✅ 预处理验证完成")


def main():
   
    parser = argparse.ArgumentParser()
    
    # 数据参数
    parser.add_argument("--data_repo_id", type=str, default=None)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    
    # 模型参数
    parser.add_argument("--cut3r_weights_path", type=str, required=True)
    
    # 🔥 历史帧参数
    parser.add_argument("--num_history_frames", type=int, default=3,
                       help="历史帧数量（默认3）")
    
    # 其他参数
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--action_horizon", type=int, default=50)
    parser.add_argument("--debug_episodes", type=int, default=None)
    parser.add_argument("--save_history_features", action="store_true",
                       help="是否保存历史索引信息")
    parser.add_argument("--test_history_indices", action="store_true",
                       help="测试历史索引计算逻辑")
    parser.add_argument("--validate_preprocessing", action="store_true", 
                       help="验证预处理流程的正确性")    
    args = parser.parse_args()
    
    print("LeRobot CUT3R特征预提取（保存历史索引）")
    print(f"输出目录: {args.output_dir}")
    if args.save_history_features:
        print(f"历史帧数量: {args.num_history_frames}")
        print("📌 保存历史帧索引（不保存完整历史数据）")
    else:
        print("仅保存当前时刻特征，没有历史信息")
    
    # 可选：测试历史索引计算
    if args.test_history_indices:
        validate_history_indices()
    # 🔥 关键：加载数据集和归一化器
    # dataset, normalizer = load_dataset_with_normalizer(
    #     args.data_repo_id, 
    #     args.data_root, 
    #     args.image_size, 
    #     args.action_horizon, 
    #     args.debug_episodes
    # )
    dataset = LerobotPI0Dataset(
        repo_id=args.data_repo_id,
        root=args.data_root,
        image_size=args.image_size,
        action_horizon=args.action_horizon,
        dataset_fps=10.0,
        debug_episodes=args.debug_episodes
    )
    if args.validate_preprocessing:
        validate_preprocessing_with_history(dataset, args.device)

    
    print(f"数据集加载成功，共 {len(dataset)} 条数据")
    
   
    # 按episode分组
    episodes_dict = group_by_episode(dataset)
    
    # 初始化CUT3R
    spatial_tower = init_cut3r(args.cut3r_weights_path, args.device)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 处理每个episode
    for episode_id, episode_data in episodes_dict.items():
        # 🔥 关键：提取包含历史信息的特征
        # episode_features = extract_episode_features_with_history(
        #     spatial_tower, episode_data, episode_id, dataset, args.save_history_features,
        #     args.max_history_frames, args.device
        # )
        episode_features = extract_episode_features_with_history_indices(
            spatial_tower, episode_data, episode_id, dataset, 
            args.save_history_features, args.num_history_frames, args.device
        )
        if args.save_history_features:
            save_path = os.path.join(args.output_dir, 
                                    f"episode_spatial_features_with_history_{episode_id:06d}.pkl")
        else:
            save_path = os.path.join(args.output_dir, 
                                    f"episode_spatial_features_{episode_id:06d}.pkl")
        
        with open(save_path, 'wb') as f:
            pickle.dump(episode_features, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"Episode {episode_id} 保存到: {save_path}")
        # 保存特征
        # if args.save_history_features:
        #     save_path = os.path.join(args.output_dir, f"episode_spatial_features_with_history_{episode_id:06d}.pkl")
        # else:
        #     save_path = os.path.join(args.output_dir, f"episode_spatial_features_{episode_id:06d}.pkl")
        # with open(save_path, 'wb') as f:
        #     pickle.dump(episode_features, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # print(f"Episode {episode_id} 保存到: {save_path}")
    
    print(f"✅ 特征提取完成！处理了 {len(episodes_dict)} 个episodes")
    print(f"🔥 重要提醒：现在每帧只保存历史索引（而非完整历史数据），极大节省存储空间")
    print(f"📋 历史索引采样策略:")
    print(f"  - t >= 30: [t-30, t-20, t-10]")
    print(f"  - 3 <= t < 30: [0, t//3, 2*t//3]")
    print(f"  - t < 3: 填充策略 [0, 1, t]")
    print(f"📌 使用说明：")
    print(f"  - 训练时：根据索引从dataset中动态加载历史帧")
    print(f"  - 推理时：使用UnlimitedHistoryBuffer类管理实时历史")
    
    # print(f"✅ 带历史信息的特征提取完成！处理了 {len(episodes_dict)} 个episodes")
    # print(f"🔥 重要提醒：现在每帧都包含历史帧信息，可以直接用于训练时的历史特征融合")
    # print(f"📝 使用说明：")
    # print(f"  - 训练时：直接在embed_image_with_preprocessing_feature中融合历史特征")
    # print(f"  - 推理时：使用embed_image方法和UnlimitedHistoryBuffer类")


if __name__ == "__main__":
    main()



