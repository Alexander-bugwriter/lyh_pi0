import argparse
import os
import sys
import torch
import pickle
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# LeRobot imports
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from utils.dataset_config import get_dataset_info, generate_delta_timestamps
from torchvision.transforms.v2 import Resize

# 你的项目导入
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from V3R_pi0.multimodal_spatial_encoder.cut3r_spatial_encoder import Cut3rSpatialTower, Cut3rSpatialConfig


def load_dataset(repo_id, root, image_size=224, action_horizon=50, debug_episodes=None):
    """加载LeRobot数据集"""
    print(f"加载数据集: {repo_id or root}")
    
    episodes = None
    if debug_episodes:
        episodes = list(range(debug_episodes))
        print(f"调试模式：只加载前 {debug_episodes} 个episodes")
    
    image_transforms = Resize((image_size, image_size))
    info = get_dataset_info(root if root else repo_id)
    delta_timestamps = generate_delta_timestamps(info['fps'], info['features'], action_horizon)
    
    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=root,
        image_transforms=image_transforms,
        delta_timestamps=delta_timestamps,
        episodes=episodes
    )
    
    print(f"数据集加载成功，共 {len(dataset)} 条数据")
    return dataset


def group_by_episode(dataset):
    """按episode分组并按frame_index排序"""
    print("📊 按episode分组数据...")
    
    episodes = defaultdict(list)
    
    for idx in tqdm(range(len(dataset)), desc="分组数据"):
        item = dataset[idx]
        episode_idx = item['episode_index'].item()
        episodes[episode_idx].append(item)
    
    # 按frame_index排序每个episode
    for episode_idx in episodes:
        episodes[episode_idx].sort(key=lambda x: x['frame_index'].item())
    
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


def extract_episode_features(spatial_tower, episode_data, episode_id, device='cuda:0'):
    """提取单个episode的特征"""
    print(f"处理Episode {episode_id}: {len(episode_data)} 帧")
    
    # 重置CUT3R状态
    if hasattr(spatial_tower, 'reset_state'):
        spatial_tower.reset_state()
    
    episode_features = {
        'episode_id': episode_id,
        'num_frames': len(episode_data),
        'features': []
    }
    
    with torch.no_grad():
        for frame_data in tqdm(episode_data, desc=f"Episode {episode_id}"):
            # 获取图像 (n_cameras, 3, H, W)
            images = frame_data['image'].to(device=device, dtype=torch.float16)
            
            # 提取base和wrist图像
            base_image = images[0:1]  # (1, 3, H, W)
            wrist_image = images[1:2] if images.shape[0] > 1 else images[0:1]  # (1, 3, H, W)
            
            # 🔥 按顺序通过CUT3R提取特征
            base_camera_tokens, base_patch_tokens = spatial_tower(base_image)
            wrist_camera_tokens, wrist_patch_tokens = spatial_tower(wrist_image)
            
            # 保存帧特征
            frame_features = {
                'frame_index': frame_data['frame_index'].item(),
                'episode_index': frame_data['episode_index'].item(),
                'timestamp': frame_data['timestamp'].item(),
                
                # base相机特征
                'base_camera_tokens': base_camera_tokens.cpu() if base_camera_tokens is not None else None,
                'base_patch_tokens': base_patch_tokens.cpu() if base_patch_tokens is not None else None,
                
                # wrist相机特征
                'wrist_camera_tokens': wrist_camera_tokens.cpu() if wrist_camera_tokens is not None else None,
                'wrist_patch_tokens': wrist_patch_tokens.cpu() if wrist_patch_tokens is not None else None,
            }
            
            episode_features['features'].append(frame_features)
    
    return episode_features


def main():
    parser = argparse.ArgumentParser()
    
    # 数据参数
    parser.add_argument("--data_repo_id", type=str, default=None)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    
    # 模型参数
    parser.add_argument("--cut3r_weights_path", type=str, required=True)
    
    # 其他参数
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--action_horizon", type=int, default=50)
    parser.add_argument("--debug_episodes", type=int, default=None)
    
    args = parser.parse_args()
    
    print("LeRobot CUT3R特征预提取")
    print(f"输出目录: {args.output_dir}")
    
    # 加载数据集
    dataset = load_dataset(
        args.data_repo_id, 
        args.data_root, 
        args.image_size, 
        args.action_horizon, 
        args.debug_episodes
    )
    
    # 按episode分组
    episodes_dict = group_by_episode(dataset)
    
    # 初始化CUT3R
    spatial_tower = init_cut3r(args.cut3r_weights_path, args.device)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 处理每个episode
    for episode_id, episode_data in episodes_dict.items():
        # 提取特征
        episode_features = extract_episode_features(
            spatial_tower, episode_data, episode_id, args.device
        )
        
        # 保存特征
        save_path = os.path.join(args.output_dir, f"episode_spatial_features_{episode_id:06d}.pkl")
        with open(save_path, 'wb') as f:
            pickle.dump(episode_features, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"Episode {episode_id} 保存到: {save_path}")
    
    print(f"特征提取完成！处理了 {len(episodes_dict)} 个episodes")


if __name__ == "__main__":
    main()
