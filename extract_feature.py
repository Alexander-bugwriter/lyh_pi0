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


def load_dataset_with_normalizer(repo_id, root, image_size=224, action_horizon=50, debug_episodes=None):
    """加载LeRobot数据集并设置归一化器"""
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
    
    # 🔥 关键：创建归一化器，与训练时保持一致
    normalizer = Normalizer(
        norm_stats=dataset.meta.stats,
        norm_type={
            "image": "identity",
            "wrist_image": "identity", 
            "state": "meanstd",
            "actions": "meanstd",
        }
    )
    print("归一化器设置完成")
    
    return dataset, normalizer


def normalize_and_prepare_images(item, normalizer):
    """
    🔥 关键函数：应用与训练时完全相同的图像预处理流程
    模拟 LerobotPI0Dataset.__getitem__ 中的图像处理逻辑
    """
    # 1. 先归一化原始数据
    normalized_item = normalizer.normalize(item)
    
    # 2. 处理图像，转换为uint8格式
    images = {}
    
    # 基础相机 (必需)
    if "image" in normalized_item:
        base_image = normalized_item["image"]
        while base_image.dim() > 3 and 1 in base_image.shape:
            base_image = base_image.squeeze()
        base_image = (base_image * 255).to(torch.uint8)  # 🔥 关键：转换为uint8
        images["base_0_rgb"] = base_image
    
    # 手腕相机 (可选)
    if "wrist_image" in normalized_item:
        wrist_image = normalized_item["wrist_image"]
        while wrist_image.dim() > 3 and 1 in wrist_image.shape:
            wrist_image = wrist_image.squeeze()
        wrist_image = (wrist_image * 255).to(torch.uint8)  # 🔥 关键：转换为uint8
        images["left_wrist_0_rgb"] = wrist_image
    
    return images


def apply_model_image_preprocessing(images, device='cuda:0'):
    """
    🔥 应用与模型 prepare_images 方法完全相同的预处理
    将uint8图像归一化到[-1, 1]范围
    """
    processed_images = []
    IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    
    for key in IMAGE_KEYS:
        if key in images:
            img = images[key].to(device=device, dtype=torch.float32)
            # 🔥 关键：应用与模型相同的归一化 img.to(dtype) / 127.5 - 1.0
            img = img / 127.5 - 1.0  # 归一化到[-1, 1]
            processed_images.append(img)
        else:
            # 如果某个相机不存在，创建零填充
            if len(processed_images) > 0:
                dummy_img = torch.full_like(processed_images[0], fill_value=-1.0)
                processed_images.append(dummy_img)
    
    return processed_images


def group_by_episode(dataset):
    """按episode分组并按frame_index排序"""
    print("📊 按episode分组数据...")
    
    episodes = defaultdict(list)
    
    for idx in tqdm(range(len(dataset)), desc="分组数据"):
        item = dataset[idx]
        episode_idx = item['episode_index'].item()
        episodes[episode_idx].append((idx, item))  # 🔥 保存原始索引
    
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


class HistoryBuffer:
    """历史帧缓存管理器 - 用于提取特征时管理历史信息"""
    
    def __init__(self, max_history_frames=5):
        self.max_history_frames = max_history_frames
        self.buffer = []  # 存储 (frame_index, base_image_uint8, spatial_tokens) 元组
    
    def add_frame(self, frame_index, base_image_uint8, spatial_tokens):
        """添加新帧到历史缓存"""
        frame_data = {
            'frame_index': frame_index,
            'base_image_uint8': base_image_uint8.clone().cpu(),  # 保存uint8格式的base image
            'base_camera_tokens': spatial_tokens['base_camera_tokens'].clone().cpu() if spatial_tokens['base_camera_tokens'] is not None else None,
            'base_patch_tokens': spatial_tokens['base_patch_tokens'].clone().cpu() if spatial_tokens['base_patch_tokens'] is not None else None,
        }
        
        self.buffer.append(frame_data)
    
    def get_history_frames(self, current_frame_index):
        """获取当前帧的历史帧信息（不包括当前帧）"""
        # 只获取比当前帧更早的帧
        history_frames = [f for f in self.buffer if f['frame_index'] < current_frame_index]
        
        # 限制历史帧数量
        if len(history_frames) > self.max_history_frames:
            # 采用均匀采样策略
            indices = np.linspace(0, len(history_frames) - 1, self.max_history_frames, dtype=int)
            history_frames = [history_frames[i] for i in indices]
        
        return history_frames
    
    def clear(self):
        """清空缓存（episode边界时调用）"""
        self.buffer.clear()


def extract_episode_features_with_history(spatial_tower, episode_data, episode_id, normalizer, save_history,
                                        max_history_frames=5, device='cuda:0'):
    """提取单个episode的特征 - 包含历史信息"""
    print(f"处理Episode {episode_id}: {len(episode_data)} 帧 ")
    
    # 重置CUT3R状态
    if hasattr(spatial_tower, 'reset_state'):
        spatial_tower.reset_state()
    
    # 创建历史缓存管理器
    history_buffer = HistoryBuffer(max_history_frames) if save_history else None
    
    episode_features = {
        'episode_id': episode_id,
        'num_frames': len(episode_data),
        'max_history_frames': max_history_frames,
        'features': []
    }
    
    with torch.no_grad():
        for frame_idx, (dataset_idx, frame_data) in enumerate(tqdm(episode_data, desc=f"Episode {episode_id}")):
            
            current_frame_index = frame_data['frame_index'].item()
            
            # 🔥 关键步骤1：应用与训练时相同的归一化和图像预处理
            images_dict = normalize_and_prepare_images(frame_data, normalizer)
            
            # 🔥 关键步骤2：应用与模型相同的图像预处理（归一化到[-1,1]）
            processed_images = apply_model_image_preprocessing(images_dict, device)
            
            # 🔥 关键步骤3：确保CUT3R接收到正确格式的图像
            if len(processed_images) >= 1:
                base_image = processed_images[0].to(device=device, dtype=torch.float16)
                wrist_image = processed_images[1].to(device=device, dtype=torch.float16) if len(processed_images) > 1 else base_image
            else:
                print(f"警告：Episode {episode_id} frame {current_frame_index} 缺少图像数据")
                continue
            
            # 🔥 按顺序通过CUT3R提取特征
            base_camera_tokens, base_patch_tokens = spatial_tower(base_image.unsqueeze(0))
            wrist_camera_tokens, wrist_patch_tokens = spatial_tower(wrist_image.unsqueeze(0))
            
            # 🔥 简化：只有启用历史特征时才获取历史信息
            history_info = None
            if save_history and history_buffer is not None:
                history_frames = history_buffer.get_history_frames(current_frame_index)
                history_info = {
                    'num_history_frames': len(history_frames),
                    'max_history_frames': max_history_frames,
                    'history_frames': history_frames
                }
            
            # 保存帧特征
            frame_features = {
                # 基本信息
                'frame_index': current_frame_index,
                'episode_index': frame_data['episode_index'].item(),
                'timestamp': frame_data['timestamp'].item(),
                'dataset_idx': dataset_idx,
                
                # 🔥 当前帧的spatial tokens
                'base_camera_tokens': base_camera_tokens.cpu() if base_camera_tokens is not None else None,
                'base_patch_tokens': base_patch_tokens.cpu() if base_patch_tokens is not None else None,
                'wrist_camera_tokens': wrist_camera_tokens.cpu() if wrist_camera_tokens is not None else None,
                'wrist_patch_tokens': wrist_patch_tokens.cpu() if wrist_patch_tokens is not None else None,
            }
            
            # 🔥 只有启用历史特征时才添加history_info
            if history_info is not None:
                frame_features['history_info'] = history_info
            
            episode_features['features'].append(frame_features)
            
            # 🔥 只有启用历史特征时才更新历史缓存
            if save_history and history_buffer is not None:
                current_spatial_tokens = {
                    'base_camera_tokens': base_camera_tokens,
                    'base_patch_tokens': base_patch_tokens,
                }
                history_buffer.add_frame(
                    current_frame_index, 
                    images_dict["base_0_rgb"],
                    current_spatial_tokens
                )
    if history_buffer:
        history_buffer.clear()
        del history_buffer
        
        
    
    gc.collect()
    
    return episode_features


def validate_preprocessing_with_history(dataset, normalizer, device='cuda:0'):
    """验证预处理流程的正确性（包含历史信息检查）"""
    print("🔍 验证预处理流程（包含历史信息）...")
    
    # 取一个样本进行验证
    sample_item = dataset[0]
    
    print("原始数据范围:")
    if "image" in sample_item:
        img = sample_item["image"]
        wrist_img = sample_item.get("wrist_image")
        print(f"  image: min={img.min():.3f}, max={img.max():.3f}, shape={img.shape}, dtype={img.dtype}")
        if wrist_img is not None:
            print(f"  wrist_image: min={wrist_img.min():.3f}, max={wrist_img.max():.3f}, shape={wrist_img.shape}, dtype={wrist_img.dtype}")
    
    # 应用归一化
    images_dict = normalize_and_prepare_images(sample_item, normalizer)
    print("归一化后 (uint8):")
    for key, img in images_dict.items():
        print(f"  {key}: min={img.min()}, max={img.max()}, shape={img.shape}, dtype={img.dtype}")
    
    # 应用模型预处理
    processed_images = apply_model_image_preprocessing(images_dict, device)
    print("模型预处理后 ([-1,1]):")
    for i, img in enumerate(processed_images):
        print(f"  image_{i}: min={img.min():.3f}, max={img.max():.3f}, shape={img.shape}, dtype={img.dtype}")
    
    print("✅ 预处理验证完成")


def test_history_buffer():
    """测试历史缓存功能"""
    print("🧪 测试历史缓存功能...")
    
    buffer = HistoryBuffer(max_history_frames=3)
    
    # 模拟添加帧
    for i in range(7):
        fake_image = torch.randint(0, 255, (3, 224, 224), dtype=torch.uint8)
        fake_tokens = {
            'base_camera_tokens': torch.randn(1, 1, 768),
            'base_patch_tokens': torch.randn(1, 729, 768)
        }
        buffer.add_frame(i, fake_image, fake_tokens)
        
        history = buffer.get_history_frames(i)
        print(f"  帧 {i}: 历史帧数量 = {len(history)}")
        if history:
            history_indices = [h['frame_index'] for h in history]
            print(f"    历史帧索引: {history_indices}")
    
    print("✅ 历史缓存测试完成")


def main():
    parser = argparse.ArgumentParser()
    
    # 数据参数
    parser.add_argument("--data_repo_id", type=str, default=None)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    
    # 模型参数
    parser.add_argument("--cut3r_weights_path", type=str, required=True)
    
    # 🔥 新增：历史帧相关参数
    parser.add_argument("--max_history_frames", type=int, default=3,
                       help="最大历史帧数量")
    
    # 其他参数
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--action_horizon", type=int, default=50)
    parser.add_argument("--debug_episodes", type=int, default=None)
    parser.add_argument("--save_history_features", action="store_true",
                   help="是否保存历史特征（不指定则只保存当前帧）")
    parser.add_argument("--validate_preprocessing", action="store_true", 
                       help="验证预处理流程的正确性")
    parser.add_argument("--test_history_buffer", action="store_true",
                       help="测试历史缓存功能")
    
    args = parser.parse_args()
    
    print("LeRobot CUT3R特征预提取（包含历史信息）")
    print(f"输出目录: {args.output_dir}")
    if args.save_history_features:
        print(f"最大历史帧数: {args.max_history_frames}")
    else:
        print("保存当前时刻特征，没有历史特征")
    
    # 可选：测试历史缓存功能
    if args.test_history_buffer:
        test_history_buffer()
        return
    
    # 🔥 关键：加载数据集和归一化器
    dataset, normalizer = load_dataset_with_normalizer(
        args.data_repo_id, 
        args.data_root, 
        args.image_size, 
        args.action_horizon, 
        args.debug_episodes
    )
    
    # 可选：验证预处理流程
    if args.validate_preprocessing:
        validate_preprocessing_with_history(dataset, normalizer, args.device)
    
    # 按episode分组
    episodes_dict = group_by_episode(dataset)
    
    # 初始化CUT3R
    spatial_tower = init_cut3r(args.cut3r_weights_path, args.device)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 处理每个episode
    for episode_id, episode_data in episodes_dict.items():
        # 🔥 关键：提取包含历史信息的特征
        episode_features = extract_episode_features_with_history(
            spatial_tower, episode_data, episode_id, normalizer,args.save_history_features, 
            args.max_history_frames, args.device
        )
        
        # 保存特征
        if args.save_history_features:
            save_path = os.path.join(args.output_dir, f"episode_spatial_features_with_history_{episode_id:06d}.pkl")
        else:
            save_path = os.path.join(args.output_dir, f"episode_spatial_features_{episode_id:06d}.pkl")
        with open(save_path, 'wb') as f:
            pickle.dump(episode_features, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"Episode {episode_id} 保存到: {save_path}")
        
    
    print(f"✅ 带历史信息的特征提取完成！处理了 {len(episodes_dict)} 个episodes")
    print(f"🔥 重要提醒：现在每帧都包含历史帧信息，可以直接用于训练时的历史特征融合")
    print(f"📝 使用说明：")
    print(f"  - 训练时：直接在embed_image_with_preprocessing_feature中融合历史特征")
    print(f"  - 推理时：使用embed_image方法和UnlimitedHistoryBuffer类")


if __name__ == "__main__":
    main()
