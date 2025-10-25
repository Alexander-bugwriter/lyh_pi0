from collections import defaultdict, OrderedDict
import os
import argparse
import torch
import pytorch_lightning as L
import json
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from torchvision.transforms.v2 import Compose, Resize
from typing import List, Dict
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import pickle
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import h5py
import numpy as np
from tqdm import tqdm

from .normalizers import Normalizer
from .dataset_config import get_dataset_info, generate_delta_timestamps
import time


class LerobotPI0Dataset(Dataset):
    """标准Lerobot格式数据集包装器"""
    
    def __init__(self, repo_id=None, root=None, image_size=224, action_horizon=50,dataset_fps=10.0,debug_episodes=None):
        print(f"加载Lerobot数据集: {repo_id}")
        episodes=None
        if debug_episodes:
            episodes = list(range(debug_episodes))
            print(f"调试模式：只加载前 {debug_episodes} 个episodes")
        # 添加缓存逻辑
        
        cache_file = f".dataset_cache_{debug_episodes if debug_episodes is not None else 'all'}.pkl"
        cache_path = os.path.join(root, cache_file)  # 数据集目录下

        print(f"DEBUG: cache_file = {cache_file}")
        print(f"DEBUG: cache_path = {cache_path}")
        print(f"DEBUG: 当前工作目录 = {os.getcwd()}")
        print(f"DEBUG: 缓存文件是否存在 = {os.path.exists(cache_path)}")
        if os.path.exists(cache_path):
            print(f"从缓存加载: {cache_path}")
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
                self.dataset = cache_data['dataset']
                self.normalizer = cache_data['normalizer']
                print(f"缓存加载成功，数据集长度: {len(self.dataset)}")
        else:
            
            image_transforms = Resize((image_size, image_size))
            info = get_dataset_info(root)
            delta_timestamps = generate_delta_timestamps(info['fps'], info['features'], action_horizon)

            self.dataset = LeRobotDataset(
                    repo_id=repo_id,
                    root=root,
                    image_transforms=image_transforms,
                    delta_timestamps=delta_timestamps,
                    episodes=episodes
                )
            print(f"数据集加载成功，共 {len(self.dataset)} 条数据")
            self.normalizer = Normalizer(
                norm_stats=self.dataset.meta.stats,
                norm_type={
                    "image": "identity",
                    "wrist_image": "identity",
                    "state": "meanstd",
                    "actions": "meanstd",
                }
            )
            with open(cache_path, 'wb') as f:
                pickle.dump({
                    'dataset': self.dataset,
                    'normalizer': self.normalizer
                }, f)
            print(f"已缓存到: {cache_path}")

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        normalized_item = self.normalizer.normalize(item)
        
        # 处理图像
        images = {}
        
        # 基础相机 (必需)
        if "image" in normalized_item:
            base_image = normalized_item["image"]
            while base_image.dim() > 3 and 1 in base_image.shape:
                base_image = base_image.squeeze()
            base_image = (base_image * 255).to(torch.uint8)
            images["base_0_rgb"] = base_image
        
        # 手腕相机 (可选)
        if "wrist_image" in normalized_item:
            wrist_image = normalized_item["wrist_image"]
            while wrist_image.dim() > 3 and 1 in wrist_image.shape:
                wrist_image = wrist_image.squeeze()
            wrist_image = (wrist_image * 255).to(torch.uint8)
            images["left_wrist_0_rgb"] = wrist_image
        
        # 任务指令
        task_text = item.get("task", "complete the task")
        if isinstance(task_text, str):
            prompt = [task_text]
        elif isinstance(task_text, (list, tuple)):
            prompt = [str(t) for t in task_text]
        else:
            prompt = [str(task_text)]
        
        # 构建返回结果
        return {
            "image": images,
            "state": normalized_item["state"][0],
            "action": normalized_item["actions"],
            "action_is_pad": normalized_item.get("action_is_pad", 
                torch.zeros_like(normalized_item["actions"][..., 0], dtype=torch.bool)
            ),
            "prompt": prompt,
            "episode_index": item["episode_index"],
        }


class Enhanced_LerobotPI0Dataset(LerobotPI0Dataset):
    """增强版数据集 - 支持spatial features和历史信息，自动重整为索引格式"""
    
    def __init__(self, repo_id=None, root=None, image_size=224, action_horizon=50,
                 dataset_fps=10.0, debug_episodes=None, spatial_features_dir=None, use_history=False):
        
        super().__init__(repo_id, root, image_size, action_horizon, dataset_fps, debug_episodes)
        
        self.spatial_features_dir = spatial_features_dir
        self.debug_episodes = debug_episodes
        self.use_history = use_history
        self.spatial_h5_file = None
        
        print(f"🔧 spatial_features_dir = {spatial_features_dir}")
        print(f"🔧 use_history = {use_history}")
        
        if spatial_features_dir:
            # 🔥 检查是否存在重整后的索引文件
            debug_suffix = f"_{self.debug_episodes}" if self.debug_episodes is not None else "_all"
            history_suffix = "_with_history"
            indexed_filename = f"spatial_features_indexed{history_suffix}{debug_suffix}.h5"
            indexed_path = Path(spatial_features_dir) / indexed_filename
            
            if indexed_path.exists():
                print(f"✅ 找到索引文件: {indexed_path}")
                self._load_indexed_features(indexed_path)
            else:
                print(f"⚠️  未找到索引文件，开始自动重整...")
                self._reorganize_and_save(indexed_path)
                self._load_indexed_features(indexed_path)
    
    def _load_indexed_features(self, indexed_path):
        """加载索引化的 spatial features"""
        print(f"📂 加载索引文件: {indexed_path}")
        self.spatial_h5_file = h5py.File(indexed_path, 'r')
        print(f"✅ 索引文件加载成功，共 {len(self.spatial_h5_file)} 帧")
    
    def _reorganize_and_save(self, output_path):
        """
        🔥 自动重整 spatial features 为 frame-level 索引格式
        将 episode-level 的文件重整为按 dataset[idx] 顺序的 HDF5 格式
        """
        print("=" * 80)
        print("🔄 开始重整 Spatial Features")
        print("=" * 80)
        
        # 1. 获取所有 spatial features 文件
        features_dir = Path(self.spatial_features_dir)
        
        feature_files = list(features_dir.glob("episode_spatial_features_with_history_*.pkl"))
        if not feature_files:
            # 如果没找到，再找标准文件
            feature_files = list(features_dir.glob("episode_spatial_features_*.pkl"))
            if not feature_files:
                raise FileNotFoundError(f"未找到 spatial features 文件在 {features_dir}")
            print(f"使用标准 spatial features（无历史信息）")
        else:
            print(f"使用带历史信息的 spatial features")
    

        # 排序
        feature_files = sorted(feature_files, key=lambda x: int(x.stem.split('_')[-1]))
        
        if self.debug_episodes is not None:
            feature_files = feature_files[:self.debug_episodes]
            print(f"🔧 调试模式：只处理前 {self.debug_episodes} 个episodes")
        
        print(f"📊 总计 {len(feature_files)} 个 episode 文件")

        
        # 2. 加载所有 episode features 到内存（一次性）
        print("📥 加载所有 episode features 到内存...")
        episode_features = {}
        
        for file_path in tqdm(feature_files, desc="加载episodes"):
            episode_id = int(file_path.stem.split('_')[-1])
            try:
                with open(file_path, 'rb') as f:
                    episode_features[episode_id] = pickle.load(f)
            except Exception as e:
                print(f"⚠️  加载 {file_path} 失败: {e}")
                continue
        
        print(f"✅ 成功加载 {len(episode_features)} 个episodes")
        
        # 尝试加载缓存的索引映射
        debug_suffix = f"_{self.debug_episodes}" if self.debug_episodes is not None else "_all"
        frame_map_cache_file = features_dir / f".episode_frame_mapping{debug_suffix}.pkl"

        if frame_map_cache_file.exists():
            print(f"📂 从缓存加载索引映射: {frame_map_cache_file}")
            with open(frame_map_cache_file, 'rb') as f:
                frame_map = pickle.load(f)
            print(f"✅ 索引映射加载完成，共 {len(frame_map)} 条")
        else:
            print("🔍 建立 (episode_id, frame_id) -> dataset_idx 映射...")
            frame_map = {}
            for idx in tqdm(range(len(self.dataset)), desc="建立索引"):
                item = self.dataset[idx]
                episode_id = item["episode_index"].item()
                frame_id = item.get("frame_index", idx).item()
                frame_map[(episode_id, frame_id)] = idx
            
            # 保存缓存
            print(f"💾 保存索引映射到: {frame_map_cache_file}")
            with open(frame_map_cache_file, 'wb') as f:
                pickle.dump(frame_map, f)
            print(f"✅ 索引映射完成，共 {len(frame_map)} 条")

        # 3. 按照 dataset 顺序重整
        print(f"🔄 重整为 frame-level 格式（总共 {len(self.dataset)} 帧）...")
        
        success_count = 0
        missing_count = 0
        
        with h5py.File(output_path, 'w') as h5f:
            for idx in tqdm(range(len(self.dataset)), desc="重整帧"):
                try:
                    item = self.dataset[idx]
                    episode_id = item["episode_index"].item()
                    frame_id = item.get("frame_index", idx).item()
                    
                    # 查找对应的 spatial features
                    if episode_id not in episode_features:
                        missing_count += 1
                        continue
                    
                    ep_features = episode_features[episode_id]
                    frame_features = None
                    
                    # 在 episode 的所有帧中查找当前 frame
                    for frame_feat in ep_features['features']:
                        if frame_feat['frame_index'] == frame_id:
                            frame_features = frame_feat
                            break
                    
                    if frame_features is None:
                        missing_count += 1
                        continue
                    
                    # 创建 HDF5 group
                    grp = h5f.create_group(f'frame_{idx:07d}')
                    
                    # 保存基础 spatial tokens
                    grp.create_dataset('base_camera_tokens', 
                                     data=frame_features['base_camera_tokens'].numpy(),
                                     )
                    grp.create_dataset('wrist_camera_tokens', 
                                     data=frame_features['wrist_camera_tokens'].numpy(),
                                     )
                    grp.create_dataset('base_patch_tokens', 
                                     data=frame_features['base_patch_tokens'].numpy(),
                                     )
                    grp.create_dataset('wrist_patch_tokens', 
                                     data=frame_features['wrist_patch_tokens'].numpy(),
                                     )
                    
                    # 保存元数据
                    grp.attrs['episode_id'] = episode_id
                    grp.attrs['frame_id'] = frame_id
                    grp.attrs['dataset_idx'] = idx
                    
                    # 🔥 如果有历史信息且 use_history=True
                    if 'history_info' in frame_features:
                        history_info = frame_features['history_info']
                        history_indices = history_info['history_indices']
                        
                        # 保存历史帧索引
                        grp.create_dataset('history_indices', data=history_indices)
                        
                        # 保存每个历史帧的 dataset_idx（用于快速查找）
                        #history_dataset_indices = []
                        #for hist_frame_id in history_indices:
                            # 查找这个历史帧在 dataset 中的索引
                            #hist_dataset_idx = -1
                            #for check_idx in range(len(self.dataset)):
                                #check_item = self.dataset[check_idx]
                                #if (check_item["episode_index"].item() == episode_id and 
                                    #check_item.get("frame_index", check_idx).item() == hist_frame_id):
                                    #hist_dataset_idx = check_idx
                                    #break
                            #history_dataset_indices.append(hist_dataset_idx)
                        history_dataset_indices = [
                            frame_map.get((episode_id, hist_fid), -1) 
                            for hist_fid in history_indices
                        ]
                        grp.create_dataset('history_dataset_indices', data=history_dataset_indices)
                        grp.attrs['has_history'] = True
                    else:
                        grp.attrs['has_history'] = False
                    
                    success_count += 1
                    
                except Exception as e:
                    print(f"⚠️  处理 dataset[{idx}] 时出错: {e}")
                    missing_count += 1
                    continue
        
        # 4. 统计信息
        print("=" * 80)
        print("✅ 重整完成！")
        print(f"   📊 成功重整: {success_count} 帧")
        print(f"   ⚠️  缺失数据: {missing_count} 帧")
        print(f"   💾 保存到: {output_path}")
        file_size = output_path.stat().st_size / 1e9
        print(f"   📦 文件大小: {file_size:.2f} GB")
        print("=" * 80)
    
    def __getitem__(self, idx):
        """🔥 从索引文件快速读取 spatial features"""
        # 基础数据
        item = self.dataset[idx]
        normalized_item = self.normalizer.normalize(item)
        
        # 图像处理
        images = {}
        
        if "image" in normalized_item:
            base_image = normalized_item["image"]
            while base_image.dim() > 3 and 1 in base_image.shape:
                base_image = base_image.squeeze()
            base_image = (base_image * 255).to(torch.uint8)
            images["base_0_rgb"] = base_image
        
        if "wrist_image" in normalized_item:
            wrist_image = normalized_item["wrist_image"]
            while wrist_image.dim() > 3 and 1 in wrist_image.shape:
                wrist_image = wrist_image.squeeze()
            wrist_image = (wrist_image * 255).to(torch.uint8)
            images["left_wrist_0_rgb"] = wrist_image
        
        # 🔥 从 HDF5 快速读取 spatial features
        precomputed_spatial_features = None
        if self.spatial_h5_file is not None:
            frame_key = f'frame_{idx:07d}'
            
            if frame_key in self.spatial_h5_file:
                grp = self.spatial_h5_file[frame_key]
                
                # 读取基础 tokens
                precomputed_spatial_features = {
                    "base_camera_tokens": torch.from_numpy(grp['base_camera_tokens'][:]),
                    "wrist_camera_tokens": torch.from_numpy(grp['wrist_camera_tokens'][:]),
                    "base_patch_tokens": torch.from_numpy(grp['base_patch_tokens'][:]),
                    "wrist_patch_tokens": torch.from_numpy(grp['wrist_patch_tokens'][:]),
                }
                
                # 🔥 如果需要历史信息
                if self.use_history and grp.attrs.get('has_history', False):
                    history_dataset_indices = grp['history_dataset_indices'][:]
                    history_indices = grp['history_indices'][:]
                    
                    # 加载历史帧
                    history_frames = []
                    for hist_dataset_idx, hist_frame_id in zip(history_dataset_indices, history_indices):
                        if hist_dataset_idx < 0:
                            continue
                        
                        hist_frame_key = f'frame_{hist_dataset_idx:07d}'
                        if hist_frame_key not in self.spatial_h5_file:
                            continue
                        
                        hist_grp = self.spatial_h5_file[hist_frame_key]
                        
                        # 获取历史帧的图像
                        hist_item = self.dataset[hist_dataset_idx]
                        hist_normalized = self.normalizer.normalize(hist_item)
                        
                        # base image
                        hist_base_image = hist_normalized["image"]
                        while hist_base_image.dim() > 3 and 1 in hist_base_image.shape:
                            hist_base_image = hist_base_image.squeeze()
                        hist_base_image_uint8 = (hist_base_image * 255).to(torch.uint8)
                        
                        # wrist image
                        hist_wrist_image_uint8 = None
                        if "wrist_image" in hist_normalized:
                            hist_wrist_image = hist_normalized["wrist_image"]
                            while hist_wrist_image.dim() > 3 and 1 in hist_wrist_image.shape:
                                hist_wrist_image = hist_wrist_image.squeeze()
                            hist_wrist_image_uint8 = (hist_wrist_image * 255).to(torch.uint8)
                        
                        # 历史帧的 spatial tokens
                        history_frames.append({
                            'frame_index': int(hist_frame_id),
                            'base_image_uint8': hist_base_image_uint8,
                            'wrist_image_uint8': hist_wrist_image_uint8,
                            'base_camera_tokens': torch.from_numpy(hist_grp['base_camera_tokens'][:]),
                            'base_patch_tokens': torch.from_numpy(hist_grp['base_patch_tokens'][:]),
                            'wrist_camera_tokens': torch.from_numpy(hist_grp['wrist_camera_tokens'][:]),
                            'wrist_patch_tokens': torch.from_numpy(hist_grp['wrist_patch_tokens'][:]),
                        })
                    
                    # 添加历史信息
                    precomputed_spatial_features['history_info'] = {
                        'history_indices': history_indices.tolist(),
                        'current_frame_index': item.get("frame_index", idx).item(),
                        'history_frames': history_frames
                    }
        
        # 任务指令
        task_text = item.get("task", "complete the task")
        if isinstance(task_text, str):
            prompt = [task_text]
        elif isinstance(task_text, (list, tuple)):
            prompt = [str(t) for t in task_text]
        else:
            prompt = [str(task_text)]
        
        # 构建返回结果
        result = {
            "image": images,
            "state": normalized_item["state"][0],
            "action": normalized_item["actions"],
            "action_is_pad": normalized_item.get("action_is_pad", 
                torch.zeros_like(normalized_item["actions"][..., 0], dtype=torch.bool)
            ),
            "prompt": prompt,
            "episode_index": item["episode_index"],
        }
        
        # 🔥 添加预计算特征（如果有）
        if precomputed_spatial_features is not None:
            result["precomputed_spatial_features"] = precomputed_spatial_features
        
        return result
    
    def __del__(self):
        """关闭 HDF5 文件"""
        if hasattr(self, 'spatial_h5_file') and self.spatial_h5_file is not None:
            self.spatial_h5_file.close()


def enhanced_collate_fn(batch):
    """增强版批处理函数 - 支持历史特征"""
    result = {}
    
    # 处理图像
    if "image" in batch[0]:
        images = {}
        for key in batch[0]["image"].keys():
            images[key] = torch.stack([item["image"][key] for item in batch])
        result["image"] = images
    
    # 🔥 处理预计算特征（包含历史信息）
    if "precomputed_spatial_features" in batch[0]:
        precomputed_batch = {}
        
        # 处理标准spatial tokens
        for key in ["base_camera_tokens", "wrist_camera_tokens", "base_patch_tokens", "wrist_patch_tokens"]:
            features = []
            for item in batch:
                if "precomputed_spatial_features" in item:
                    feat = item["precomputed_spatial_features"].get(key)
                    if feat is not None:
                        features.append(feat)
            
            if features:
                precomputed_batch[key] = torch.stack(features)
        
        # 🔥 处理历史信息（不stack，保持为list）
        history_infos = []
        for item in batch:
            if ("precomputed_spatial_features" in item and 
                "history_info" in item["precomputed_spatial_features"]):
                history_infos.append(item["precomputed_spatial_features"]["history_info"])
            else:
                history_infos.append(None)
        
        precomputed_batch["history_info"] = history_infos
        
        if precomputed_batch:
            result["precomputed_spatial_features"] = precomputed_batch
    
    # 其他字段
    result["state"] = torch.stack([item["state"] for item in batch])
    result["action"] = torch.stack([item["action"] for item in batch])
    result["action_is_pad"] = torch.stack([item["action_is_pad"] for item in batch])
    result["episode_index"] = torch.stack([item["episode_index"] for item in batch])
    
    # 处理prompt
    prompts = []
    for item in batch:
        prompt = item["prompt"]
        if isinstance(prompt, list):
            prompts.append(prompt[0] if len(prompt) > 0 else "complete the task")
        else:
            prompts.append(str(prompt))
    result["prompt"] = prompts
    
    return result
