from collections import defaultdict
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

from .normalizers import Normalizer
from .dataset_config import get_dataset_info, generate_delta_timestamps







class LerobotPI0Dataset(Dataset):
    """标准Lerobot格式数据集包装器"""
    
    def __init__(self, repo_id=None, root=None, image_size=224, action_horizon=50,dataset_fps=10.0,debug_episodes=None):
        print(f"加载Lerobot数据集: {repo_id}")
        episodes=None
        if debug_episodes:
            episodes = list(range(debug_episodes))
            print(f"调试模式：只加载前 {debug_episodes} 个episodes")
        image_transforms = Resize((image_size, image_size))
        info = get_dataset_info(root)
        delta_timestamps = generate_delta_timestamps(info['fps'], info['features'], action_horizon) 
        # 标准lerobot格式的时间戳配置
        #delta_timestamps = {
        #    "observation.images.base": [0],
        #    "observation.images.wrist": [0], 
        #    "observation.state": [0],
        #    "action": [i / dataset.fps for i in range(action_horizon)],
        #    }
        
        try:
            self.dataset = LeRobotDataset(
                repo_id=repo_id,
                root=root,
                image_transforms=image_transforms,
                delta_timestamps=delta_timestamps,
                episodes=episodes
            )
            print(f"数据集加载成功，共 {len(self.dataset)} 条数据")
            
        except Exception as e:
            print(f"数据集加载失败: {e}")
            
            # 调试信息：检查本地路径结构
            if root and os.path.exists(root):
                print(f"检查本地路径结构:")
                try:
                    # 检查关键文件
                    key_paths = [
                        os.path.join(root, "meta", "info.json"),
                        os.path.join(root, "meta", "stats.json"),
                        os.path.join(root, "data")
                    ]
                    for key_path in key_paths:
                        if os.path.exists(key_path):
                            print(f" {os.path.relpath(key_path, root)}")
                        else:
                            print(f" {os.path.relpath(key_path, root)}")
                            
                    # 检查data目录内容
                    data_dir = os.path.join(root, "data")
                    if os.path.exists(data_dir):
                        chunks = [d for d in os.listdir(data_dir) if d.startswith("chunk-")]
                        print(f"发现 {len(chunks)} 个chunk目录")
                        
                except Exception as debug_e:
                    print(f"调试失败: {debug_e}")
        
        # 标准化器配置
        self.normalizer = Normalizer(
            norm_stats=self.dataset.meta.stats,
            norm_type={
                "image": "identity",
                "wrist_image": "identity", 
                "state": "meanstd",
                "actions": "meanstd",
            }
        )


    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        #print(f"Original dataset keys: {list(item.keys())}")
        #if idx % 10 == 0:  # 每1000个样本打印一次
            #print(f"数据键: {list(item.keys())}")
            #for key in item.keys():
                #if 'episode' in key.lower():
                    #print(f"Episode相关字段: {key} = {item[key]}")
        normalized_item = self.normalizer.normalize(item)
        
        # 图像处理
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
            #prompt = task_text
            prompt = [str(t) for t in task_text]  # 确保每个元素都是字符串
        else:
            prompt = [str(task_text)]
        #print(f"prompt type and content: {type(prompt)}, {prompt}") 
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
    """增强版数据集 - 支持spatial features和历史信息"""
    
    def __init__(self, repo_id=None, root=None, image_size=224, action_horizon=50,
                 dataset_fps=10.0, debug_episodes=None, spatial_features_dir=None):
        
        super().__init__(repo_id, root, image_size, action_horizon, dataset_fps, debug_episodes)
        
        self.spatial_features_dir = spatial_features_dir
        self.spatial_features_cache = {}
        
        if spatial_features_dir:
            self._load_spatial_features_index()
    
    def _load_spatial_features_index(self):
        """加载spatial features索引（支持历史信息）"""
        import pickle
        from pathlib import Path
        
        features_dir = Path(self.spatial_features_dir)
        
        # 🔥 先尝试新的历史特征文件
        feature_files = list(features_dir.glob("episode_spatial_features_with_history_*.pkl"))
        
        # 如果没有，使用标准特征文件
        if not feature_files:
            feature_files = list(features_dir.glob("episode_spatial_features_*.pkl"))
            print("使用标准spatial features（无历史信息）")
        else:
            print("使用带历史信息的spatial features")
        
        print(f"加载 {len(feature_files)} 个特征文件...")
        
        for feature_file in feature_files:
            try:
                with open(feature_file, 'rb') as f:
                    episode_features = pickle.load(f)
                    episode_id = episode_features['episode_id']
                    self.spatial_features_cache[episode_id] = episode_features
            except Exception as e:
                print(f"加载 {feature_file} 失败: {e}")
        
        print(f"特征索引加载完成，覆盖 {len(self.spatial_features_cache)} 个episodes")
    
    def __getitem__(self, idx):
        """🔥 这是唯一需要添加的方法"""
        item = self.dataset[idx]
        normalized_item = self.normalizer.normalize(item)
        
        # 图像处理（与父类相同）
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
        
        # 🔥 加载对应的预计算特征（包含历史信息）
        precomputed_spatial_features = None
        if self.spatial_features_dir:
            episode_id = item["episode_index"].item()
            frame_id = item.get("frame_index", idx).item()
            
            if episode_id in self.spatial_features_cache:
                episode_features = self.spatial_features_cache[episode_id]
                for frame_feat in episode_features['features']:
                    if frame_feat['frame_index'] == frame_id:
                        precomputed_spatial_features = {
                            "base_camera_tokens": frame_feat['base_camera_tokens'],
                            "wrist_camera_tokens": frame_feat['wrist_camera_tokens'],
                            "base_patch_tokens": frame_feat['base_patch_tokens'],
                            "wrist_patch_tokens": frame_feat['wrist_patch_tokens'],
                            "history_info": frame_feat.get('history_info', None)  # 🔥 支持历史信息
                        }
                        break
        
        # 任务指令（与父类相同）
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
                history_infos.append(None)  # 占位符
        
        precomputed_batch["history_info"] = history_infos
        
        if precomputed_batch:
            result["precomputed_spatial_features"] = precomputed_batch
    
    # 其他字段（保持不变）
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