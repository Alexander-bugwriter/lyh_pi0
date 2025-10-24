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
import pickle
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

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
        # 添加缓存逻辑
        
        cache_file = f".dataset_cache_{debug_episodes if debug_episodes is not None else 'all'}.pkl"
        # cache_path = os.path.abspath(cache_file)  # 获取绝对路径
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
            print(f"数据集加载成功，共 {len(self.dataset)} 条数据")
        else:
            print("首次加载dataset，构建缓存中......")
            image_transforms = Resize((image_size, image_size))
            info = get_dataset_info(root)
            delta_timestamps = generate_delta_timestamps(info['fps'], info['features'], action_horizon)

            try:
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

            except Exception as e:
                print(f"数据集加载失败: {e}")
        




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
            "frame_index": item["frame_index"],
            "timestamp":item["timestamp"],
        }
    
class Enhanced_LerobotPI0Dataset(LerobotPI0Dataset):
    """增强版数据集 - 支持spatial features和历史信息"""
    
    def __init__(self, repo_id=None, root=None, image_size=224, action_horizon=50,
                 dataset_fps=10.0, debug_episodes=None, spatial_features_dir=None):
        
        super().__init__(repo_id, root, image_size, action_horizon, dataset_fps, debug_episodes)
        print(f"DEBUG: spatial_features_dir = {spatial_features_dir}")
        print(f"DEBUG: spatial_features_dir type = {type(spatial_features_dir)}")
        print(f"DEBUG: spatial_features_dir exists = {Path(spatial_features_dir).exists() if spatial_features_dir else False}")
    
        self.spatial_features_dir = spatial_features_dir
        self.spatial_features_cache = {}
        self.episode_frame_mapping = {}  # 🔥 新增这一行
        self.debug_episodes = debug_episodes


        if spatial_features_dir:
            self._load_spatial_features_index()
            self._build_episode_frame_mapping()  # 🔥 新增这一行调用

    @staticmethod
    def load_single_file(file_path):
        try:
            with open(file_path, 'rb') as f:
                episode_features = pickle.load(f)
                return file_path, episode_features['episode_id'], episode_features
        except Exception as e:
            print(f"加载 {file_path} 失败: {e}")
            return file_path, None, None

    def _load_spatial_features_index(self):
        """建立文件路径索引，确保排序"""
        debug_suffix = f"_{self.debug_episodes}" if self.debug_episodes is not None else "_all"
        # index_cache_file = f".spatial_index_cache{debug_suffix}.pkl"
        index_cache_file = os.path.join(self.spatial_features_dir, f".spatial_index_cache{debug_suffix}.pkl")  # 数据集目录下

        if os.path.exists(index_cache_file):
            print(f"从缓存加载索引: {index_cache_file}")
            with open(index_cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                spatial_files_index  = cache_data['spatial_path_index']
            print(f"索引缓存加载成功，共 {len(spatial_files_index)} 个episodes")
            self.spatial_features_cache = spatial_files_index
            # 🔥 还是打印验证信息
            #self._print_cache_validation()
        else:
            features_dir = Path(self.spatial_features_dir)
            feature_files = list(features_dir.glob("episode_spatial_features_with_history_*.pkl"))
            if not feature_files:
                feature_files = list(features_dir.glob("episode_spatial_features_*.pkl"))
                print("使用标准spatial features（无历史信息）")
            else:
                print("使用带历史信息的spatial features")

            # 🔥 关键1：确保所有情况下都排序
            feature_files = sorted(feature_files, key=lambda x: int(x.stem.split('_')[-1]))
    
            if self.debug_episodes is not None:
                feature_files = feature_files[:self.debug_episodes]
                print(f"调试模式：只处理前 {self.debug_episodes} 个episodes的特征文件")

            print(f"建立 {len(feature_files)} 个文件的路径索引...")
            # 🔥 构建路径索引（不读取数据）
            spatial_files_index = {}
            for file_path in feature_files:
                episode_id = int(file_path.stem.split('_')[-1])
                spatial_files_index[episode_id] = file_path
        
            # 保存路径索引缓存
            with open(index_cache_file, 'wb') as f:
                pickle.dump({'spatial_path_index': spatial_files_index}, f)
            print(f"路径索引已缓存到: {index_cache_file}")
            self.spatial_features_cache = spatial_files_index
            #self._print_cache_validation()
        # 🔥 第二步：根据路径索引，单线程读取所有数据到内存
        # print(f"单线程加载 {len(spatial_files_index)} 个特征文件到内存...")
        # self.spatial_features_cache = {}  # 这里存完整数据

        # for i, (episode_id, file_path) in enumerate(sorted(spatial_files_index.items())):
        #     if i % 100 == 0:
        #         print(f"  加载进度: {i}/{len(spatial_files_index)}")

        #     with open(file_path, 'rb') as f:
        #         episode_features = pickle.load(f)
        #         self.spatial_features_cache[episode_id] = episode_features

        #print(f"所有特征已加载到内存，共 {len(self.spatial_features_cache)} 个episodes")
    def _print_cache_validation(self):
        """打印缓存验证信息"""
        print("📋 缓存内容验证:")
        episode_ids = sorted(self.spatial_features_cache.keys())
        print(f"Episode ID范围: {episode_ids[0]} 到 {episode_ids[-1]}")
        print(f"前10个episode的文件映射:")
        for i, episode_id in enumerate(episode_ids[:10]):
            file_path = self.spatial_features_cache[episode_id]
            filename_id = int(file_path.stem.split('_')[-1])
            status = "✓" if filename_id == episode_id else "✗"
            print(f"  Episode {episode_id} -> {file_path.name} {status}")
    
        # 检查连续性
        expected_ids = list(range(len(episode_ids)))
        if episode_ids == expected_ids:
            print("✅ Episode ID连续性检查: 通过")
        else:
            print(f"⚠️  Episode ID不连续: 期望{expected_ids[:5]}...，实际{episode_ids[:5]}...") 
    # def _load_spatial_features_index(self):
    #     """加载spatial features索引（支持历史信息）"""
    #     import pickle
    #     from pathlib import Path
    #
    #     features_dir = Path(self.spatial_features_dir)
    #
    #     # 🔥 先尝试新的历史特征文件
    #     feature_files = list(features_dir.glob("episode_spatial_features_with_history_*.pkl"))
    #
    #     # 如果没有，使用标准特征文件
    #     if not feature_files:
    #         feature_files = list(features_dir.glob("episode_spatial_features_*.pkl"))
    #         print("使用标准spatial features（无历史信息）")
    #     else:
    #         print("使用带历史信息的spatial features")
    #
    #     if self.debug_episodes is not None:
    #         # 根据文件名排序，确保加载前N个episodes
    #         feature_files = sorted(feature_files, key=lambda x: int(x.stem.split('_')[-1]))
    #         feature_files = feature_files[:self.debug_episodes]
    #         print(f"🐛 调试模式：只加载前 {self.debug_episodes} 个episodes的特征文件")
    #
    #     print(f"加载 {len(feature_files)} 个特征文件...")
    #
    #     for feature_file in feature_files:
    #         try:
    #             with open(feature_file, 'rb') as f:
    #                 episode_features = pickle.load(f)
    #                 episode_id = episode_features['episode_id']
    #                 self.spatial_features_cache[episode_id] = episode_features
    #         except Exception as e:
    #             print(f"加载 {feature_file} 失败: {e}")
    #
    #     print(f"特征索引加载完成，覆盖 {len(self.spatial_features_cache)} 个episodes")
    def _build_episode_frame_mapping(self):
        """🔥 新增: 构建 (episode_id, frame_index) -> dataset_idx 的映射，支持缓存"""
        
        # 🔥 缓存文件路径
        debug_suffix = f"_{self.debug_episodes}" if self.debug_episodes is not None else "_all"
        mapping_cache_file = os.path.join(self.spatial_features_dir, f".episode_frame_mapping{debug_suffix}.pkl")
        
        # 🔥 尝试加载缓存
        if os.path.exists(mapping_cache_file):
            print(f"从缓存加载映射表: {mapping_cache_file}")
            with open(mapping_cache_file, 'rb') as f:
                self.episode_frame_mapping = pickle.load(f)
            print(f"✅ 映射表加载成功,共 {len(self.episode_frame_mapping)} 条记录")
            return
        
        # 🔥 缓存不存在，构建映射表
        print("构建episode-frame映射表...")
        
        for dataset_idx in range(len(self.dataset)):
            item = self.dataset[dataset_idx]
            episode_id = item["episode_index"].item()
            frame_id = item.get("frame_index", dataset_idx).item()
            
            self.episode_frame_mapping[(episode_id, frame_id)] = dataset_idx
        
        print(f"✅ 映射表构建完成,共 {len(self.episode_frame_mapping)} 条记录")
        
        # 🔥 保存缓存
        with open(mapping_cache_file, 'wb') as f:
            pickle.dump(self.episode_frame_mapping, f)
        print(f"✅ 映射表已缓存到: {mapping_cache_file}")

    def _load_history_frames_by_indices(self, episode_id, history_indices):
        """
        🔥 新增: 根据历史索引动态加载历史帧
        
        Returns:
            history_frames: List[dict] 包含 {frame_index, base_image_uint8, wrist_image_uint8, spatial_tokens}
        """
        history_frames = []
        
        if episode_id not in self.spatial_features_cache:
            return []
        
        # 加载该episode的spatial特征文件
        file_path = self.spatial_features_cache[episode_id]  # 🔥 使用正确的属性名
        with open(file_path, 'rb') as f:
            episode_features = pickle.load(f)
        
        for hist_frame_idx in history_indices:
            # 1. 从dataset获取原始图像
            dataset_idx = self.episode_frame_mapping.get((episode_id, hist_frame_idx))
            if dataset_idx is None:
                continue
            
            hist_item = self.dataset[dataset_idx]
            hist_normalized = self.normalizer.normalize(hist_item)
            
            # 2. 处理base图像
            base_image = hist_normalized["image"]
            while base_image.dim() > 3 and 1 in base_image.shape:
                base_image = base_image.squeeze()
            base_image_uint8 = (base_image * 255).to(torch.uint8)
            
            # 3. 处理wrist图像
            wrist_image_uint8 = None
            if "wrist_image" in hist_normalized:
                wrist_image = hist_normalized["wrist_image"]
                while wrist_image.dim() > 3 and 1 in wrist_image.shape:
                    wrist_image = wrist_image.squeeze()
                wrist_image_uint8 = (wrist_image * 255).to(torch.uint8)
            
            # 4. 找到对应的spatial tokens
            frame_spatial_tokens = None
            for frame_feat in episode_features['features']:
                if frame_feat['frame_index'] == hist_frame_idx:
                    frame_spatial_tokens = {
                        'base_camera_tokens': frame_feat['base_camera_tokens'],
                        'base_patch_tokens': frame_feat['base_patch_tokens'],
                        'wrist_camera_tokens': frame_feat['wrist_camera_tokens'],
                        'wrist_patch_tokens': frame_feat['wrist_patch_tokens'],
                    }
                    break
            
            if frame_spatial_tokens is None:
                continue
            
            # 5. 组装
            history_frames.append({
                'frame_index': hist_frame_idx,
                'base_image_uint8': base_image_uint8,
                'wrist_image_uint8': wrist_image_uint8,
                **frame_spatial_tokens
            })
        
        return history_frames

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
            # if episode_id in self.spatial_features_cache:
            #     file_path = self.spatial_features_cache[episode_id]  # 现在这里是路径
            #     # 🔥 按需加载
            #     with open(file_path, 'rb') as f:
            #         episode_features = pickle.load(f)
            #     for frame_feat in episode_features['features']:
            #         if frame_feat['frame_index'] == frame_id:
            #             precomputed_spatial_features = {
            #                 "base_camera_tokens": frame_feat['base_camera_tokens'],
            #                 "wrist_camera_tokens": frame_feat['wrist_camera_tokens'],
            #                 "base_patch_tokens": frame_feat['base_patch_tokens'],
            #                 "wrist_patch_tokens": frame_feat['wrist_patch_tokens'],
            #                 "history_info": frame_feat.get('history_info', None)  # 🔥 支持历史信息
            #             }
            #             break
            # 🔥 新逻辑：动态加载历史if episode_id in self.spatial_path_index:
            if episode_id in self.spatial_features_cache:
                file_path = self.spatial_features_cache[episode_id]
                with open(file_path, 'rb') as f:
                    episode_features = pickle.load(f)
                
                for frame_feat in episode_features['features']:
                    if frame_feat['frame_index'] == frame_id:
                        precomputed_spatial_features = {
                            "base_camera_tokens": frame_feat['base_camera_tokens'],
                            "wrist_camera_tokens": frame_feat['wrist_camera_tokens'],
                            "base_patch_tokens": frame_feat['base_patch_tokens'],
                            "wrist_patch_tokens": frame_feat['wrist_patch_tokens'],
                        }
                        
                        # 🔥 如果有历史索引,动态加载历史帧
                        if 'history_info' in frame_feat:
                            history_info = frame_feat['history_info']
                            history_indices = history_info['history_indices']
                            
                            # 调用新函数动态加载完整历史数据
                            history_frames = self._load_history_frames_by_indices(
                                episode_id, history_indices
                            )
                            
                            precomputed_spatial_features['history_info'] = {
                                'num_history_frames': history_info['num_history_frames'],
                                'history_indices': history_indices,
                                'current_frame_index': frame_id,
                                'history_frames': history_frames  # 完整的历史帧数据
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
