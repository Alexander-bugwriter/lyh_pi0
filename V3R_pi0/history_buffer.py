import torch

class UnlimitedHistoryBuffer:
    """无限长度历史特征缓存管理 - 支持均匀采样"""

    def __init__(self, config):
        self.num_sampled_frames = config.num_sampled_history_frames
        self.sampling_method = config.history_sampling_method
        self.buffer = []  # 无长度限制的历史帧存储
        self.frame_count = 0

    def append(self, current_features, separator_token):
        """存储时就拼接分隔符"""
        batch_size = current_features.shape[0]
        expanded_sep = separator_token.expand(batch_size, -1, -1)
        stored_item = torch.cat([current_features, expanded_sep], dim=1)
        self.buffer.append(stored_item.detach().clone().cpu()) #显式移动到CPU内存
        self.frame_count += 1

    def sample_uniform_frames(self, num_frames=5):
        """从完整历史中均匀采样指定数量的帧 - 返回特征列表而非拼接结果"""
        if len(self.buffer) < num_frames:
            if len(self.buffer) == 0:
                return []
            return self.buffer.copy()  # 返回列表，不拼接

        if self.sampling_method == "uniform":
            total_frames = len(self.buffer)
            indices = torch.linspace(0, total_frames - 1, num_frames).long()
            selected_features = [self.buffer[i] for i in indices]
        elif self.sampling_method == "recent":
            selected_features = self.buffer[-num_frames:]
        else:
            total_frames = len(self.buffer)
            indices = torch.linspace(0, total_frames - 1, num_frames).long()
            selected_features = [self.buffer[i] for i in indices]

        return selected_features  # 🔥 返回特征列表

    def get_enhanced_features_with_separators(self, current_features):
        if len(self.buffer) == 0:
            return current_features

        history_feature_list = self.sample_uniform_frames(self.num_sampled_frames)
        if not history_feature_list:
            return current_features
        # 将历史特征移回GPU
        device = current_features.device
        dtype = current_features.dtype
        history_feature_list = [f.to(device=device, dtype=dtype) for f in history_feature_list]
    
        # 直接拼接：[hist1+sep] + [hist2+sep] + ... + [current]
        all_features = history_feature_list + [current_features]
        return torch.cat(all_features, dim=1)



    def get_buffer_info(self):
        """获取缓存状态信息 (用于调试)"""
        return {
            "total_frames": len(self.buffer),
            "buffer_size_mb": sum(f.numel() * f.element_size() for f in self.buffer) / (1024 ** 2),
            "last_frame_shape": self.buffer[-1].shape if self.buffer else None
        }

    def clear(self):
        """清空历史缓存 (episode 边界调用)"""
        self.buffer.clear()
        self.frame_count = 0

    def __len__(self):
        """返回历史帧数量"""
        return len(self.buffer)