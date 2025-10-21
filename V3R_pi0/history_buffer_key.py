import torch

# class UnlimitedHistoryBuffer:
#     """无限长度历史特征缓存管理 - 支持均匀采样"""

#     def __init__(self, config):
#         self.num_sampled_frames = config.num_sampled_history_frames
#         self.sampling_method = config.history_sampling_method
#         self.buffer = []  # 无长度限制的历史帧存储
#         self.frame_count = 0

#     def append(self, current_features, separator_token):
#         """存储时就拼接分隔符"""
#         batch_size = current_features.shape[0]
#         expanded_sep = separator_token.expand(batch_size, -1, -1)
#         stored_item = torch.cat([current_features, expanded_sep], dim=1)
#         self.buffer.append(stored_item.detach().clone().cpu()) #显式移动到CPU内存
#         self.frame_count += 1

#     def sample_uniform_frames(self, num_frames=5):
#         """从完整历史中均匀采样指定数量的帧 - 返回特征列表而非拼接结果"""
#         if len(self.buffer) < num_frames:
#             if len(self.buffer) == 0:
#                 return []
#             return self.buffer.copy()  # 返回列表，不拼接

#         if self.sampling_method == "uniform":
#             total_frames = len(self.buffer)
#             indices = torch.linspace(0, total_frames - 1, num_frames).long()
#             selected_features = [self.buffer[i] for i in indices]
#         elif self.sampling_method == "recent":
#             selected_features = self.buffer[-num_frames:]
#         else:
#             total_frames = len(self.buffer)
#             indices = torch.linspace(0, total_frames - 1, num_frames).long()
#             selected_features = [self.buffer[i] for i in indices]

#         return selected_features  # 🔥 返回特征列表

#     def get_enhanced_features_with_separators(self, current_features,separator_token,max_frame_num):
#         # if len(self.buffer) == 0:
#         #     return current_features

#         history_feature_list = self.sample_uniform_frames(self.num_sampled_frames)
#         # if not history_feature_list:
#         #     return current_features
#         # 将历史特征移回GPU
#         device = current_features.device
#         dtype = current_features.dtype
#         history_feature_list = [f.to(device=device, dtype=dtype) for f in history_feature_list]
        
#         while len(history_feature_list) < max_frame_num:
#             # 用当前帧+分隔符来填充
#             expanded_sep = separator_token.expand(current_features.shape[0], -1, -1).to(device=device, dtype=dtype)
#             current_with_sep = torch.cat([current_features, expanded_sep], dim=1)
#             history_feature_list.append(current_with_sep)
        
#         # 直接拼接：[hist1+sep] + [hist2+sep] + ... + [current]
#         all_features = history_feature_list + [current_features]
#         return torch.cat(all_features, dim=1)



#     def get_buffer_info(self):
#         """获取缓存状态信息 (用于调试)"""
#         return {
#             "total_frames": len(self.buffer),
#             "buffer_size_mb": sum(f.numel() * f.element_size() for f in self.buffer) / (1024 ** 2),
#             "last_frame_shape": self.buffer[-1].shape if self.buffer else None
#         }

#     def clear(self):
#         """清空历史缓存 (episode 边界调用)"""
#         self.buffer.clear()
#         self.frame_count = 0

#     def __len__(self):
#         """返回历史帧数量"""
#         return len(self.buffer)
class UnlimitedHistoryBuffer:
    """无限长度历史特征缓存管理 - 🔥 支持多相机分别存储"""

    def __init__(self, config):
        self.num_sampled_frames = config.num_sampled_history_frames
        # self.sampling_method = config.history_sampling_method
        self.sampling_method = getattr(config, 'history_sampling_method', 'adaptive')
        # 🔥 修改：为每个相机维护独立的buffer
        self.buffers = {
            "base_0_rgb": [],
            "left_wrist_0_rgb": [],
            "right_wrist_0_rgb": []
        }
        self.frame_count = 0

    def append(self, current_features, separator_token, camera_key="base_0_rgb"):
        """
        🔥 修改：存储时指定相机
        
        Args:
            current_features: 当前帧特征
            separator_token: 分隔符
            camera_key: 相机名称（默认base）
        """
        batch_size = current_features.shape[0]
        expanded_sep = separator_token.expand(batch_size, -1, -1)
        stored_item = torch.cat([current_features, expanded_sep], dim=1)
        
        # 存到对应相机的buffer
        self.buffers[camera_key].append(stored_item.detach().clone().cpu())
        self.frame_count += 1

    # def sample_uniform_frames(self, num_frames=5, camera_key="base_0_rgb"):
    #     """
    #     🔥 修改：从指定相机的历史中采样
        
    #     Args:
    #         num_frames: 采样数量
    #         camera_key: 相机名称
    #     """
    #     buffer = self.buffers.get(camera_key, [])
        
    #     if len(buffer) < num_frames:
    #         if len(buffer) == 0:
    #             return []
    #         return buffer.copy()

    #     if self.sampling_method == "uniform":
    #         total_frames = len(buffer)
    #         indices = torch.linspace(0, total_frames - 1, num_frames).long()
    #         selected_features = [buffer[i] for i in indices]
    #     elif self.sampling_method == "recent":
    #         selected_features = buffer[-num_frames:]
    #     else:
    #         total_frames = len(buffer)
    #         indices = torch.linspace(0, total_frames - 1, num_frames).long()
    #         selected_features = [buffer[i] for i in indices]

    #     return selected_features
    def sample_history_frames(self, num_frames=3, camera_key="base_0_rgb"):
        """
        🔥 修改：采用和 extract_feature 相同的采样策略
        
        采样策略:
        - 如果 t >= 30: [t-30, t-20, t-10]
        - 如果 3 <= t < 30: [0, t//3, 2*t//3]
        - 如果 t < 3: [0, 1, t] 或根据t的大小填充
        
        Args:
            num_frames: 历史帧数量（默认3）
            camera_key: 相机名称
        
        Returns:
            selected_features: List[Tensor] 采样的历史帧特征
        """
        buffer = self.buffers.get(camera_key, [])
        current_t = len(buffer)  # 当前帧索引（buffer长度）
        
        if current_t == 0:
            return []
        
        # 🔥 使用和 extract_feature 相同的采样逻辑
        if current_t >= 30:
            # 策略1: 固定间隔采样
            indices = [current_t - 30, current_t - 20, current_t - 10]
        elif current_t >= num_frames:
            # 策略2: 均匀分布采样
            indices = [0, current_t // 3, (2 * current_t) // 3]
        else:
            # 策略3: 填充策略
            if current_t == 1:
                indices = [0, 0, 0]
            elif current_t == 2:
                indices = [0, 1, 1]
            else:  # current_t == 3
                indices = [0, 1, 2]
        
        # 确保索引有效
        indices = [min(max(0, idx), current_t - 1) for idx in indices]
        
        selected_features = [buffer[i] for i in indices]
        return selected_features
    
    def get_enhanced_features_with_separators(self, current_features, separator_token, 
                                         max_frame_num, camera_key="base_0_rgb"):
        """
        获取拼接历史后的特征
        
        Args:
            current_features: 当前帧特征
            separator_token: 分隔符
            max_frame_num: 期望的历史帧数量
            camera_key: 相机名称
        """
        # 🔥 使用新的采样方法
        history_feature_list = self.sample_history_frames(max_frame_num, camera_key)
        
        # 将历史特征移回GPU
        device = current_features.device
        dtype = current_features.dtype
        history_feature_list = [f.to(device=device, dtype=dtype) for f in history_feature_list]
        
        # 🔥 填充到max_frame_num（如果历史不足）
        while len(history_feature_list) < max_frame_num:
            expanded_sep = separator_token.expand(current_features.shape[0], -1, -1).to(device=device, dtype=dtype)
            current_with_sep = torch.cat([current_features, expanded_sep], dim=1)
            history_feature_list.append(current_with_sep)
        
        # 直接拼接：[hist1+sep] + [hist2+sep] + ... + [current]
        all_features = history_feature_list + [current_features]
        return torch.cat(all_features, dim=1)

    # def get_enhanced_features_with_separators(self, current_features, separator_token, 
    #                                          max_frame_num, camera_key="base_0_rgb"):
    #     """
    #     🔥 修改：从指定相机获取历史特征
        
    #     Args:
    #         current_features: 当前帧特征
    #         separator_token: 分隔符
    #         max_frame_num: 最大历史帧数
    #         camera_key: 相机名称
    #     """
    #     history_feature_list = self.sample_uniform_frames(self.num_sampled_frames, camera_key)
        
    #     # 将历史特征移回GPU
    #     device = current_features.device
    #     dtype = current_features.dtype
    #     history_feature_list = [f.to(device=device, dtype=dtype) for f in history_feature_list]
        
    #     # 填充到max_frame_num
    #     while len(history_feature_list) < max_frame_num:
    #         expanded_sep = separator_token.expand(current_features.shape[0], -1, -1).to(device=device, dtype=dtype)
    #         current_with_sep = torch.cat([current_features, expanded_sep], dim=1)
    #         history_feature_list.append(current_with_sep)
        
    #     # 直接拼接：[hist1+sep] + [hist2+sep] + ... + [current]
    #     all_features = history_feature_list + [current_features]
    #     return torch.cat(all_features, dim=1)

    def get_buffer_info(self, camera_key=None):
        """
        🔥 修改：获取指定相机或所有相机的缓存状态
        """
        if camera_key:
            buffer = self.buffers.get(camera_key, [])
            return {
                "camera": camera_key,
                "total_frames": len(buffer),
                "buffer_size_mb": sum(f.numel() * f.element_size() for f in buffer) / (1024 ** 2),
            }
        else:
            # 返回所有相机的信息
            return {
                cam: {
                    "total_frames": len(buf),
                    "buffer_size_mb": sum(f.numel() * f.element_size() for f in buf) / (1024 ** 2),
                }
                for cam, buf in self.buffers.items()
            }

    def clear(self, camera_key=None):
        """
        🔥 修改：清空指定相机或所有相机的历史缓存
        
        Args:
            camera_key: 指定相机名称，None则清空所有
        """
        if camera_key:
            self.buffers[camera_key].clear()
        else:
            for buf in self.buffers.values():
                buf.clear()
        self.frame_count = 0

    def __len__(self):
        """返回所有相机的总历史帧数量"""
        return sum(len(buf) for buf in self.buffers.values())