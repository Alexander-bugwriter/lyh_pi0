# import torch

# class UnlimitedHistoryBuffer:
#     """无限长度历史特征缓存管理 - 🔥 支持多相机分别存储"""

#     def __init__(self, config):
#         self.num_sampled_frames = config.num_sampled_history_frames
#         # self.sampling_method = config.history_sampling_method
#         self.sampling_method = getattr(config, 'history_sampling_method', 'adaptive')
#         # 🔥 修改：为每个相机维护独立的buffer
#         self.buffers = {
#             "base_0_rgb": [],
#             "left_wrist_0_rgb": [],
#             "right_wrist_0_rgb": []
#         }
#         self.frame_count = 0

#     def append(self, current_features, separator_token, camera_key="base_0_rgb"):
#         """
#         🔥 修改：存储时指定相机
        
#         Args:
#             current_features: 当前帧特征
#             separator_token: 分隔符
#             camera_key: 相机名称（默认base）
#         """
#         batch_size = current_features.shape[0]
#         expanded_sep = separator_token.expand(batch_size, -1, -1)
#         stored_item = torch.cat([current_features, expanded_sep], dim=1)
        
#         # 存到对应相机的buffer
#         self.buffers[camera_key].append(stored_item.detach().clone().cpu())
#         self.frame_count += 1

 
#     def sample_history_frames(self, num_frames=3, camera_key="base_0_rgb"):
#         """
#         🔥 修改：采用和 extract_feature 相同的采样策略
        
#         采样策略:
#         - 如果 t >= 30: [t-30, t-20, t-10]
#         - 如果 3 <= t < 30: [0, t//3, 2*t//3]
#         - 如果 t < 3: [0, 1, t] 或根据t的大小填充
        
#         Args:
#             num_frames: 历史帧数量（默认3）
#             camera_key: 相机名称
        
#         Returns:
#             selected_features: List[Tensor] 采样的历史帧特征
#         """
#         buffer = self.buffers.get(camera_key, [])
#         current_t = len(buffer)  # 当前帧索引（buffer长度）
        
#         if current_t == 0:
#             return []
        
#         # 🔥 使用和 extract_feature 相同的采样逻辑
#         if current_t >= 30:
#             # 策略1: 固定间隔采样
#             indices = [current_t - 30, current_t - 20, current_t - 10]
#         elif current_t >= num_frames:
#             # 策略2: 均匀分布采样
#             indices = [0, current_t // 3, (2 * current_t) // 3]
#         else:
#             # 策略3: 填充策略
#             if current_t == 1:
#                 indices = [0, 0, 0]
#             elif current_t == 2:
#                 indices = [0, 1, 1]
#             else:  # current_t == 3
#                 indices = [0, 1, 2]
        
#         # 确保索引有效
#         indices = [min(max(0, idx), current_t - 1) for idx in indices]
        
#         selected_features = [buffer[i] for i in indices]
#         return selected_features
    
#     def get_enhanced_features_with_separators(self, current_features, separator_token, 
#                                          max_frame_num, camera_key="base_0_rgb"):
#         """
#         获取拼接历史后的特征
        
#         Args:
#             current_features: 当前帧特征
#             separator_token: 分隔符
#             max_frame_num: 期望的历史帧数量
#             camera_key: 相机名称
#         """
#         # 🔥 使用新的采样方法
#         history_feature_list = self.sample_history_frames(max_frame_num, camera_key)
        
#         # 将历史特征移回GPU
#         device = current_features.device
#         dtype = current_features.dtype
#         history_feature_list = [f.to(device=device, dtype=dtype) for f in history_feature_list]
        
#         # 🔥 填充到max_frame_num（如果历史不足）
#         while len(history_feature_list) < max_frame_num:
#             expanded_sep = separator_token.expand(current_features.shape[0], -1, -1).to(device=device, dtype=dtype)
#             current_with_sep = torch.cat([current_features, expanded_sep], dim=1)
#             history_feature_list.append(current_with_sep)
        
#         # 直接拼接：[hist1+sep] + [hist2+sep] + ... + [current]
#         all_features = history_feature_list + [current_features]
#         return torch.cat(all_features, dim=1)


#     def get_buffer_info(self, camera_key=None):
#         """
#         🔥 修改：获取指定相机或所有相机的缓存状态
#         """
#         if camera_key:
#             buffer = self.buffers.get(camera_key, [])
#             return {
#                 "camera": camera_key,
#                 "total_frames": len(buffer),
#                 "buffer_size_mb": sum(f.numel() * f.element_size() for f in buffer) / (1024 ** 2),
#             }
#         else:
#             # 返回所有相机的信息
#             return {
#                 cam: {
#                     "total_frames": len(buf),
#                     "buffer_size_mb": sum(f.numel() * f.element_size() for f in buf) / (1024 ** 2),
#                 }
#                 for cam, buf in self.buffers.items()
#             }

#     def clear(self, camera_key=None):
#         """
#         🔥 修改：清空指定相机或所有相机的历史缓存
        
#         Args:
#             camera_key: 指定相机名称，None则清空所有
#         """
#         if camera_key:
#             self.buffers[camera_key].clear()
#         else:
#             for buf in self.buffers.values():
#                 buf.clear()
#         self.frame_count = 0

#     def __len__(self):
#         """返回所有相机的总历史帧数量"""
#         return sum(len(buf) for buf in self.buffers.values())
import torch

class UnlimitedHistoryBuffer:
    """无限长度历史特征缓存管理 - 🔥 支持多相机分别存储"""

    def __init__(self, config):
        self.num_sampled_frames = config.num_sampled_history_frames
        self.sampling_method = getattr(config, 'history_sampling_method', 'adaptive')
        # 🔥 为每个相机维护独立的buffer
        self.buffers = {
            "base_0_rgb": [],
            "left_wrist_0_rgb": [],
            "right_wrist_0_rgb": []
        }
        self.frame_count = 0

    def append(self, current_features, separator_token, camera_key="base_0_rgb"):
        """
        🔥 修改：只存储特征，不存储分隔符（因为分隔符现在是时序感知的）
        
        Args:
            current_features: 当前帧特征
            separator_token: TemporalSeparatorToken 模块（这里不使用，保持接口兼容）
            camera_key: 相机名称
        """
        # ✅ 只存储特征本身，不拼接分隔符
        self.buffers[camera_key].append(current_features.detach().clone().cpu())
        self.frame_count += 1

    def sample_history_frames(self, num_frames=3, camera_key="base_0_rgb"):
        """
        采样历史帧
        
        返回: List[Tensor] - 纯特征，不含分隔符
        """
        buffer = self.buffers.get(camera_key, [])
        current_t = len(buffer)
        
        if current_t == 0:
            return []
        
        # 采样策略（保持你原来的逻辑）
        if current_t >= 30:
            indices = [current_t - 30, current_t - 20, current_t - 10]
        elif current_t >= num_frames:
            indices = [0, current_t // 3, (2 * current_t) // 3]
        else:
            if current_t == 1:
                indices = [0, 0, 0]
            elif current_t == 2:
                indices = [0, 1, 1]
            else:  # current_t == 3
                indices = [0, 1, 2]
        
        indices = [min(max(0, idx), current_t - 1) for idx in indices]
        
        selected_features = [buffer[i] for i in indices]
        return selected_features
    
    def get_enhanced_features_with_separators(self, current_features, separator_token, 
                                         max_frame_num, camera_key="base_0_rgb"):
        """
        🔥 修改：使用时序感知的分隔符
        
        Args:
            current_features: 当前帧特征
            separator_token: TemporalSeparatorToken 模块
            max_frame_num: 历史帧数量（3）
            camera_key: 相机名称
        """
        # 采样历史帧
        history_feature_list = self.sample_history_frames(max_frame_num, camera_key)
        
        if not history_feature_list:
            return current_features
        
        device = current_features.device
        dtype = current_features.dtype
        batch_size = current_features.shape[0]
        
        # 将历史特征移到GPU
        history_feature_list = [f.to(device=device, dtype=dtype) for f in history_feature_list]
        
        # ✅ 填充到 max_frame_num（如果历史不足，用当前帧填充）
        while len(history_feature_list) < max_frame_num:
            history_feature_list.append(current_features.clone())
        
        # 🔥 拼接序列：[hist1] + [sep_3] + [hist2] + [sep_2] + [hist3] + [sep_1] + [current]
        sequence_parts = []
        num_history = len(history_feature_list)
        
        for i, hist_feat in enumerate(history_feature_list):
            sequence_parts.append(hist_feat)
            
            # time_step: 3, 2, 1 (从最远到最近)
            time_step = num_history - i
            
            # ✅ 调用模块生成时序感知的分隔符
            separator = separator_token(
                time_step=time_step,
                batch_size=batch_size,
                device=device,
                dtype=dtype
            )
            sequence_parts.append(separator)
        
        # 当前帧（不加分隔符）
        sequence_parts.append(current_features)
        
        return torch.cat(sequence_parts, dim=1)

    def get_buffer_info(self, camera_key=None):
        """获取缓存状态"""
        if camera_key:
            buffer = self.buffers.get(camera_key, [])
            return {
                "camera": camera_key,
                "total_frames": len(buffer),
                "buffer_size_mb": sum(f.numel() * f.element_size() for f in buffer) / (1024 ** 2),
            }
        else:
            return {
                cam: {
                    "total_frames": len(buf),
                    "buffer_size_mb": sum(f.numel() * f.element_size() for f in buf) / (1024 ** 2),
                }
                for cam, buf in self.buffers.items()
            }

    def clear(self, camera_key=None):
        """清空历史缓存"""
        if camera_key:
            self.buffers[camera_key].clear()
        else:
            for buf in self.buffers.values():
                buf.clear()
        self.frame_count = 0

    def __len__(self):
        """返回所有相机的总历史帧数量"""
        return sum(len(buf) for buf in self.buffers.values())