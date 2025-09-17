
import torch
import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class OccupancyRenderer:
    """4D占用渲染工具"""
    
    def __init__(self, config):
        self.config = config
        
        # 语义颜色映射
        self.semantic_colors = {
            0: [0.5, 0.5, 0.5],    # 静态背景 - 灰色
            1: [0.0, 1.0, 0.0],    # 任务相关物品 - 绿色
            2: [1.0, 0.0, 0.0],    # 末端执行器 - 红色
            3: [0.0, 0.0, 1.0],    # 任务无关物品 - 蓝色
            4: [1.0, 1.0, 0.0]     # 工作空间 - 黄色
        }
    
    def render_4d_sequence(self, occupancy_4d: torch.Tensor) -> List[np.ndarray]:
        """
        渲染4D占用序列
        
        Args:
            occupancy_4d: (B, T, X, Y, Z, C) 4D占用
            
        Returns:
            List[np.ndarray]: 渲染的帧列表
        """
        B, T, X, Y, Z, C = occupancy_4d.shape
        rendered_frames = []
        
        # 获取语义标签（取最大概率类别）
        semantic_labels = occupancy_4d.argmax(dim=-1)  # (B, T, X, Y, Z)
        
        for t in range(T):
            frame_3d = semantic_labels[0, t]  # (X, Y, Z)
            rendered_frame = self.render_3d_frame(frame_3d)
            rendered_frames.append(rendered_frame)
        
        return rendered_frames
    
    def render_3d_frame(self, frame_3d: torch.Tensor) -> np.ndarray:
        """
        渲染单个3D帧（投影到2D）
        
        Args:
            frame_3d: (X, Y, Z) 语义标签
            
        Returns:
            np.ndarray: (H, W, 3) RGB图像
        """
        X, Y, Z = frame_3d.shape
        
        # Z轴最大投影
        projection = torch.zeros(X, Y, 3)
        
        for x in range(X):
            for y in range(Y):
                # 找到Z轴上最上层的非背景语义
                z_values = frame_3d[x, y, :]
                non_bg_mask = z_values != 0  # 非背景
                
                if non_bg_mask.any():
                    # 取最上层的语义类别
                    top_z = z_values[non_bg_mask][-1].item()
                    color = self.semantic_colors.get(top_z, [0, 0, 0])
                    projection[x, y] = torch.tensor(color)
        
        return projection.numpy()
    
    def create_video(self, rendered_frames: List[np.ndarray], 
                    filename: str = 'memory_imagine.mp4'):
        """
        创建4D渲染视频
        
        Args:
            rendered_frames: 渲染的帧列表
            filename: 输出文件名
        """
        fig, ax = plt.subplots()
        
        def animate(frame_idx):
            ax.clear()
            ax.imshow(rendered_frames[frame_idx])
            ax.set_title(f'Frame {frame_idx}')
            ax.axis('off')
        
        anim = FuncAnimation(
            fig, animate, frames=len(rendered_frames), 
            interval=100, repeat=True
        )
        
        anim.save(filename, writer='pillow')
        plt.close()
