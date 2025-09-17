
from dataclasses import dataclass
from typing import Optional

@dataclass
class MemoryImagineConfig:
    """Memory-Imagine Expert配置"""
    
    # 特征配置
    paligemma_feature_dim: int = 2048
    robot_state_dim: int = 6  # [x,y,z,rx,ry,rz]
    hidden_dim: int = 1024
    
    # 时序配置
    memory_frames: int = 30   # 3秒 * 10Hz
    current_frames: int = 1   # 当前时刻
    imagine_frames: int = 10  # 1秒 * 10Hz
    
    # 空间配置
    spatial_x: int = 64
    spatial_y: int = 64
    spatial_z: int = 32
    semantic_classes: int = 5  # 静态背景、任务相关、末端执行器、任务无关、工作空间
    
    # 网络配置
    num_transformer_layers: int = 6
    num_attention_heads: int = 8
    dropout: float = 0.1
    
    # 训练配置
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    
    # 损失配置
    memory_weight: float = 1.0      # Memory阶段权重
    current_weight: float = 1.2     # 当前时刻权重
    imagine_env_weight: float = 0.8 # Imagine环境权重
    imagine_robot_weight: float = 0.3 # Imagine机器人权重
    
    @property
    def total_time_frames(self) -> int:
        return self.memory_frames + self.current_frames + self.imagine_frames
    
    @property
    def total_spatial_dim(self) -> int:
        return self.spatial_x * self.spatial_y * self.spatial_z * self.semantic_classes

