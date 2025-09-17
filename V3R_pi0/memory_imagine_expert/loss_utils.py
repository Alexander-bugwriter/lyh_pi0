
import torch
import torch.nn as nn
import torch.nn.functional as F

class MemoryImagineLoss(nn.Module):
    """Memory-Imagine差异化损失函数"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 语义类别索引
        self.static_classes = [0]  # 静态背景
        self.env_classes = [0, 1, 3, 4]  # 环境类别（背景、任务相关、任务无关、工作空间）
        self.robot_classes = [2]  # 机器人末端执行器
    
    def forward(self, pred_4d, gt_4d):
        """
        计算时间差异化损失
        
        Args:
            pred_4d: (B, T, X, Y, Z, C) 预测的4D占用
            gt_4d: (B, T, X, Y, Z, C) 真实的4D占用
        """
        total_loss = 0.0
        
        # Memory阶段损失 (t-30:t-1)
        memory_pred = pred_4d[:, :self.config.memory_frames]
        memory_gt = gt_4d[:, :self.config.memory_frames]
        memory_loss = F.cross_entropy(
            memory_pred.reshape(-1, self.config.semantic_classes),
            memory_gt.reshape(-1, self.config.semantic_classes).argmax(dim=-1)
        )
        total_loss += memory_loss * self.config.memory_weight
        
        # 当前时刻损失 (t0)
        current_pred = pred_4d[:, self.config.memory_frames]
        current_gt = gt_4d[:, self.config.memory_frames]
        current_loss = F.cross_entropy(
            current_pred.reshape(-1, self.config.semantic_classes),
            current_gt.reshape(-1, self.config.semantic_classes).argmax(dim=-1)
        )
        total_loss += current_loss * self.config.current_weight
        
        # Imagine阶段损失 (t+1:t+10)
        imagine_pred = pred_4d[:, self.config.memory_frames + 1:]
        imagine_gt = gt_4d[:, self.config.memory_frames + 1:]
        
        # 环境类别：高权重
        env_pred = imagine_pred[..., self.env_classes]
        env_gt = imagine_gt[..., self.env_classes]
        env_loss = F.cross_entropy(
            env_pred.reshape(-1, len(self.env_classes)),
            env_gt.reshape(-1, len(self.env_classes)).argmax(dim=-1)
        )
        total_loss += env_loss * self.config.imagine_env_weight
        
        # 机器人类别：低权重，鼓励多样性
        robot_pred = imagine_pred[..., self.robot_classes]
        robot_gt = imagine_gt[..., self.robot_classes]
        robot_loss = F.mse_loss(robot_pred, robot_gt)  # 用MSE鼓励多样性
        total_loss += robot_loss * self.config.imagine_robot_weight
        
        return total_loss
