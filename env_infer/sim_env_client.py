"""
支持reset字段的模拟环境客户端
观测数据格式: {
    "observation/image": ...,
    "observation/state": ..., 
    "prompt": "...",
    "reset": true/false  # 控制是否需要重置
}
"""

import time
import logging
import numpy as np
import torch
from typing import Dict
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from websocket_client_policy import WebsocketClientPolicy

logging.basicConfig(level=logging.INFO)


class SimulatedEnvironmentWithReset:
    def __init__(self, host="0.0.0.0", port=8000, device="cuda"):
        """初始化模拟环境"""
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.client = WebsocketClientPolicy(host, port)
        logging.info(f"Connected to server at {host}:{port}")
        logging.info(f"Using device: {self.device}")
        
    def create_mock_observation(self, step_num: int, need_reset: bool = False) -> Dict:
        """创建包含reset字段的观测数据"""
        
        # 创建模拟图像数据 (224, 224, 3)
        image_np = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        
        # 添加一些变化模拟真实场景
        noise = np.random.randint(-10, 10, (224, 224, 3)).astype(np.int16)
        image_np = np.clip(image_np.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # 创建模拟状态数据 (8,)
        state_np = np.random.randn(8).astype(np.float32) * 0.2
        state_np += step_num * 0.01  # 添加步骤相关变化
        
        # 构建完整的观测数据（包含reset字段）
        observation = {
            "observation/image": image_np,           # 主相机图像
            "observation/wrist_image": image_np,     # 手腕相机图像
            "observation/state": state_np,           # 机器人状态
            "prompt": f"simulate task step {step_num}",  # 任务描述
            "reset": need_reset                      # 关键：reset字段
        }
        
        return observation
    
    def send_observation_with_reset(self, step: int, need_reset: bool = False):
        """发送包含reset字段的观测"""
        try:
            # 创建观测数据
            observation = self.create_mock_observation(step, need_reset)
            
            reset_status = "RESET" if need_reset else "NORMAL"
            logging.info(f"步骤 {step} - 发送观测 [{reset_status}]")
            
            # 发送观测并接收动作
            response = self.client.infer(observation)
            
            if "actions" in response:
                actions = response["actions"]
                logging.info(f"步骤 {step} - 收到动作，形状: {np.array(actions).shape}")
                return True
            else:
                logging.warning(f"步骤 {step} - 未收到动作数据")
                return False
                
        except Exception as e:
            logging.error(f"步骤 {step} 出错: {e}")
            return False
    
    def run_simulation_with_resets(self, num_steps=10, delay=2.0, reset_intervals=None):
        """
        运行包含reset的模拟
        
        Args:
            num_steps: 总步数
            delay: 每步间隔
            reset_intervals: 需要reset的步骤列表，例如 [1, 5, 8] 表示第1、5、8步需要reset
        """
        if reset_intervals is None:
            reset_intervals = [1]  # 默认第一步reset
        
        logging.info(f"开始模拟，总步数: {num_steps}")
        logging.info(f"Reset步骤: {reset_intervals}")
        
        success_count = 0
        
        for step in range(1, num_steps + 1):
            # 检查是否需要reset
            need_reset = step in reset_intervals
            
            # 发送观测
            if self.send_observation_with_reset(step, need_reset):
                success_count += 1
            
            time.sleep(delay)
        
        logging.info(f"模拟完成！成功率: {success_count}/{num_steps}")
        return success_count
    
    def run_episode_based_simulation(self, episodes=3, steps_per_episode=5, delay=1.5):
        """
        运行基于episode的模拟（每个episode开始时reset）
        
        Args:
            episodes: episode数量
            steps_per_episode: 每个episode的步数
            delay: 每步间隔
        """
        logging.info(f"开始基于episode的模拟: {episodes} episodes, 每episode {steps_per_episode} 步")
        
        total_success = 0
        total_steps = 0
        
        for episode in range(1, episodes + 1):
            logging.info(f"\n Episode {episode}/{episodes} 开始")
            
            for step in range(1, steps_per_episode + 1):
                total_steps += 1
                global_step = (episode - 1) * steps_per_episode + step
                
                # 每个episode的第一步需要reset
                need_reset = (step == 1)
                
                if self.send_observation_with_reset(global_step, need_reset):
                    total_success += 1
                
                time.sleep(delay)
            
            logging.info(f" Episode {episode} 完成")
        
        logging.info(f"\n 总体模拟完成！成功率: {total_success}/{total_steps}")
        return total_success
    
    def close(self):
        """关闭连接"""
        if hasattr(self, 'client') and self.client:
            self.client.close()


def main():
    """主函数 - 演示不同的模拟模式"""
    sim_env = None
    try:
        # 创建模拟环境
        sim_env = SimulatedEnvironmentWithReset(
            host="0.0.0.0", 
            port=8000, 
            device="cuda"
        )
        
        print("\n选择模拟模式:")
        print("1. 自定义reset步骤")
        print("2. 基于episode的模拟")
        print("3. 简单模拟（只有开始reset）")
        
        
        choice=1

        if choice == 1:
            # 自定义reset模拟
            sim_env.run_simulation_with_resets(
                num_steps=10,
                delay=2.0,
                reset_intervals=[1, 4, 7]  # 第1、4、7步reset
            )
            
        elif choice == 2:
            # 基于episode的模拟
            sim_env.run_episode_based_simulation(
                episodes=3,
                steps_per_episode=4,
                delay=1.5
            )
            
        else:
            # 简单模拟
            sim_env.run_simulation_with_resets(
                num_steps=8,
                delay=2.0,
                reset_intervals=[1]  # 只有第一步reset
            )
        
    except KeyboardInterrupt:
        logging.info("用户中断模拟")
    except Exception as e:
        logging.error(f"模拟失败: {e}")
        raise
    finally:
        if sim_env:
            sim_env.close()


if __name__ == "__main__":
    main()
