
import torch
import threading
import time
from queue import Queue, Empty
from typing import Optional, Tuple

class AsyncFeaturePool:
    """异步特征池 - 解耦控制系统实时性"""
    
    def __init__(self, feature_dim: int, state_dim: int, device: str = 'cuda'):
        self.feature_dim = feature_dim
        self.state_dim = state_dim
        self.device = device
        
        # 特征存储
        self.latest_features = None
        self.latest_state = None
        self.latest_timestamp = None
        
        # 异步队列
        self.update_queue = Queue(maxsize=2)  # 小队列，避免积压
        
        # 线程控制
        self.lock = threading.RLock()
        self.is_running = True
        self.worker_thread = None
        self.has_data = False
    
    def update_async(self, paligemma_features: torch.Tensor, robot_state: torch.Tensor) -> None:
        """
        异步更新特征（非阻塞）
        
        Args:
            paligemma_features: (B, L, 2048) PaliGemma特征
            robot_state: (B, 6) 机器人状态
        """
        if not self.is_running:
            return
            
        update_data = {
            'features': paligemma_features.detach().cpu(),
            'state': robot_state.detach().cpu(),
            'timestamp': time.time()
        }
        
        # 非阻塞放入队列
        if self.update_queue.full():
            # 队列满时丢弃最旧的数据
            self.update_queue.get_nowait()
        
        self.update_queue.put_nowait(update_data)
    
    def get_latest(self) -> Optional[Tuple[torch.Tensor, torch.Tensor, float]]:
        """
        获取最新特征
        
        Returns:
            (features, state, timestamp) or None
        """
        if not self.has_data:
            return None
            
        with self.lock:
            if self.latest_features is None:
                return None
            return (
                self.latest_features.to(self.device),
                self.latest_state.to(self.device), 
                self.latest_timestamp
            )
    
    def start_background_worker(self):
        """启动后台处理线程"""
        if self.worker_thread is not None:
            return
            
        self.worker_thread = threading.Thread(target=self._process_updates, daemon=True)
        self.worker_thread.start()
    
    def stop(self):
        """停止特征池"""
        self.is_running = False
        if self.worker_thread is not None:
            self.worker_thread.join(timeout=1.0)
    
    def _process_updates(self):
        """后台线程处理更新"""
        while self.is_running:
            try:
                # 获取更新数据
                data = self.update_queue.get(timeout=0.1)
                
                with self.lock:
                    self.latest_features = data['features']
                    self.latest_state = data['state']
                    self.latest_timestamp = data['timestamp']
                    self.has_data = True
                    
            except Empty:
                continue
            except Exception as e:
                print(f"FeaturePool background worker error: {e}")
                continue
