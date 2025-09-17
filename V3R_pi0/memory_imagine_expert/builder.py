
import torch
from .feature_pool import AsyncFeaturePool
from .time_decoder import MemoryImagineDecoder
from .base_config import MemoryImagineConfig

def build_memory_imagine_expert(config, device='cuda'):
    """
    构建Memory-Imagine Expert组件
    
    Args:
        config: 包含memory_imagine_config的配置对象
        device: 设备类型
    
    Returns:
        dict: 包含feature_pool和decoder的字典
    """
    # 获取配置，使用默认配置作为fallback
    memory_config = getattr(config, 'memory_imagine_config', MemoryImagineConfig())
    if isinstance(memory_config, dict):
        memory_config = MemoryImagineConfig(**memory_config)
    
    # 构建异步特征池
    feature_pool = AsyncFeaturePool(
        feature_dim=memory_config.paligemma_feature_dim,
        state_dim=memory_config.robot_state_dim,
        device=device
    )
    
    # 构建解码器
    decoder = MemoryImagineDecoder(memory_config).to(device)
    
    # 启动特征池后台线程
    feature_pool.start_background_worker()
    
    return {
        'feature_pool': feature_pool,
        'decoder': decoder,
        'config': memory_config
    }
