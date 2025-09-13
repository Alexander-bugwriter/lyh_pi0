"""
超简化数据集配置工具 - 只读取基本信息，让LeRobotDataset自己处理统计
"""
import json
from pathlib import Path
from typing import Dict, Any

def get_dataset_info(dataset_root: str) -> Dict[str, Any]:
    """只读取基本信息，不处理统计数据"""
    meta_dir = Path(dataset_root) / "meta"
    
    # 读取基础信息
    with open(meta_dir / "info.json", 'r') as f:
        info = json.load(f)
    
    # 读取任务（可选）
    tasks = ["complete the task"]  # 默认值
    tasks_file = meta_dir / "tasks.jsonl"
    if tasks_file.exists():
        tasks = []
        with open(tasks_file, 'r') as f:
            for line in f:
                if line.strip():
                    task_data = json.loads(line.strip())
                    tasks.append(task_data.get('task', 'complete the task'))
    
    return {
        'fps': info['fps'],
        'features': info['features'],
        'tasks': tasks
    }

def generate_delta_timestamps(fps: int, features: Dict[str, Any], action_horizon: int) -> Dict[str, list]:
    """生成delta_timestamps"""
    delta_timestamps = {}
    
    # 图像特征 - 只要当前帧
    for key, feature in features.items():
        if feature.get('dtype') == 'image':
            delta_timestamps[key] = [0]
    
    # 状态特征 - 只要当前帧
    if 'state' in features:
        delta_timestamps['state'] = [0]
    
    # 动作特征 - 需要action_horizon步
    if 'actions' in features:
        delta_timestamps['actions'] = [i / fps for i in range(action_horizon)]
    
    return delta_timestamps
