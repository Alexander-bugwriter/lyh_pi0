"""
简洁的本地LeRobot数据集v2.0到v2.1转换脚本
直接使用官方lerobot库的转换函数
"""

import json
import os
from pathlib import Path

# 直接引用官方lerobot的转换函数
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import write_info
from lerobot.common.datasets.v21.convert_stats import convert_stats


def convert_local_dataset_v20_to_v21(dataset_path):
    """转换本地数据集从v2.0到v2.1，使用官方转换函数"""
    
    dataset_path = Path(dataset_path)
    print(f"转换数据集: {dataset_path}")
    
    # 文件路径
    info_file = dataset_path / "meta" / "info.json"
    stats_file = dataset_path / "meta" / "stats.json" 
    episodes_stats_file = dataset_path / "meta" / "episodes_stats.jsonl"
    
    # 读取info.json
    with open(info_file) as f:
        info = json.load(f)
    
    print(f"当前版本: {info.get('codebase_version')}")
    
    # 创建LeRobotDataset实例来使用官方转换函数
    # 设置root参数指向本地数据集路径的父目录
    dataset_name = dataset_path.name
    
    # 使用官方LeRobotDataset加载本地数据集
    dataset = LeRobotDataset(
        repo_id=dataset_name,  # 使用目录名作为repo_id
        root=dataset_path,     # 指向数据集父目录
        revision="main",
        
    )
    
    print("使用官方convert_stats函数生成episodes_stats.jsonl...")
    
    # 删除可能存在的旧episodes_stats.jsonl
    if episodes_stats_file.exists():
        episodes_stats_file.unlink()
    
    # 使用官方转换函数
    convert_stats(dataset, num_workers=4)
    
    print(f"生成 episodes_stats.jsonl 完成")
    
    # 更新info.json版本
    info['codebase_version'] = 'v2.1'
    write_info(info, dataset_path)
    
    # 备份并删除旧的stats.json
    if stats_file.exists():
        stats_backup = dataset_path / "meta" / "stats.json.backup"
        os.rename(stats_file, stats_backup)
        print(f"备份 stats.json -> {stats_backup.name}")
    
    print("转换完成")
    print(f"更新版本到 v2.1")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("用法: python convert_lerobot_v20_to_v21_local.py <dataset_path>")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    convert_local_dataset_v20_to_v21(dataset_path)
