#!/usr/bin/env python3
"""
路径修正脚本 - 在算力云服务器上运行一次即可
"""

import os
import sys
import pandas as pd
import json
import yaml

# 获取当前项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))  # 或手动设置为 '/root/xx'

print("=" * 60)
print("路径修正工具")
print(f"项目根目录: {PROJECT_ROOT}")
print("=" * 60)

# 1. 修正 collected_windows.csv 中的 window_data_path
csv_path = os.path.join(PROJECT_ROOT, "storage/autotune_results/collected_windows.csv")
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    print(f"修正前: {df['window_data_path'].iloc[0] if len(df) > 0 else '无数据'}")
    
    # 统一替换为 Linux 路径
    df['window_data_path'] = df['window_data_path'].str.replace('\\\\', '/', regex=False)
    df['window_data_path'] = df['window_data_path'].str.replace('storage/autotune_results', 
                                                                  f'{PROJECT_ROOT}/storage/autotune_results')
    # 处理绝对路径前缀
    df['window_data_path'] = df['window_data_path'].str.replace(r'^.*?storage/autotune_results/', 
                                                                f'{PROJECT_ROOT}/storage/autotune_results/', 
                                                                regex=True)
    
    df.to_csv(csv_path, index=False)
    print(f"修正后: {df['window_data_path'].iloc[0] if len(df) > 0 else '无数据'}")
    print("✅ collected_windows.csv 已修正")
else:
    print(f"⚠️ 文件不存在: {csv_path}")

# 2. 修正 config.yaml
config_path = os.path.join(PROJECT_ROOT, "experiments/autotune/config.yaml")
if os.path.exists(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 修改路径配置
    config['llog_dir'] = os.path.join(PROJECT_ROOT, "llog")
    config['output_dir'] = os.path.join(PROJECT_ROOT, "storage/autotune_results")
    
    # 如果有其他路径配置，一并修改
    if 'datasets' in config and config['datasets']:
        for ds in config['datasets']:
            if 'data_dir' in ds:
                ds['data_dir'] = os.path.join(PROJECT_ROOT, ds['data_dir'])
    
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True)
    print("✅ config.yaml 已修正")
else:
    print(f"⚠️ 文件不存在: {config_path}")

# 3. 检查窗口数据文件是否存在
window_dir = os.path.join(PROJECT_ROOT, "storage/autotune_results/window_data")
if os.path.exists(window_dir):
    pkl_files = [f for f in os.listdir(window_dir) if f.endswith('.pkl')]
    print(f"📊 window_data 目录: {len(pkl_files)} 个 .pkl 文件")
else:
    print(f"⚠️ window_data 目录不存在: {window_dir}")
    print("   请确保已将 window_data 目录复制到服务器")

print("\n✅ 路径修正完成！")