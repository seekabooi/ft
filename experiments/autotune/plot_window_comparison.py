#!/usr/bin/env python
"""
从测试日志中提取每个窗口的 MASE，绘制各模式对比折线图

用法：
    python -m experiments.autotune.plot_window_comparison --resume llog/cs2

输出：
    llog/cs2/semantic_vs_rl_results/window_comparison.png
"""

import os
import sys
import re
import json
import argparse
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def parse_log_file(log_path: str) -> Dict[str, Dict[int, float]]:
    """
    从详细日志中解析每个模式、每个窗口的 MASE
    
    日志格式示例：
    [2026-07-01 17:30:08]    ✅ 完成: MASE=0.7908
    [2026-07-01 17:30:08]    ✅ 完成: MASE=1.2345
    [2026-07-01 17:30:08]    ✅ 完成: MASE=0.9876
    """
    if not os.path.exists(log_path):
        print(f"⚠️ 日志文件不存在: {log_path}")
        return {}
    
    # 模式映射：日志中的模式名 -> 显示名
    mode_map = {
        'no_rule': 'no_rule',
        'semantic_top1': 'semantic_top1',
        'semantic_top5': 'semantic_top5',
        'semantic_top10': 'semantic_top10',
        'semantic_topAll': 'semantic_topAll',
        'semantic_top30_theta_max': 'top30% + θ_max',
        'semantic_top50_theta_max': 'top50% + θ_max',
    }
    
    # 存储结果: {模式名: {窗口ID: MASE}}
    results = {}
    
    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    current_mode = None
    
    for line in lines:
        # 检测当前模式
        # 格式: 📊 开始评估模式: semantic_top30_theta_max (共 50 个窗口)
        mode_match = re.search(r'开始评估模式: (\S+)', line)
        if mode_match:
            current_mode = mode_match.group(1)
            if current_mode not in results:
                results[current_mode] = {}
            continue
        
        # 提取 MASE
        # 格式: ✅ 完成: MASE=0.7908
        if '✅ 完成: MASE=' in line:
            mase_match = re.search(r'MASE=([0-9.]+)', line)
            if mase_match and current_mode:
                mase = float(mase_match.group(1))
                # 提取窗口ID
                window_match = re.search(r'窗口 (\d+)', line)
                if window_match:
                    window_id = int(window_match.group(1))
                    results[current_mode][window_id] = mase
                else:
                    # 如果日志中没有窗口ID，按顺序分配
                    idx = len(results[current_mode])
                    results[current_mode][idx + 1] = mase
    
    return results


def plot_window_comparison(results: Dict[str, Dict[int, float]], output_dir: str, title: str = None):
    """
    绘制各模式窗口 MASE 对比折线图
    """
    if not results:
        print("❌ 没有数据可绘图")
        return None
    
    # 过滤掉空模式
    results = {k: v for k, v in results.items() if v}
    
    if not results:
        print("❌ 没有有效数据")
        return None
    
    # 获取所有窗口ID（取所有模式窗口的并集）
    all_window_ids = set()
    for windows in results.values():
        all_window_ids.update(windows.keys())
    all_window_ids = sorted(all_window_ids)
    
    if not all_window_ids:
        print("❌ 没有找到窗口ID")
        return None
    
    # 按模式排序
    mode_order = ['no_rule', 'semantic_top1', 'semantic_top30_theta_max', 
                  'semantic_top50_theta_max', 'semantic_topAll_theta_max']
    
    # 显示名称映射
    display_names = {
        'no_rule': 'no_rule (基线)',
        'semantic_top1': 'semantic_top1',
        'semantic_top30_theta_max': 'top30% + θ_max',
        'semantic_top50_theta_max': 'top50% + θ_max',
        'semantic_topAll_theta_max': '全局 θ_max',
    }
    
    # 颜色映射
    colors = {
        'no_rule': '#808080',
        'semantic_top1': '#2E86AB',
        'semantic_top30_theta_max': '#F5A623',
        'semantic_top50_theta_max': '#E68A2E',
        'semantic_topAll_theta_max': '#D4693A',
    }
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # 按模式顺序绘制
    for mode in mode_order:
        if mode not in results:
            continue
        
        windows = results[mode]
        # 按窗口ID排序
        sorted_items = sorted(windows.items())
        window_ids = [w[0] for w in sorted_items]
        mases = [w[1] for w in sorted_items]
        
        if not window_ids:
            continue
        
        display_name = display_names.get(mode, mode)
        color = colors.get(mode, '#000000')
        
        ax.plot(window_ids, mases, marker='o', color=color, 
                linewidth=2, markersize=4, label=display_name)
    
    ax.set_xlabel('窗口ID', fontsize=12)
    ax.set_ylabel('MASE', fontsize=12)
    ax.set_title(title or '各模式窗口 MASE 对比折线图', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'window_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 折线图已保存: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="从测试日志生成窗口 MASE 对比折线图")
    parser.add_argument('--resume', type=str, required=True,
                        help='运行目录（如 llog/cs2）')
    parser.add_argument('--title', type=str, default=None,
                        help='图表标题')
    args = parser.parse_args()
    
    run_dir = args.resume
    if not os.path.exists(run_dir):
        test_path = os.path.join("llog", args.resume)
        if os.path.exists(test_path):
            run_dir = test_path
        else:
            print(f"❌ 目录不存在: {run_dir}")
            return
    
    # 解析日志
    log_path = os.path.join(run_dir, 'semantic_vs_rl_detailed.log')
    results = parse_log_file(log_path)
    
    if not results:
        print("❌ 未能从日志中解析出数据")
        print("   请确保日志文件存在且包含 MASE 记录")
        return
    
    print(f"\n📊 解析到 {len(results)} 个模式的数据:")
    for mode, windows in results.items():
        print(f"   - {mode}: {len(windows)} 个窗口")
    
    # 绘图
    output_dir = os.path.join(run_dir, 'semantic_vs_rl_results')
    plot_window_comparison(results, output_dir, args.title)
    
    print("\n✅ 完成")


if __name__ == '__main__':
    main()