#!/usr/bin/env python
"""
检查当前策略的 RL 参数（θ / logit_weight）分布
用法：
    python check_rl_params.py --resume llog/cs2
    
功能：
    1. 显示所有策略的 θ 值（按从大到小排序）
    2. 显示 θ 分布统计（min, max, mean, median, quartiles）
    3. 按状态分组显示 θ 均值
    4. 高亮 θ > 0.5 的策略（高偏好）
    5. 显示 θ < -0.5 的策略（被冷落）
"""

import os
import sys
import json
import argparse
import numpy as np
from typing import List, Dict
from collections import Counter

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from experiments.autotune.skill_policy import SkillPolicy


def load_policies(run_dir: str) -> List[SkillPolicy]:
    """加载策略列表"""
    checkpoint_path = os.path.join(run_dir, "checkpoint.json")
    if not os.path.exists(checkpoint_path):
        print(f"❌ 检查点不存在: {checkpoint_path}")
        sys.exit(1)

    with open(checkpoint_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    policies_data = data.get('current_policies_data', [])
    policies = [SkillPolicy.from_dict(p) for p in policies_data]
    return policies


def analyze_rl_params(policies: List[SkillPolicy]):
    """分析 RL 参数分布"""
    print("\n" + "=" * 80)
    print("📊 策略 RL 参数 (θ / logit_weight) 分布报告")
    print("=" * 80)

    # 按 θ 从大到小排序
    sorted_policies = sorted(policies, key=lambda p: p.logit_weight, reverse=True)

    # 统计
    theta_values = [p.logit_weight for p in policies]
    status_theta = {}

    print(f"\n📋 策略总数: {len(policies)}")
    print("")

    # 打印表头
    print(f"{'排名':<6} {'策略名称':<25} {'状态':<12} {'θ (logit_weight)':<16} {'MASE':<12} {'选择次数':<10}")
    print("-" * 90)

    # 打印所有策略（按 θ 从大到小）
    for i, p in enumerate(sorted_policies, 1):
        # 收集状态分组
        if p.status not in status_theta:
            status_theta[p.status] = []
        status_theta[p.status].append(p.logit_weight)

        name = p.name[:22] if len(p.name) > 22 else p.name
        theta_str = f"{p.logit_weight:.6f}"
        mase_str = f"{p.avg_mase:.4f}" if p.avg_mase is not None else "N/A"
        selection_str = str(p.selection_count)

        # 高亮
        marker = ""
        if p.logit_weight > 0.5:
            marker = "🔥"
        elif p.logit_weight < -0.5:
            marker = "❄️"

        print(f"{i:<6} {name:<25} {p.status:<12} {theta_str:<16} {mase_str:<12} {selection_str:<10} {marker}")

    # ============================================================
    # 统计信息
    # ============================================================
    print("\n" + "=" * 80)
    print("📊 θ 分布统计")
    print("=" * 80)

    if theta_values:
        theta_arr = np.array(theta_values)
        print(f"   最小值: {np.min(theta_arr):.6f}")
        print(f"   最大值: {np.max(theta_arr):.6f}")
        print(f"   均值:   {np.mean(theta_arr):.6f}")
        print(f"   中位数: {np.median(theta_arr):.6f}")
        print(f"   标准差: {np.std(theta_arr):.6f}")
        print(f"   Q1:     {np.percentile(theta_arr, 25):.6f}")
        print(f"   Q3:     {np.percentile(theta_arr, 75):.6f}")

        # 正负统计
        positive = sum(1 for v in theta_arr if v > 0.01)
        negative = sum(1 for v in theta_arr if v < -0.01)
        zero = len(theta_arr) - positive - negative
        print(f"\n   θ > 0.01: {positive} 条策略 (偏好)")
        print(f"   θ < -0.01: {negative} 条策略 (冷落)")
        print(f"   θ ≈ 0: {zero} 条策略 (中性)")

        # 高偏好 / 高冷落
        high_pref = [p for p in sorted_policies if p.logit_weight > 0.5]
        high_cold = [p for p in sorted_policies if p.logit_weight < -0.5]

        if high_pref:
            print(f"\n   🔥 高偏好策略 (θ > 0.5): {len(high_pref)} 条")
            for p in high_pref[:5]:
                print(f"      - {p.name} (θ={p.logit_weight:.4f}, status={p.status})")
            if len(high_pref) > 5:
                print(f"      ... 还有 {len(high_pref) - 5} 条")

        if high_cold:
            print(f"\n   ❄️ 高冷落策略 (θ < -0.5): {len(high_cold)} 条")
            for p in high_cold[:5]:
                print(f"      - {p.name} (θ={p.logit_weight:.4f}, status={p.status})")
            if len(high_cold) > 5:
                print(f"      ... 还有 {len(high_cold) - 5} 条")

    # ============================================================
    # 按状态分组
    # ============================================================
    print("\n" + "=" * 80)
    print("📊 按状态分组的 θ 均值")
    print("=" * 80)

    for status, theta_list in status_theta.items():
        if theta_list:
            mean_theta = np.mean(theta_list)
            count = len(theta_list)
            print(f"   {status:<12}: {count:>4} 条策略, 平均 θ = {mean_theta:.6f}")

    # ============================================================
    # Top 10 和 Bottom 10
    # ============================================================
    print("\n" + "=" * 80)
    print("📊 Top 10 θ (最受偏好策略)")
    print("=" * 80)
    print(f"{'排名':<6} {'策略名称':<25} {'状态':<12} {'θ':<12} {'MASE':<12} {'选择次数':<10}")
    print("-" * 80)
    for i, p in enumerate(sorted_policies[:10], 1):
        name = p.name[:22] if len(p.name) > 22 else p.name
        print(f"{i:<6} {name:<25} {p.status:<12} {p.logit_weight:<12.6f} {p.avg_mase:<12.4f} {p.selection_count:<10}")

    print("\n" + "=" * 80)
    print("📊 Bottom 10 θ (最被冷落策略)")
    print("=" * 80)
    print(f"{'排名':<6} {'策略名称':<25} {'状态':<12} {'θ':<12} {'MASE':<12} {'选择次数':<10}")
    print("-" * 80)
    for i, p in enumerate(sorted_policies[-10:], 1):
        name = p.name[:22] if len(p.name) > 22 else p.name
        print(f"{i:<6} {name:<25} {p.status:<12} {p.logit_weight:<12.6f} {p.avg_mase:<12.4f} {p.selection_count:<10}")

    print("\n" + "=" * 80)
    print("✅ 分析完成")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="检查策略 RL 参数 (θ) 分布")
    parser.add_argument('--resume', type=str, default='llog/cs2',
                        help='运行目录（默认 llog/cs2）')
    args = parser.parse_args()

    run_dir = args.resume
    if not os.path.exists(run_dir):
        print(f"❌ 目录不存在: {run_dir}")
        sys.exit(1)

    policies = load_policies(run_dir)
    if not policies:
        print("❌ 没有策略")
        sys.exit(1)

    analyze_rl_params(policies)


if __name__ == '__main__':
    main()