#!/usr/bin/env python
"""
轮次策略状态趋势分析工具
统计从指定轮次开始，每轮策略池中各状态（ACTIVE/TRIAL/DEPRECATED/ARCHIVE）的数量变化，
以表格形式展示，便于监控退休机制是否生效。

用法：
    python -m experiments.autotune.round_status_trend --resume llog/cs2 --start-round 1

功能：
    1. 扫描 run_dir 下所有 round_*/refined_policies_optimized.json
    2. 提取每轮策略的状态分布、总数量、平均MASE
    3. 以表格展示（支持终端彩色输出）
    4. 可选保存为 CSV 文件
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Optional
from collections import defaultdict
import pandas as pd

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from experiments.autotune.skill_policy import SkillPolicy
from experiments.autotune.checkpoint_manager import CheckpointManager


def load_round_policies(run_dir: str, round_num: int) -> Optional[List[SkillPolicy]]:
    """加载指定轮次的策略列表（从 optimized 文件）"""
    round_dir = os.path.join(run_dir, f"round_{round_num}")
    policy_file = os.path.join(round_dir, "refined_policies_optimized.json")
    if not os.path.exists(policy_file):
        # 尝试旧格式 raw
        policy_file = os.path.join(round_dir, "refined_policies_raw.json")
        if not os.path.exists(policy_file):
            return None

    try:
        with open(policy_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        policies_data = data.get('policies', [])
        if not policies_data:
            return None
        return [SkillPolicy.from_dict(p) for p in policies_data]
    except Exception as e:
        print(f"   ⚠️ 加载第 {round_num} 轮失败: {e}")
        return None


def analyze_round(policies: List[SkillPolicy]) -> Dict:
    """分析单轮策略，返回状态分布和统计信息"""
    status_count = defaultdict(int)
    total_mase = 0.0
    valid_mase_count = 0

    for p in policies:
        status_count[p.status] += 1
        if p.avg_mase is not None and p.avg_mase != float('inf'):
            total_mase += p.avg_mase
            valid_mase_count += 1

    avg_mase = total_mase / valid_mase_count if valid_mase_count > 0 else float('nan')

    return {
        'total': len(policies),
        'ACTIVE': status_count.get('ACTIVE', 0),
        'TRIAL': status_count.get('TRIAL', 0),
        'DEPRECATED': status_count.get('DEPRECATED', 0),
        'ARCHIVE': status_count.get('ARCHIVE', 0),
        'DELETE': status_count.get('DELETE', 0),
        'avg_mase': avg_mase
    }


def scan_rounds(run_dir: str, start_round: int = 1) -> Dict[int, Dict]:
    """扫描所有轮次，返回 round_num -> 统计信息"""
    results = {}
    round_num = start_round
    while True:
        policies = load_round_policies(run_dir, round_num)
        if policies is None:
            # 如果当前轮次不存在，检查是否已经到达末尾（下一轮也不存在则停止）
            next_round = round_num + 1
            next_policies = load_round_policies(run_dir, next_round)
            if next_policies is None:
                break
            # 如果当前轮不存在但下一轮存在，说明轮次不连续，跳过当前
            round_num = next_round
            continue
        stats = analyze_round(policies)
        results[round_num] = stats
        round_num += 1
    return results


def print_table(results: Dict[int, Dict], start_round: int, run_dir: str):
    """打印对比表格"""
    if not results:
        print("❌ 没有找到任何轮次数据")
        return

    # 确定列
    columns = ['轮次', '总数', 'ACTIVE', 'TRIAL', 'DEPRECATED', 'ARCHIVE', 'DELETE', '平均MASE']
    # 收集数据
    data = []
    for round_num, stats in sorted(results.items()):
        if round_num < start_round:
            continue
        data.append([
            f"R{round_num}",
            stats['total'],
            stats['ACTIVE'],
            stats['TRIAL'],
            stats['DEPRECATED'],
            stats['ARCHIVE'],
            stats['DELETE'],
            f"{stats['avg_mase']:.4f}" if not pd.isna(stats['avg_mase']) else 'N/A'
        ])

    if not data:
        print("❌ 没有符合条件的轮次数据")
        return

    # 使用 pandas 打印表格
    df = pd.DataFrame(data, columns=columns)
    print("\n" + "=" * 100)
    print(f"📊 策略状态趋势（从第 {start_round} 轮开始）")
    print("=" * 100)
    print(df.to_string(index=False))
    print("=" * 100)

    # 计算趋势
    if len(data) >= 2:
        # 提取各列数值（索引：0-轮次, 1-总数, 2-ACTIVE, 3-TRIAL, 4-DEPRECATED, 5-ARCHIVE, 6-DELETE, 7-平均MASE）
        total_trend = [int(row[1]) for row in data]
        active_trend = [int(row[2]) for row in data]
        trial_trend = [int(row[3]) for row in data]
        deprecated_trend = [int(row[4]) for row in data]
        archive_trend = [int(row[5]) for row in data]
        retired_trend = [deprecated_trend[i] + archive_trend[i] for i in range(len(data))]

        first_total = total_trend[0]
        last_total = total_trend[-1]
        first_active = active_trend[0]
        last_active = active_trend[-1]
        first_retired = retired_trend[0]
        last_retired = retired_trend[-1]
        first_trial = trial_trend[0]
        last_trial = trial_trend[-1]

        print("\n📈 趋势分析:")
        if last_retired > first_retired:
            print(f"   ✅ 退休策略（DEPRECATED+ARCHIVE）从 {first_retired} 增加到 {last_retired}（+{last_retired - first_retired}），退休机制正在生效")
        elif last_retired == first_retired:
            if last_retired == 0:
                print(f"   ⚠️ 退休策略（DEPRECATED+ARCHIVE）始终为 0，退休机制可能未生效")
            else:
                print(f"   ℹ️ 退休策略（DEPRECATED+ARCHIVE）保持 {first_retired}，没有变化")
        else:
            print(f"   ⚠️ 退休策略（DEPRECATED+ARCHIVE）从 {first_retired} 减少到 {last_retired}（{last_retired - first_retired}），可能是归档后删除")

        if last_active < first_active:
            print(f"   ✅ ACTIVE 策略从 {first_active} 减少到 {last_active}，策略池正在净化")
        elif last_active > first_active:
            print(f"   ⚠️ ACTIVE 策略从 {first_active} 增加到 {last_active}，可能需要更激进的退休")
        else:
            print(f"   ℹ️ ACTIVE 策略保持 {first_active}，没有变化")

        if last_trial > first_trial:
            print(f"   ℹ️ TRIAL 策略从 {first_trial} 增加到 {last_trial}，Re-Induction 仍在添加新策略")
        elif last_trial < first_trial:
            print(f"   ℹ️ TRIAL 策略从 {first_trial} 减少到 {last_trial}，TRIAL 策略正在被评估或退休")
        else:
            print(f"   ℹ️ TRIAL 策略保持 {first_trial}，没有变化")

        if last_total > first_total:
            print(f"   ℹ️ 策略池总规模从 {first_total} 增加到 {last_total}（+{last_total - first_total}），注意控制膨胀")
        elif last_total < first_total:
            print(f"   ✅ 策略池总规模从 {first_total} 减少到 {last_total}（{last_total - first_total}），池子正在精简")
        else:
            print(f"   ℹ️ 策略池总规模保持 {first_total}，没有变化")

    print("\n💡 建议：如果 DEPRECATED 数量持续增长且 ACTIVE 减少，则退休机制健康。")

    # 保存 CSV
    csv_path = os.path.join(run_dir, "status_trend.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n📁 表格已保存到: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="轮次策略状态趋势分析")
    parser.add_argument('--resume', type=str, default='llog/cs2',
                        help='运行目录（默认 llog/cs2）')
    parser.add_argument('--start-round', type=int, default=1,
                        help='起始轮次（默认 1）')
    args = parser.parse_args()

    run_dir = args.resume
    if not os.path.exists(run_dir):
        print(f"❌ 目录不存在: {run_dir}")
        sys.exit(1)

    results = scan_rounds(run_dir, args.start_round)
    if not results:
        print(f"❌ 未找到任何轮次文件（从 round_{args.start_round} 开始）")
        sys.exit(1)

    print_table(results, args.start_round, run_dir)


if __name__ == '__main__':
    main()