#!/usr/bin/env python
"""
策略状态演变追踪工具
统计从指定轮次开始，每一轮中 ACTIVE、TRIAL、DEPRECATED、ARCHIVE 的策略数量，
并以表格形式展示，便于观察退休机制是否生效。

用法：
    python track_status_evolution.py --resume llog/cs2 --start-round 1

参数：
    --resume       运行目录，默认 llog/cs2
    --start-round  起始轮次（包含），默认 1
    --end-round    结束轮次（包含），默认自动检测最大轮次
    --format       输出格式：table (默认) 或 csv
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Optional
from collections import defaultdict

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from experiments.autotune.skill_policy import SkillPolicy


def load_round_policies(run_dir: str, round_num: int) -> List[SkillPolicy]:
    """加载指定轮次的策略列表（从 refined_policies_optimized.json）"""
    round_dir = os.path.join(run_dir, f"round_{round_num}")
    if not os.path.isdir(round_dir):
        return []

    # 优先使用 optimized，如果不存在则尝试 raw
    for fname in ["refined_policies_optimized.json", "refined_policies_raw.json"]:
        path = os.path.join(round_dir, fname)
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                policies_data = data.get('policies', [])
                return [SkillPolicy.from_dict(p) for p in policies_data]
            except Exception:
                continue
    return []


def count_statuses(policies: List[SkillPolicy]) -> Dict[str, int]:
    """统计策略状态分布"""
    statuses = {
        'ACTIVE': 0,
        'TRIAL': 0,
        'DEPRECATED': 0,
        'ARCHIVE': 0,
        'DELETE': 0
    }
    for p in policies:
        status = p.status
        if status in statuses:
            statuses[status] += 1
        else:
            statuses['ACTIVE'] += 1  # 兜底
    return statuses


def find_max_round(run_dir: str) -> int:
    """自动检测最大轮次"""
    max_round = 0
    for item in os.listdir(run_dir):
        if item.startswith('round_') and os.path.isdir(os.path.join(run_dir, item)):
            try:
                r = int(item.split('_')[1])
                if r > max_round:
                    max_round = r
            except ValueError:
                continue
    return max_round


def main():
    parser = argparse.ArgumentParser(description="策略状态演变追踪工具")
    parser.add_argument('--resume', type=str, default='llog/cs2',
                        help='运行目录（默认 llog/cs2）')
    parser.add_argument('--start-round', type=int, default=1,
                        help='起始轮次（包含），默认 1')
    parser.add_argument('--end-round', type=int, default=None,
                        help='结束轮次（包含），默认自动检测最大轮次')
    parser.add_argument('--format', type=str, choices=['table', 'csv'], default='table',
                        help='输出格式：table 或 csv')

    args = parser.parse_args()

    if not os.path.exists(args.resume):
        print(f"❌ 目录不存在: {args.resume}")
        sys.exit(1)

    max_round = find_max_round(args.resume)
    if max_round == 0:
        print(f"❌ 未找到任何 round_* 目录")
        sys.exit(1)

    end_round = args.end_round if args.end_round is not None else max_round
    start_round = max(1, args.start_round)

    print(f"📊 策略状态演变追踪")
    print(f"📁 运行目录: {args.resume}")
    print(f"🔄 轮次范围: {start_round} ~ {end_round} (检测到最大轮次 {max_round})")
    print("-" * 80)

    # 收集数据
    round_data = []
    for r in range(start_round, end_round + 1):
        policies = load_round_policies(args.resume, r)
        if not policies:
            # 可能该轮不存在，跳过
            continue
        statuses = count_statuses(policies)
        total = sum(statuses.values())
        round_data.append({
            'round': r,
            'total': total,
            'ACTIVE': statuses['ACTIVE'],
            'TRIAL': statuses['TRIAL'],
            'DEPRECATED': statuses['DEPRECATED'],
            'ARCHIVE': statuses['ARCHIVE'],
            'DELETE': statuses['DELETE']
        })

    if not round_data:
        print("❌ 没有找到任何有效轮次的策略文件")
        sys.exit(1)

    # 输出表格
    if args.format == 'csv':
        # CSV 格式
        print("round,total,ACTIVE,TRIAL,DEPRECATED,ARCHIVE,DELETE")
        for d in round_data:
            print(f"{d['round']},{d['total']},{d['ACTIVE']},{d['TRIAL']},{d['DEPRECATED']},{d['ARCHIVE']},{d['DELETE']}")
    else:
        # 表格格式
        header = f"{'轮次':>6} | {'总计':>5} | {'ACTIVE':>8} | {'TRIAL':>8} | {'DEPRECATED':>12} | {'ARCHIVE':>10} | {'DELETE':>8}"
        print(header)
        print("-" * 80)
        for d in round_data:
            print(f"{d['round']:>6} | {d['total']:>5} | {d['ACTIVE']:>8} | {d['TRIAL']:>8} | "
                  f"{d['DEPRECATED']:>12} | {d['ARCHIVE']:>10} | {d['DELETE']:>8}")

    # 简单变化统计
    if len(round_data) >= 2:
        first = round_data[0]
        last = round_data[-1]
        print("\n📈 变化摘要:")
        print(f"   ACTIVE    : {first['ACTIVE']} → {last['ACTIVE']} (Δ{last['ACTIVE'] - first['ACTIVE']:+d})")
        print(f"   TRIAL     : {first['TRIAL']} → {last['TRIAL']} (Δ{last['TRIAL'] - first['TRIAL']:+d})")
        print(f"   DEPRECATED: {first['DEPRECATED']} → {last['DEPRECATED']} (Δ{last['DEPRECATED'] - first['DEPRECATED']:+d})")
        print(f"   ARCHIVE   : {first['ARCHIVE']} → {last['ARCHIVE']} (Δ{last['ARCHIVE'] - first['ARCHIVE']:+d})")
        print(f"   ���计策略数 : {first['total']} → {last['total']} (Δ{last['total'] - first['total']:+d})")


if __name__ == '__main__':
    main()