# experiments/autotune/print_round_stats.py
"""
打印各 round_* 文件夹中策略状态的分布（ACTIVE, TRIAL, DEPRECATED, ARCHIVE）
用法: python -m experiments.autotune.print_round_stats --run_dir llog/cs2
"""

import os
import sys
import json
import argparse
from collections import Counter

def main():
    parser = argparse.ArgumentParser(description="统计各轮策略状态分布")
    parser.add_argument('--run_dir', required=True, help="运行目录，如 llog/cs2")
    args = parser.parse_args()

    run_dir = args.run_dir
    if not os.path.exists(run_dir):
        print(f"错误：目录 {run_dir} 不存在")
        sys.exit(1)

    # 查找所有 round_* 文件夹
    round_dirs = [d for d in os.listdir(run_dir) if d.startswith('round_') and os.path.isdir(os.path.join(run_dir, d))]
    round_dirs.sort(key=lambda x: int(x.split('_')[1]))

    print("\n" + "=" * 80)
    print("各轮策略状态分布统计")
    print(f"运行目录: {run_dir}")
    print("=" * 80)

    # 表头
    header = f"{'Round':<8} {'ACTIVE':>8} {'TRIAL':>8} {'DEPRECATED':>12} {'ARCHIVE':>10} {'合计':>8}"
    print(header)
    print("-" * 80)

    for rd in round_dirs:
        file_path = os.path.join(run_dir, rd, "refined_policies_optimized.json")
        if not os.path.exists(file_path):
            # 尝试 raw 文件
            file_path = os.path.join(run_dir, rd, "refined_policies_raw.json")
        if not os.path.exists(file_path):
            print(f"{rd:<8} {'(文件缺失)':<50}")
            continue

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # 数据可能是列表或对象数组
            if isinstance(data, list):
                policies = data
            elif isinstance(data, dict) and 'policies' in data:
                policies = data['policies']
            else:
                policies = []
            
            statuses = [p.get('status', 'UNKNOWN') for p in policies]
            counter = Counter(statuses)
            active = counter.get('ACTIVE', 0)
            trial = counter.get('TRIAL', 0)
            deprecated = counter.get('DEPRECATED', 0)
            archive = counter.get('ARCHIVE', 0)
            total = len(policies)
            print(f"{rd:<8} {active:>8} {trial:>8} {deprecated:>12} {archive:>10} {total:>8}")
        except Exception as e:
            print(f"{rd:<8} (读取失败: {e})")

    print("=" * 80)
    print("统计完成。")

if __name__ == "__main__":
    main()