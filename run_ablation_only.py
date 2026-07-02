#python run_ablation_only.py --resume run_20260625_181236


#!/usr/bin/env python
"""
独立 Ablation 测试 - 直接复用 tuner_eval.run_ablation_parallel
"""

import os
import sys
import json
import argparse
import pandas as pd
from datetime import datetime

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from experiments.autotune.skill_policy import SkillPolicy
from experiments.autotune.spls_loop import SPLSLoop
from experiments.autotune.tuner_eval import run_ablation_parallel, print_ablation_summary
from experiments.autotune.utils import ProgressLogger, load_config, generate_comparison_report
from src.agents.llm_client import LLMClient
from run_benchmark import build_full_registry


def main():
    parser = argparse.ArgumentParser(description="独立 Ablation 测试")
    parser.add_argument('--resume', type=str, required=True,
                        help='运行目录（如 run_20260625_181236）')
    parser.add_argument('--config', type=str, default=None,
                        help='配置文件路径')
    parser.add_argument('--workers', type=int, default=10,
                        help='并行线程数（默认 10）')
    args = parser.parse_args()

    if os.path.exists(args.resume):
        run_dir = args.resume
    else:
        run_dir = os.path.join("llog", args.resume)

    if not os.path.exists(run_dir):
        print(f"❌ 目录不存在: {run_dir}")
        return

    logger = ProgressLogger(log_dir=run_dir, verbose=True, run_folder=False)
    logger.start_log("ablation_only")
    config = load_config(args.config) if args.config else {}

    logger.log("=" * 70)
    logger.log("📊 独立 Ablation 测试（复用主程序引擎）")
    logger.log(f"📁 运行目录: {run_dir}")
    logger.log("=" * 70)

    # 加载测试集
    csv_path = "storage/autotune_results/collected_windows.csv"
    if not os.path.exists(csv_path):
        logger.log(f"❌ 未找到采集数据: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    test_df = df[df['split'] == 'test'].copy()
    logger.log(f"📊 测试集: {len(test_df)} 个窗口")

    # 加载各轮策略
    policy_snapshots = {}
    policy_snapshots['no_rule'] = []

    for round_num in range(1, 9):
        round_dir = os.path.join(run_dir, f"round_{round_num}")
        json_path = os.path.join(round_dir, "refined_policies_optimized.json")
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                policies = [SkillPolicy.from_dict(p) for p in data.get('policies', [])]
                policy_snapshots[f'round_{round_num}'] = policies
                logger.log(f"   round_{round_num}: {len(policies)} 条策略")
            except Exception as e:
                logger.log(f"   ⚠️ 加载 round_{round_num} 失败: {e}")

    # 加载最终策略
    json_path = os.path.join(run_dir, "refined_policies.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            policies = [SkillPolicy.from_dict(p) for p in data.get('policies', [])]
            policy_snapshots['final_with_patch'] = policies
            policy_snapshots['final'] = policies
            logger.log(f"   final_with_patch: {len(policies)} 条策略")
        except Exception as e:
            logger.log(f"   ⚠️ 加载 final 策略失败: {e}")

    # 创建 SPLSLoop
    loop = SPLSLoop(config, logger)

    # 执行 Ablation
    ablation_results = run_ablation_parallel(
        logger=logger,
        loop=loop,
        policy_snapshots=policy_snapshots,
        num_rounds=8,
        dataset_name="melbourne_temp",
        window_size=600,
        horizon=12,
        test_df=test_df,
        test_parallel=True,
        test_workers=args.workers
    )

    # 生成报告
    comparison_report = generate_comparison_report(ablation_results, run_dir)
    logger.log(f"\n📁 对比报告已保存: {comparison_report}")

    print_ablation_summary(logger, ablation_results)

    stats_str = LLMClient.print_token_stats("Ablation Token 统计")
    logger.log("\n" + stats_str)
    logger.log("✅ Ablation 完成！")


if __name__ == '__main__':
    main()

