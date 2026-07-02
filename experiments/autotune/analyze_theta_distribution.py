#!/usr/bin/env python
"""
分析测试集窗口语义匹配策略的 θ 参数分布

功能：
1. 加载指定轮次的策略池
2. 对测试集每个窗口，计算所有策略的语义匹配分数
3. 按语义分数排序，取 top 10%, 20%, 50%, 60% 的策略
4. 统计这些策略的 θ（logit_weight）分布
5. 输出到文件

用法：
    python -m experiments.autotune.analyze_theta_distribution \
        --resume llog/cs2 \
        --round 57 \
        --workers 4

输出：
    llog/cs2/theta_analysis_results/
        theta_distribution.csv          # 汇总统计
        theta_distribution_detailed.csv # 每个窗口的详细数据
"""

import os
import sys
import json
import argparse
import concurrent.futures
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

import pandas as pd
import numpy as np
from tqdm import tqdm

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from experiments.autotune.utils import (
    load_config, load_window_data, extract_features
)
from experiments.autotune.skill_policy import SkillPolicy
from experiments.autotune.state_encoder import StateEncoder
from experiments.autotune.prompts import build_strategy_generation_prompt
from experiments.autotune.inducer_candidate import _safe_extract_json, _extract_strategies_from_text
from src.agents.llm_client import LLMClient
from src.agents.llm_planner import LLMPlannerAgent
from src.tasks.instance import TaskInstance
from run_benchmark import build_full_registry
from run_benchmark import build_full_registry


class ThetaDistributionAnalyzer:
    """分析测试集窗口语义匹配策略的 θ 分布"""

    def __init__(self, run_dir: str, round_num: int, config_path: str = None,
                 test_ratio: float = 0.5, workers: int = 4):
        self.run_dir = run_dir
        self.round_num = round_num
        self.config = load_config(config_path)
        self.output_dir = self.config.get('output_dir', 'storage/autotune_results')
        self.test_workers = workers
        self.test_ratio = test_ratio

        # 百分位配置
        self.percentiles = [0.10, 0.20, 0.50, 0.60]
        self.percentile_labels = ['top10%', 'top20%', 'top50%', 'top60%']

        # 加载策略
        self.policies = self._load_round_policies(round_num)
        if not self.policies:
            print(f"❌ 未找到第 {round_num} 轮策略")
            sys.exit(1)
        print(f"📋 加载第 {round_num} 轮策略: {len(self.policies)} 条")

        # 加载测试集
        self.test_df = self._load_test_df()
        print(f"📊 测试集: {len(self.test_df)} 个窗口")

        # 提取策略 θ 信息
        self._policy_theta_map = {p.policy_id: p.logit_weight for p in self.policies}
        self._policy_name_map = {p.policy_id: p.name for p in self.policies}

        # 创建输出目录
        self.output_dir_path = os.path.join(run_dir, "theta_analysis_results")
        os.makedirs(self.output_dir_path, exist_ok=True)

    def _load_test_df(self) -> pd.DataFrame:
        csv_path = os.path.join(self.output_dir, "collected_windows.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"❌ 未找到采集数据: {csv_path}")

        df = pd.read_csv(csv_path)

        if 'split' in df.columns:
            test_df = df[df['split'] == 'test'].copy()
            if len(test_df) > 0:
                return test_df

        b_mask = df['split'].str.startswith('B') if 'split' in df.columns else pd.Series([True] * len(df))
        if 'split' not in df.columns:
            n = len(df)
            a_end = int(n * 0.5)
            b_mask = pd.Series([False] * n)
            b_mask.iloc[a_end:] = True
            df['split'] = ['A'] * a_end + ['B'] * (n - a_end)

        b_df = df[b_mask].copy().sort_values('window_id').reset_index(drop=True)
        n_b = len(b_df)
        test_size = int(n_b * self.test_ratio)
        test_df = b_df.iloc[:test_size].copy()
        test_df['split'] = 'test'
        return test_df

    def _load_round_policies(self, round_num: int) -> List[SkillPolicy]:
        path = os.path.join(self.run_dir, f"round_{round_num}", "refined_policies_optimized.json")
        if not os.path.exists(path):
            path = os.path.join(self.run_dir, f"round_{round_num}", "refined_policies_raw.json")
        if not os.path.exists(path):
            return []

        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return [SkillPolicy.from_dict(p) for p in data.get('policies', [])]
        except Exception as e:
            print(f"⚠️ 加载失败: {e}")
            return []

    def _process_single_window(self, idx: int, row: pd.Series) -> Optional[Dict]:
        """处理单个窗口，返回语义匹配结果"""
        window_id = row.get('window_id', 'unknown')
        window_data_path = row.get('window_data_path')

        if not window_data_path or not os.path.exists(window_data_path):
            return None

        try:
            wdata = load_window_data(window_data_path)
            train = wdata['train']
            features = extract_features(train)

            # 计算所有策略的语义分数
            scored = []
            for policy in self.policies:
                if policy.status in ['ARCHIVE', 'DELETE']:
                    continue
                score = policy.compute_applicability_score(features)
                scored.append({
                    'policy_id': policy.policy_id,
                    'policy_name': policy.name,
                    'semantic_score': score,
                    'theta': policy.logit_weight,
                    'avg_mase': policy.avg_mase
                })

            # 按语义分数降序排序
            scored.sort(key=lambda x: x['semantic_score'], reverse=True)
            total = len(scored)

            # 计算各百分位的 θ 统计
            result = {
                'window_id': window_id,
                'total_policies': total,
                'percentiles': {}
            }

            for pct, label in zip(self.percentiles, self.percentile_labels):
                k = max(1, int(total * pct))
                top_policies = scored[:k]

                theta_values = [p['theta'] for p in top_policies]
                semantic_values = [p['semantic_score'] for p in top_policies]

                # 记录 top K 的策略详情（用于详细输出）
                top_details = []
                for p in top_policies[:10]:  # 只记录前10个，避免文件过大
                    top_details.append({
                        'policy_id': p['policy_id'],
                        'policy_name': p['policy_name'][:30],
                        'semantic_score': round(p['semantic_score'], 4),
                        'theta': round(p['theta'], 4)
                    })

                result['percentiles'][label] = {
                    'k': k,
                    'theta_min': round(np.min(theta_values), 4),
                    'theta_max': round(np.max(theta_values), 4),
                    'theta_mean': round(np.mean(theta_values), 4),
                    'theta_median': round(np.median(theta_values), 4),
                    'theta_std': round(np.std(theta_values), 4),
                    'semantic_min': round(np.min(semantic_values), 4),
                    'semantic_max': round(np.max(semantic_values), 4),
                    'semantic_mean': round(np.mean(semantic_values), 4),
                    'top_policies': top_details
                }

            return result

        except Exception as e:
            print(f"⚠️ 窗口 {window_id} 处理失败: {e}")
            return None

    def run(self):
        """执行分析"""
        print("\n" + "=" * 80)
        print("📊 测试集窗口语义匹配策略 θ 分布分析")
        print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📁 运行目录: {self.run_dir}")
        print(f"🔢 指定轮次: round_{self.round_num}")
        print(f"📋 策略总数: {len(self.policies)}")
        print(f"📊 测试窗口: {len(self.test_df)}")
        print(f"⚡ 并行线程: {self.test_workers}")
        print("=" * 80)

        # 并行处理所有窗口
        tasks = [(idx, row) for idx, row in self.test_df.iterrows()]
        results = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.test_workers) as executor:
            futures = {}
            for idx, row in tasks:
                future = executor.submit(self._process_single_window, idx, row)
                futures[future] = idx

            pbar = tqdm(total=len(futures), desc="分析进度", unit="窗口", ncols=100)
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result is not None:
                    results.append(result)
                pbar.update(1)
            pbar.close()

        if not results:
            print("❌ 无有效结果")
            return

        print(f"✅ 成功处理 {len(results)} 个窗口")

        # 生成汇总统计
        self._generate_summary(results)
        self._generate_detailed_csv(results)

        print(f"\n📁 结果已保存至: {self.output_dir_path}")
        print(f"   - theta_distribution.csv (汇总统计)")
        print(f"   - theta_distribution_detailed.csv (详细数据)")
        print(f"   - theta_distribution_report.txt (文本报告)")

    def _generate_summary(self, results: List[Dict]):
        """生成汇总统计"""
        # 构建汇总 DataFrame
        rows = []
        for result in results:
            window_id = result['window_id']
            total = result['total_policies']
            for label in self.percentile_labels:
                data = result['percentiles'][label]
                rows.append({
                    'window_id': window_id,
                    'percentile': label,
                    'k': data['k'],
                    'theta_min': data['theta_min'],
                    'theta_max': data['theta_max'],
                    'theta_mean': data['theta_mean'],
                    'theta_median': data['theta_median'],
                    'theta_std': data['theta_std'],
                    'semantic_mean': data['semantic_mean']
                })

        df = pd.DataFrame(rows)

        # 保存 CSV
        csv_path = os.path.join(self.output_dir_path, "theta_distribution.csv")
        df.to_csv(csv_path, index=False)

        # 生成统计报告
        stats = df.groupby('percentile').agg({
            'theta_min': ['min', 'mean', 'max'],
            'theta_mean': ['min', 'mean', 'max'],
            'theta_max': ['min', 'mean', 'max'],
            'theta_median': ['min', 'mean', 'max']
        }).round(4)

        # 生成文本报告
        report_path = os.path.join(self.output_dir_path, "theta_distribution_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("📊 测试集窗口语义匹配策略 θ 分布报告\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"运行目录: {self.run_dir}\n")
            f.write(f"指定轮次: round_{self.round_num}\n")
            f.write(f"策略总数: {len(self.policies)}\n")
            f.write(f"测试窗口: {len(results)}\n")
            f.write("=" * 80 + "\n\n")

            f.write("按百分位汇总统计:\n")
            f.write("-" * 80 + "\n")
            f.write(stats.to_string())
            f.write("\n\n")

            f.write("=" * 80 + "\n")
            f.write("各窗口详细统计:\n")
            f.write("-" * 80 + "\n")
            for result in results[:10]:
                f.write(f"\n窗口 {result['window_id']} (共 {result['total_policies']} 条策略):\n")
                for label in self.percentile_labels:
                    data = result['percentiles'][label]
                    f.write(f"  {label} (K={data['k']}): ")
                    f.write(f"θ范围 [{data['theta_min']:.4f}, {data['theta_max']:.4f}], ")
                    f.write(f"均值 {data['theta_mean']:.4f}, ")
                    f.write(f"中位数 {data['theta_median']:.4f}\n")
                    # 显示 top 策略
                    f.write(f"    Top策略: ")
                    top_str = ", ".join([f"{p['policy_name'][:15]}(θ={p['theta']:.4f})" for p in data['top_policies'][:3]])
                    f.write(top_str + "\n")
            if len(results) > 10:
                f.write(f"\n... 还有 {len(results) - 10} 个窗口\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("✅ 报告生成完成\n")

        print(f"   📄 文本报告: {report_path}")

    def _generate_detailed_csv(self, results: List[Dict]):
        """生成详细 CSV（每个窗口、每个百分位、每个策略的详情）"""
        rows = []
        for result in results:
            window_id = result['window_id']
            for label in self.percentile_labels:
                data = result['percentiles'][label]
                # 提取 top 策略详情（如果有）
                top_policies = data.get('top_policies', [])
                for i, p in enumerate(top_policies):
                    rows.append({
                        'window_id': window_id,
                        'percentile': label,
                        'rank': i + 1,
                        'policy_id': p['policy_id'],
                        'policy_name': p['policy_name'],
                        'semantic_score': p['semantic_score'],
                        'theta': p['theta']
                    })

        if rows:
            df = pd.DataFrame(rows)
            csv_path = os.path.join(self.output_dir_path, "theta_distribution_detailed.csv")
            df.to_csv(csv_path, index=False)


def main():
    parser = argparse.ArgumentParser(description="分析测试集窗口语义匹配策略的 θ 分布")
    parser.add_argument('--resume', type=str, required=True,
                        help='运行目录（如 llog/cs2）')
    parser.add_argument('--round', type=int, required=True,
                        help='指定轮次（如 57）')
    parser.add_argument('--config', type=str, default=None,
                        help='配置文件路径')
    parser.add_argument('--workers', type=int, default=4,
                        help='并行线程数（默认 4）')
    parser.add_argument('--test-ratio', type=float, default=0.5,
                        help='测试集比例（默认 0.5）')

    args = parser.parse_args()

    if os.path.exists(args.resume):
        run_dir = args.resume
    else:
        run_dir = os.path.join("llog", args.resume)

    if not os.path.exists(run_dir):
        print(f"❌ 目录不存在: {run_dir}")
        return

    analyzer = ThetaDistributionAnalyzer(
        run_dir=run_dir,
        round_num=args.round,
        config_path=args.config,
        test_ratio=args.test_ratio,
        workers=args.workers
    )
    analyzer.run()


if __name__ == '__main__':
    main()