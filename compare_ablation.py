# compare_ablation.py
"""
Policy Ablation Study

替代原有的三模式对比
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.agents.llm_planner import LLMPlannerAgent
from src.tasks.instance import TaskInstance
from experiments.autotune.utils import load_window_data, compute_mase, extract_features
from experiments.autotune.skill_policy import SkillPolicy, create_policy_from_legacy_rule
from run_benchmark import build_full_registry


def main():
    parser = argparse.ArgumentParser(description="Policy Ablation Study")
    parser.add_argument('--dataset', type=str, default='melbourne_temp')
    parser.add_argument('--horizon', type=int, default=7)
    parser.add_argument('--output', type=str, default='storage/ablation_results.csv')
    args = parser.parse_args()

    print(f"📊 Policy Ablation Study: {args.dataset}")
    print("=" * 60)

    csv_path = f"storage/autotune_results/collected_windows.csv"
    if not os.path.exists(csv_path):
        print(f"❌ 未找到采集数据: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    test_df = df[df['split'] == 'test']

    if len(test_df) == 0:
        print("❌ 没有测试集窗口")
        return

    print(f"📋 测试集窗口数: {len(test_df)}")

    # 加载策略
    policies_file = "storage/autotune_results/refined_policies.json"
    if os.path.exists(policies_file):
        with open(policies_file, 'r') as f:
            data = json.load(f)
            policies = [SkillPolicy.from_dict(p) for p in data.get('policies', [])]
    else:
        # 尝试从旧规则加载
        rules_file = "storage/autotune_results/generated_rules.json"
        if os.path.exists(rules_file):
            with open(rules_file, 'r') as f:
                data = json.load(f)
                legacy_rules = data.get('rules', [])
                policies = [create_policy_from_legacy_rule(r) for r in legacy_rules]
        else:
            policies = []

    print(f"📋 策略数: {len(policies)}")

    # 构建 Agent
    full_registry, _ = build_full_registry()

    agent_no_policy = LLMPlannerAgent(
        model="glm-4",
        skill_registry=full_registry,
        log_file=None,
        use_skills=True,
        rules_file=None
    )

    results = []

    for _, row in test_df.iterrows():
        window_id = row['window_id']
        window_data_path = row['window_data_path']

        if not window_data_path or not os.path.exists(window_data_path):
            continue

        wdata = load_window_data(window_data_path)
        train = wdata['train']
        test = wdata['test']
        period = wdata.get('period', 365)
        mase_scale = wdata.get('mase_scale', 1.0)
        horizon = wdata.get('horizon', args.horizon)

        print(f"🔄 处理窗口 {window_id}...")

        # 无策略
        try:
            task = TaskInstance(
                id=f"ablation_no_{window_id}",
                dataset_id=args.dataset,
                template_id="fixed_origin",
                question="",
                question_type="numerical",
                history=train.tolist(),
                horizon=horizon,
                frequency="daily",
                prediction_target={},
                resolution_date=datetime.now(),
                difficulty_level=1,
                ground_truth_extractor="",
                dates=None,
                target_date=""
            )
            pred_no = agent_no_policy.predict(task)
            pred_no = np.array(pred_no)
            mase_no = compute_mase(pred_no, test, mase_scale) if len(pred_no) == len(test) else np.nan
        except:
            mase_no = np.nan

        # 有策略（检索）
        if policies:
            try:
                features = extract_features(train)
                matched_policy = None
                for policy in policies:
                    if policy.is_applicable(features):
                        matched_policy = policy
                        break

                if matched_policy:
                    pred_policy = matched_policy.execute(train, horizon, period)
                    pred_policy = np.array(pred_policy)
                    mase_policy = compute_mase(pred_policy, test, mase_scale) if len(pred_policy) == len(
                        test) else np.nan
                else:
                    mase_policy = np.nan
            except:
                mase_policy = np.nan
        else:
            mase_policy = np.nan

        results.append({
            'window_id': window_id,
            'origin': row.get('origin', 0),
            'mase_no_policy': mase_no,
            'mase_with_policy': mase_policy,
            'improvement': (mase_no - mase_policy) / mase_no * 100 if not np.isnan(mase_no) and not np.isnan(
                mase_policy) and mase_no > 0 else np.nan
        })

    # 汇总
    if results:
        df_results = pd.DataFrame(results)
        valid = df_results.dropna()

        if len(valid) > 0:
            avg_no = valid['mase_no_policy'].mean()
            avg_policy = valid['mase_with_policy'].mean()
            avg_improvement = (avg_no - avg_policy) / avg_no * 100 if avg_no > 0 else 0

            print("\n" + "=" * 60)
            print("📊 Ablation 结果")
            print("=" * 60)
            print(f"   无策略 MASE: {avg_no:.4f} (基于 {len(valid)} 个窗口)")
            print(f"   有策略 MASE: {avg_policy:.4f}")
            print(f"   改善: {avg_improvement:.2f}%")

            valid.to_csv(args.output, index=False)
            print(f"\n📁 结果已保存: {args.output}")
        else:
            print("⚠️ 无有效结果")
    else:
        print("⚠️ 无结果")


if __name__ == '__main__':
    main()