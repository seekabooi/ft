#!/usr/bin/env python
"""
策略诊断脚本 - 打印所有策略的详细指标
用于分析为什么没有 DEPRECATED / ARCHIVE 策略

用法：
    python diagnose_policies.py

功能：
    1. 加载 llog/cs2/checkpoint.json
    2. 打印每个策略的所有关键指标
    3. 计算 retire_score 并显示是否达到退休阈值
    4. 高亮显示可能应该退休但未被标记的策略
"""

import os
import sys
import json
import pickle
from typing import Dict, List, Optional
import numpy as np

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from experiments.autotune.skill_policy import SkillPolicy
from experiments.autotune.checkpoint_manager import CheckpointManager


def load_policies(run_dir: str = "llog/cs2") -> List[SkillPolicy]:
    """加载策略列表"""
    checkpoint_path = os.path.join(run_dir, "checkpoint.json")
    if not os.path.exists(checkpoint_path):
        print(f"❌ 检查点不存在: {checkpoint_path}")
        sys.exit(1)

    with open(checkpoint_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    policies_data = data.get('current_policies_data', [])
    policies = [SkillPolicy.from_dict(p) for p in policies_data]
    print(f"📋 加载 {len(policies)} 条策略\n")
    return policies


def compute_retire_score(policy: SkillPolicy, weights: Dict = None) -> float:
    """计算退休分数（与 retirement_mechanism.py 逻辑一致）"""
    if weights is None:
        weights = {
            'utility': 0.3,
            'coverage': 0.2,
            'rare_score': 0.2,
            'uniqueness': 0.1,
            'marginal_value': 0.2
        }

    utility_term = weights['utility'] * (1.0 - max(0, min(1.0, policy.utility_ema)))
    coverage_term = weights['coverage'] * (1.0 - max(0, min(1.0, policy.coverage_rate)))
    rare_term = weights['rare_score'] * (1.0 - max(0, min(1.0, policy.rare_score)))
    uniqueness_term = weights['uniqueness'] * (1.0 - max(0, min(1.0, policy.uniqueness)))
    marginal_term = weights['marginal_value'] * (1.0 - max(0, min(1.0, policy.marginal_value)))

    score = utility_term + coverage_term + rare_term + uniqueness_term + marginal_term
    return min(1.0, max(0.0, score))


def is_policy_frozen(policy: SkillPolicy, current_round: int = 12) -> bool:
    """判断策略是否在冻结期"""
    if policy.status != 'TRIAL':
        return False
    trial_start = policy.metadata.get('trial_start_round', 0)
    trial_freeze = policy.metadata.get('trial_freeze_rounds', 2)
    return current_round - trial_start < trial_freeze


def analyze_policies(policies: List[SkillPolicy], current_round: int = 12):
    """分析所有策略"""
    print("=" * 120)
    print("📊 策略诊断报告")
    print(f"当前轮次: {current_round}")
    print("=" * 120)

    # 统计
    status_count = {}
    frozen_count = 0
    retire_candidates = []
    total_retire_score = 0.0

    # 打印表头
    print(f"\n{'ID':<10} {'名称':<20} {'状态':<12} {'冻结?':<6} {'avg_mase':<10} {'utility':<8} "
          f"{'coverage':<8} {'rare':<8} {'uniqueness':<8} {'marginal':<8} {'retire_score':<12} {'应该退休?':<8}")
    print("-" * 130)

    for policy in policies:
        # 统计状态
        status = policy.status
        status_count[status] = status_count.get(status, 0) + 1

        # 计算冻结状态
        frozen = is_policy_frozen(policy, current_round) if status == 'TRIAL' else False
        if frozen:
            frozen_count += 1

        # 计算退休分数
        retire_score = compute_retire_score(policy)
        total_retire_score += retire_score

        # 判断是否应该退休（threshold = 0.6）
        should_retire = retire_score >= 0.6 and not frozen
        if should_retire:
            retire_candidates.append(policy)

        # 截断名称
        name_display = policy.name[:18] if len(policy.name) > 18 else policy.name

        # 标记应该退休的策略
        flag = "🔴" if should_retire else ""

        # 打印行
        print(f"{policy.policy_id[:8]:<10} {name_display:<20} {status:<12} {str(frozen):<6} "
              f"{policy.avg_mase:<10.4f} {policy.utility_ema:<8.4f} {policy.coverage_rate:<8.4f} "
              f"{policy.rare_score:<8.4f} {policy.uniqueness:<8.4f} {policy.marginal_value:<8.4f} "
              f"{retire_score:<12.4f} {flag:<8}")

    # 打印摘要
    print("\n" + "=" * 120)
    print("📊 摘要统计")
    print("=" * 120)

    print(f"  总策略数: {len(policies)}")
    print(f"  状态分布: {status_count}")
    print(f"  冻结期策略: {frozen_count} 条 (TRIAL 且在冻结期内)")
    print(f"  平均 retire_score: {total_retire_score / len(policies):.4f}")

    # 退休候选
    if retire_candidates:
        print(f"\n  🔴 应该退休的策略 ({len(retire_candidates)} 条):")
        for p in retire_candidates:
            score = compute_retire_score(p)
            print(f"      - {p.name} (MASE={p.avg_mase:.4f}, retire_score={score:.4f})")
    else:
        print("\n  ✅ 没有策略达到退休阈值 (threshold=0.6)")

    # 分析退休机制未触发的原因
    print("\n" + "=" * 120)
    print("🔍 退休机制未触发的原因分析")
    print("=" * 120)

    # 检查每个策略的 retire_score 低的原因
    reasons = {
        'utility_ema 太高': 0,
        'coverage_rate 太高': 0,
        'marginal_value 太高': 0,
        'frozen': 0,
        'status 不是 ACTIVE/DEPRECATED': 0
    }

    for p in policies:
        if p.status not in ['ACTIVE', 'DEPRECATED']:
            reasons['status 不是 ACTIVE/DEPRECATED'] += 1
            continue

        if is_policy_frozen(p, current_round):
            reasons['frozen'] += 1
            continue

        score = compute_retire_score(p)
        if score < 0.6:
            # 检查哪个指标贡献低
            utility_term = 0.3 * (1.0 - p.utility_ema)
            coverage_term = 0.2 * (1.0 - p.coverage_rate)
            marginal_term = 0.2 * (1.0 - p.marginal_value)

            if utility_term < 0.1:
                reasons['utility_ema 太高'] += 1
            elif coverage_term < 0.1:
                reasons['coverage_rate 太高'] += 1
            elif marginal_term < 0.1:
                reasons['marginal_value 太高'] += 1

    for reason, count in reasons.items():
        if count > 0:
            print(f"  - {reason}: {count} 条策略")

    # 阈值问题
    print(f"\n  - 退休阈值: 0.6 (可修改为 0.4 以加速退休)")
    print(f"  - 冻结期: 2 轮 (TRIAL 策略在冻结期内不会被退休)")

    # 按 retire_score 排序，显示 TOP 10
    sorted_policies = sorted(policies, key=lambda p: compute_retire_score(p), reverse=True)
    print("\n" + "=" * 120)
    print("📋 retire_score TOP 10 (最应该退休的策略)")
    print("=" * 120)
    print(f"{'排名':<6} {'名称':<20} {'状态':<12} {'avg_mase':<10} {'retire_score':<12} {'冻结?':<6}")
    print("-" * 80)

    for i, p in enumerate(sorted_policies[:10]):
        score = compute_retire_score(p)
        frozen = is_policy_frozen(p, current_round) if p.status == 'TRIAL' else False
        name_display = p.name[:18] if len(p.name) > 18 else p.name
        flag = "🔴" if (score >= 0.6 and not frozen) else ""
        print(f"{i+1:<6} {name_display:<20} {p.status:<12} {p.avg_mase:<10.4f} {score:<12.4f} {str(frozen):<6} {flag}")

    print("\n" + "=" * 120)
    print("✅ 诊断完成")
    print("=" * 120)


def main():
    # 解析命令行参数
    import argparse
    parser = argparse.ArgumentParser(description="策略诊断工具")
    parser.add_argument('--resume', type=str, default='llog/cs2',
                        help='运行目录（默认 llog/cs2）')
    parser.add_argument('--round', type=int, default=12,
                        help='当前轮次（用于判断冻结期）')
    args = parser.parse_args()

    # 加载策略
    policies = load_policies(args.resume)

    if not policies:
        print("❌ 没有策略")
        sys.exit(1)

    # 分析
    analyze_policies(policies, args.round)


if __name__ == '__main__':
    main()