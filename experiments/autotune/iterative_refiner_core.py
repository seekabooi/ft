# experiments/autotune/iterative_refiner_core.py
"""
Policy Evolution Engine - run_round 核心演化循环
★ v6 强化学习版：禁用旧的 Merge/Refresh/Patch 机制
★ 策略演化由 PolicyDistributionModel 的 Policy Gradient 驱动
★ 保留 Re-Induction 用于生成新策略（由 RL 采样选择）
★ ★ 2026-08-XX 退休机制已彻底禁用，相关代码已注释
"""

import os
import sys
import json
import hashlib
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime
import time
from tqdm import tqdm

from experiments.autotune.skill_policy import SkillPolicy
from experiments.autotune.iterative_refiner_base import PolicyEvolutionEngine
from experiments.autotune.iterative_refiner_utils import identify_hard_windows_with_mases


def run_round_impl(self: PolicyEvolutionEngine, policies: List[SkillPolicy],
                   val_df: pd.DataFrame, round_num: int = 1, force: bool = False) -> Dict:
    """
    run_round 核心实现 - v6 强化学习版（退休已禁用）
    """
    self._round = round_num
    changes = []
    new_policies_added = 0
    before_count = len(policies)

    policy_ids = sorted([p.policy_id for p in policies])
    current_hash = hashlib.md5(''.join(policy_ids).encode()).hexdigest()
    self._last_policies_hash = current_hash
    self._last_val_df_id = id(val_df)

    # ★★★ 轮次开始分隔 ★★★
    self.logger.log("")
    self.logger.log("┌" + "─" * 58 + "┐")
    self.logger.log(f"│  🔄 第 {round_num} 轮演化优化（RL 模式）{' ' * (38 - len(str(round_num)))}│")
    self.logger.log("├" + "─" * 58 + "┤")
    self.logger.log(f"│  📋 当前策略数: {len(policies)} {' ' * (46 - len(str(len(policies))))}│")
    self.logger.log("└" + "─" * 58 + "┘")

    if len(policies) < 2:
        self.logger.log(f"   ⚠️ 策略数量 ({len(policies)}) 不足，跳过演化")
        return {'policies': policies, 'changes': changes, 'new_policies_added': 0}

    # ★★★ Patch 1：统计 stability_score 分布 ★★★
    stability_scores = []
    for p in policies:
        if hasattr(p, 'get_stability_score'):
            stability_scores.append(p.get_stability_score())
        else:
            stability_scores.append(0.5)

    if stability_scores:
        self.logger.log(f"   📊 stability_score 分布: "
                        f"min={min(stability_scores):.3f}, "
                        f"mean={np.mean(stability_scores):.3f}, "
                        f"max={max(stability_scores):.3f}")
        low_stability = [s for s in stability_scores if s < 0.3]
        if low_stability:
            self.logger.log(f"   ⚠️ 低稳定性策略 (<0.3): {len(low_stability)} 条")

    health_report = self.health_monitor.get_health_report(policies)
    self.logger.log(f"   📊 健康评分: {health_report.get('average_health_score', 0):.3f}")

    evolution_strength = self.config.get('rl', {}).get('evolution_effect_strength', 0.30)
    self.logger.log(f"   ⚡ Evolution Strength: {evolution_strength:.2f} (不稳定策略 → 更高概率被扰动)")

    if force:
        self.logger.log(f"   ⚡ 强制触发演化（RL 模式）")
    else:
        self.logger.log(f"   ℹ️ RL 模式：演化由 Policy Gradient 驱动，无需外部触发")

    # ★★★ 生成全局摘要 ★★★
    global_summary = self._build_global_summary(policies)
    global_summary_str = self._format_global_summary(global_summary)

    self.logger.log("\n" + "=" * 50)
    self.logger.log("📊 全局策略池摘要（RL 参考信息）")
    self.logger.log("=" * 50)
    self.logger.log(global_summary_str)
    self.logger.log("=" * 50)

    # ==================== ★★★ 退休机制 - 已禁用 ★★★ ====================
    self.logger.log("")
    self.logger.log("   ┌─────────────────────────────────────────────────────────────┐")
    self.logger.log("   │  📍 子阶段 0: Retire（退休）— 已禁用                     │")
    self.logger.log("   └─────────────────────────────────────────────────────────────┘")
    self.logger.log("   ℹ️ 退休机制已彻底关闭，策略将永远不会被标记为 DEPRECATED/ARCHIVE/DELETE")
    retired_count = 0  # 固定为 0

    # ==================== Re-Induction ====================
    self.logger.log("")
    self.logger.log("   ┌─────────────────────────────────────────────────────────────┐")
    self.logger.log("   │  📍 子阶段 1/2: Re-Induction 检查                          │")
    self.logger.log("   └─────────────────────────────────────────────────────────────┘")

    reinduction_added = 0
    if self.reinduction_enabled and self.inducer is not None:
        self.logger.log("   🔍 检查 Re-Induction 条件...")
        reinduction_added = self._reinduction_rl(policies, val_df, round_num)

    new_policies_added = reinduction_added

    after_count = len(policies)
    self.logger.log("")
    self.logger.log("   ┌─────────────────────────────────────────────────────────────┐")
    self.logger.log("   │  📍 子阶段 2/2: 演化完成总结                              │")
    self.logger.log("   └─────────────────────────────────────────────────────────────┘")
    self.logger.log(f"   ✅ 第 {round_num} 轮演化完成")
    self.logger.log(f"      📋 策略数: {before_count} → {after_count} ({after_count - before_count:+.0f})")
    if new_policies_added > 0:
        self.logger.log(f"      ★ 新增 {new_policies_added} 条策略（已加入分布模型，logit_weight=0.0）")
        for p in policies:
            if p.policy_id not in [pp.policy_id for pp in policies[:before_count]]:
                self.logger.log(f"         - {p.name} (cluster: {p.cluster_id}, MASE: {p.avg_mase:.4f})")

    # ★★★ 状态分布摘要（仅显示 ACTIVE/TRIAL，因 DEPRECATED 等不存在） ★★★
    active_count = sum(1 for p in policies if p.status == 'ACTIVE')
    trial_count = sum(1 for p in policies if p.status == 'TRIAL')
    self.logger.log(f"      📊 状态分布: ACTIVE={active_count}, TRIAL={trial_count}")

    return {
        'policies': policies,
        'changes': changes,
        'policy_count': len(policies),
        'new_policies_added': new_policies_added,
        'retired_count': 0  # 始终为 0
    }


# ★★★ RL 版本的 Re-Induction（含质量门控 + 试用期记录 + 概率补偿） ★★★
def reinduction_rl_impl(self: PolicyEvolutionEngine, policies: List[SkillPolicy],
                        val_df: pd.DataFrame, round_num: int) -> int:
    """
    RL 版本的 Re-Induction - 保持不变
    """
    self.logger.log("   🆕 检查 Re-Induction 条件（RL 模式）...")

    if len(val_df) < self.min_hard_windows:
        self.logger.log(f"   ℹ️ 验证集窗口数 ({len(val_df)}) < {self.min_hard_windows}，跳过")
        return 0

    # 识别困难窗口
    self.logger.log("   📊 识别困难窗口...")
    hard_windows, hard_mases, hard_indices = identify_hard_windows_with_mases(
        policies, val_df, self._hard_window_cache, self._policy_window_cache,
        self._cache_max_size, self.config, self.logger, self.hard_window_multiplier
    )

    hard_ratio = len(hard_windows) / len(val_df)

    self.logger.log(f"   📊 困难窗口: {len(hard_windows)}/{len(val_df)} ({hard_ratio:.2%})")

    if len(hard_windows) < self.min_hard_windows:
        self.logger.log(f"   ℹ️ 困难窗口数 ({len(hard_windows)}) < {self.min_hard_windows}，不触发")
        return 0

    if hard_ratio < self.hard_window_ratio_threshold:
        self.logger.log(f"   ℹ️ 困难窗口比例 ({hard_ratio:.2%}) < {self.hard_window_ratio_threshold:.2%}，不触发")
        return 0

    self.logger.log(
        f"   ✅ 触发 Re-Induction（困难窗口比例 {hard_ratio:.2%} >= {self.hard_window_ratio_threshold:.2%}）")

    if hard_indices and len(hard_indices) <= 20:
        self.logger.log(f"      📋 困难窗口 ID: {sorted(hard_indices)}")
        self.logger.log(f"      📊 MASE 范围: {min(hard_mases):.4f} ~ {max(hard_mases):.4f}")
    else:
        self.logger.log(f"      📋 困难窗口数: {len(hard_indices)} 个")
        self.logger.log(f"      📊 平均 MASE: {np.mean(hard_mases):.4f}")

    # 收集困难窗口到全局池
    collected_count = 0
    for idx, window_idx in enumerate(hard_indices):
        mase = hard_mases[idx]
        if mase > self.trouble_mase_threshold:
            row = val_df.loc[window_idx]
            window_data_path = row.get('window_data_path', '')
            origin = row.get('origin', 0)
            window_size = row.get('window_size', 600)
            best_policy = min(policies, key=lambda p: p.avg_mase) if policies else None
            best_name = best_policy.name if best_policy else 'unknown'
            self._collect_trouble_window(
                window_idx, mase, window_data_path,
                origin, window_size, best_name
            )
            collected_count += 1
    self.logger.log(f"   📥 收集 {collected_count} 个困难窗口到全局池")

    hard_df = val_df.loc[hard_indices].copy()
    self.logger.log(f"   🧠 在 {len(hard_df)} 个困难窗口上生成新策略候选...")

    try:
        result = self.inducer.induce(hard_df, force_regenerate=True)
        new_candidates_data = result.get('policies', [])
        new_candidates = [SkillPolicy.from_dict(p) for p in new_candidates_data]
    except Exception as e:
        self.logger.log(f"   ⚠️ Re-Induction 失败: {e}")
        return 0

    if not new_candidates:
        self.logger.log("   ℹ️ 未生成新策略候选")
        return 0

    self.logger.log(f"   📋 生成 {len(new_candidates)} 个候选策略")

    # ★★★ 质量门控 ★★★
    active_policies = [p for p in policies if p.status not in ['ARCHIVE', 'DELETE']]
    if active_policies:
        worst_mase = max(p.avg_mase for p in active_policies)
    else:
        worst_mase = float('inf')

    self.logger.log(f"   📊 池中最差策略 MASE: {worst_mase:.4f}")

    filtered_candidates = []
    for candidate in new_candidates:
        if candidate.avg_mase < worst_mase:
            filtered_candidates.append(candidate)
            self.logger.log(f"      ✅ 候选 {candidate.name} (MASE={candidate.avg_mase:.4f}) 通过质量门控")
        else:
            self.logger.log(f"      ❌ 候选 {candidate.name} (MASE={candidate.avg_mase:.4f}) 被质量门控过滤（>= {worst_mase:.4f}）")

    if not filtered_candidates:
        self.logger.log("   ℹ️ 所有候选策略均未通过质量门控，无新增策略")
        return 0

    self.logger.log(f"   📋 通过质量门控的候选: {len(filtered_candidates)} 个")

    # ★★★ 概率补偿 ★★★
    if hasattr(self, '_distribution_model') and self._distribution_model is not None:
        old_policy_ids = [p.policy_id for p in policies if p.status not in ['ARCHIVE', 'DELETE']]
        new_policy_ids = [c.policy_id for c in filtered_candidates]
        self._distribution_model.compensate_old_policies(old_policy_ids, new_policy_ids)
        self.logger.log(f"   📊 已补偿 {len(old_policy_ids)} 个旧策略的 θ（加入 {len(new_policy_ids)} 个新策略）")

    # ★★★ 加入候选策略 ★★★
    added = 0
    existing_ids = {p.policy_id for p in policies}

    for candidate in filtered_candidates:
        if candidate.policy_id in existing_ids:
            continue

        # 直接设为 TRIAL
        candidate.status = 'TRIAL'
        if not candidate.name.startswith('reind_'):
            candidate.name = f"reind_{round_num}_{candidate.name[:8]}"
        candidate.created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        candidate.logit_weight = 0.0
        candidate.metadata['trial_start_round'] = round_num
        candidate.metadata['trial_freeze_rounds'] = 2

        policies.append(candidate)
        added += 1
        existing_ids.add(candidate.policy_id)

        if hasattr(self, '_distribution_model') and self._distribution_model is not None:
            self._distribution_model.theta[candidate.policy_id] = 0.0
            self.logger.log(f"      📊 已同步到分布模型: {candidate.name} (theta=0.0)")

        self.logger.log(f"      ✅ 加入新策略候选: {candidate.name} (logit_weight=0.0, trial_start_round={round_num})")

        if added >= self.max_new_policies_per_round:
            self.logger.log(f"   📌 已达到本轮新增上限 ({self.max_new_policies_per_round})")
            break

    if added > 0:
        self.logger.log(f"   ✅ 本轮新增 {added} 个策略候选（将由 RL 决定是否使用）")
    else:
        self.logger.log("   ℹ️ 没有新的策略候选加入")

    return added


# ★★★ 辅助方法（保持不变） ★★★
def build_global_summary_impl(self: PolicyEvolutionEngine, policies: List[SkillPolicy]) -> Dict:
    """构建全局策略摘要"""
    if not policies:
        return {
            'total': 0,
            'status_distribution': {},
            'scene_coverage': {},
            'avg_mase': 0.0,
            'best_policy': '无',
            'worst_policy': '无'
        }

    status_dist = {}
    for p in policies:
        status_dist[p.status] = status_dist.get(p.status, 0) + 1

    scene_coverage = {}
    for p in policies:
        groups = tuple(sorted(p.feature_groups)) if p.feature_groups else ('general',)
        scene_coverage[groups] = scene_coverage.get(groups, 0) + 1

    mases = [p.avg_mase for p in policies]
    avg_mase = np.mean(mases) if mases else 0.0
    best_policy = min(policies, key=lambda x: x.avg_mase) if policies else None
    worst_policy = max(policies, key=lambda x: x.avg_mase) if policies else None

    return {
        'total': len(policies),
        'status_distribution': status_dist,
        'scene_coverage': scene_coverage,
        'avg_mase': avg_mase,
        'best_policy': best_policy.name if best_policy else '无',
        'best_mase': best_policy.avg_mase if best_policy else 0.0,
        'worst_policy': worst_policy.name if worst_policy else '无',
        'worst_mase': worst_policy.avg_mase if worst_policy else 0.0
    }


def format_global_summary_impl(self: PolicyEvolutionEngine, summary: Dict) -> str:
    """格式化全局摘要为字符串"""
    lines = ["全局策略池摘要："]
    lines.append(f"- 共 {summary['total']} 条策略")
    lines.append(f"- 状态分布: {summary['status_distribution']}")

    if summary['scene_coverage']:
        scene_str = []
        for groups, count in summary['scene_coverage'].items():
            label = '+'.join(groups) if isinstance(groups, tuple) else str(groups)
            scene_str.append(f"{label}:{count}")
        lines.append(f"- 场景覆盖: {', '.join(scene_str)}")

    lines.append(f"- 平均MASE: {summary['avg_mase']:.4f}")
    lines.append(f"- 最优策略: {summary['best_policy']} (MASE={summary['best_mase']:.4f})")
    lines.append(f"- 最差策略: {summary['worst_policy']} (MASE={summary['worst_mase']:.4f})")

    return '\n'.join(lines)


# ★★★ 注入缓存支持 ★★★
def set_cache_impl(self, cache: Dict):
    """设置 RL 缓存（供质量门控使用）"""
    self._rl_cache = cache


# 将方法注入到类中
PolicyEvolutionEngine.run_round = run_round_impl
PolicyEvolutionEngine._reinduction_rl = reinduction_rl_impl
PolicyEvolutionEngine._build_global_summary = build_global_summary_impl
PolicyEvolutionEngine._format_global_summary = format_global_summary_impl
PolicyEvolutionEngine.set_cache = set_cache_impl


# ★★★ 注入分布模型引用 ★★★
def set_distribution_model(self, distribution_model):
    self._distribution_model = distribution_model


PolicyEvolutionEngine.set_distribution_model = set_distribution_model

# 初始化 _rl_cache 属性（如果没有的话）
if not hasattr(PolicyEvolutionEngine, '_rl_cache'):
    PolicyEvolutionEngine._rl_cache = None