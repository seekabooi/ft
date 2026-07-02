# experiments/autotune/retirement_mechanism.py
"""
Retirement Mechanism - 已彻底禁用
所有退休判定已关闭，仅保留空壳函数以便兼容。
策略池只保留 ACTIVE 和 TRIAL 两种状态。
复活功能已移至 CheckpointManager 中一次性执行。
"""

import numpy as np
import random
from typing import Dict, List, Optional, Any
from collections import defaultdict

from experiments.autotune.skill_policy import SkillPolicy   # ★★★ 添加这行


class RetirementMechanism:
    """
    退休机制 - 已禁用
    """

    def __init__(self, config: Dict, logger):
        self.config = config
        self.logger = logger
        self.logger.log("   ℹ️ [退休机制] 已彻底禁用，策略永远不会被标记为 DEPRECATED/ARCHIVE/DELETE")

        # 保留配置读取（不再使用）
        ret_cfg = config.get('retirement', {})
        self.weights = ret_cfg.get('retire_score_weights', {
            'utility': 0.35,
            'coverage': 0.20,
            'rare_score': 0.20,
            'uniqueness': 0.10,
            'marginal_value': 0.15
        })
        self.min_marginal_threshold = ret_cfg.get('minimum_marginal_threshold', 0.01)
        self.deprecated_periods = ret_cfg.get('deprecated_periods', 2)
        self.archive_periods = ret_cfg.get('archive_periods', 1)

        # 概率门控参数（保留但不再使用）
        rl_cfg = config.get('rl', {})
        self.evolution_effect_strength = rl_cfg.get('evolution_effect_strength', 0.30)

        # 温和退休参数（保留但不再使用）
        self.max_retire_per_round = 3
        self.observation_rounds = 1
        self.min_per_cluster = 1
        self.min_per_group = 2

        self.threshold_trial = 0.35
        self.threshold_active = 0.45
        self.threshold_deprecated = 0.55
        self.p_evo_min = 0.15
        self.p_evo_max = 0.70
        self.force_retire_threshold = 0.80

    def compute_retire_score(self, policy) -> float:
        """计算退休分数（仅用于日志，不触发任何动作）"""
        score = (
                self.weights['utility'] * (1.0 - max(0, min(1.0, policy.utility_ema))) +
                self.weights['coverage'] * (1.0 - max(0, min(1.0, policy.coverage_rate))) +
                self.weights['rare_score'] * (1.0 - max(0, min(1.0, policy.rare_score))) +
                self.weights['uniqueness'] * (1.0 - max(0, min(1.0, policy.uniqueness))) +
                self.weights['marginal_value'] * (1.0 - max(0, min(1.0, policy.marginal_value)))
        )
        return min(1.0, max(0.0, score))

    def should_retire(self, policy, retire_score: float, threshold: float = 0.6) -> Dict:
        """始终返回 False（退休禁用）"""
        return {
            'should_retire': False,
            'reason': 'Retirement mechanism is disabled',
            'retire_score': retire_score,
            'marginal_value_ok': True
        }

    def _is_policy_frozen(self, policy, current_round: int) -> bool:
        """判断策略是否在冻结期（保留供其他逻辑使用，但不用于退休）"""
        if policy.status != 'TRIAL':
            return False
        trial_start = policy.metadata.get('trial_start_round', 0)
        trial_freeze = policy.metadata.get('trial_freeze_rounds', 2)
        return current_round - trial_start < trial_freeze

    def _get_observation_count(self, policy) -> int:
        return policy.metadata.get('retire_observation_count', 0)

    def _increment_observation_count(self, policy):
        policy.metadata['retire_observation_count'] = policy.metadata.get('retire_observation_count', 0) + 1

    def _reset_observation_count(self, policy):
        policy.metadata['retire_observation_count'] = 0

    def get_next_status(self, policy, current_step: int, current_round: int = 0) -> Dict:
        """始终返回当前状态（不进行任何状态转移）"""
        return {
            'new_status': policy.status,
            'reason': 'Retirement mechanism disabled',
            'transition': False
        }

    def get_retirement_candidates(self, policies: List, top_k: int = 3,
                                  cluster_id: Optional[str] = None,
                                  current_round: int = 0) -> List:
        """
        永久返回空列表，不退休任何策略
        """
        self.logger.log("   ℹ️ [退休机制] get_retirement_candidates 被调用，但已禁用，返回空列表")
        return []

    def get_status_summary(self, policies: List) -> Dict:
        """统计状态分布（用于日志）"""
        summary = {
            'ACTIVE': 0,
            'TRIAL': 0,
            'DEPRECATED': 0,
            'ARCHIVE': 0,
            'DELETE': 0
        }
        for policy in policies:
            status = policy.status
            if status in summary:
                summary[status] += 1
        return summary

    def revive_retired_policies(self, policies: List[SkillPolicy], current_round: int = 0,
                                initial_theta: float = -0.5) -> int:
        """
        复活策略的公共接口 - 由 tuner_core 调用
        返回复活的数量
        """
        return self._revive_policies_internal(policies, current_round, initial_theta)

    def _revive_policies_internal(self, policies: List[SkillPolicy], current_round: int,
                                  initial_theta: float) -> int:
        """
        内部复活实现 - 将 DEPRECATED/ARCHIVE/DELETE 转为 TRIAL，不冻结
        """
        revived = []
        for policy in policies:
            if policy.status in ['DEPRECATED', 'ARCHIVE', 'DELETE']:
                old_status = policy.status
                policy.status = 'TRIAL'
                # ★★★ 关键：不设置冻结期，立即参与演化 ★★★
                policy.metadata['trial_start_round'] = max(0, current_round - 2)
                policy.metadata['trial_freeze_rounds'] = 0
                policy.metadata['revived'] = True
                policy.metadata['original_status'] = old_status
                policy.metadata['revived_at_round'] = current_round
                policy.logit_weight = initial_theta
                policy.selection_count = 0
                policy.cumulative_reward = 0.0
                revived.append((policy.policy_id, policy.name, old_status))

        if revived:
            self.logger.log("")
            self.logger.log("█" * 80)
            self.logger.log("█  ♻️ 复活已退休策略（一次性操作）")
            self.logger.log("█  ─────────────────────────────────────────────")
            for pid, name, old_st in revived:
                self.logger.log(f"█  ✅ 复活: {name[:20]} (ID: {pid}) 原状态: {old_st} → TRIAL, θ={initial_theta}")
            self.logger.log("█  ─────────────────────────────────────────────")
            self.logger.log(f"█  共复活 {len(revived)} 条策略，已跳过冻结期，立即参与演化")
            self.logger.log("█  新策略将在下一轮被分配到对应的簇")
            self.logger.log("█" * 80)
            self.logger.log("")

        return len(revived)