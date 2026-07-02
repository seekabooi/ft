# experiments/autotune/rule_engine.py
"""
Policy Execution Engine - v4 全功能版
★ 从硬匹配改为相似度匹配
★ 基于 compute_applicability_score 进行软路由
"""

import numpy as np
from typing import Dict, List, Optional, Any
from experiments.autotune.skill_policy import SkillPolicy, create_policy_from_legacy_rule


class PolicyExecutionEngine:
    """
    Policy Execution Engine - v4 全功能版

    核心变化：
    - retrieve: 基于相似度排序，不再使用硬匹配
    - retrieve_top_k: 返回 top-k 个最相似策略
    """

    def __init__(self, policies_config: Dict = None, config: Optional[Dict] = None):
        self.policies: List[SkillPolicy] = []
        self.default_policy: Optional[SkillPolicy] = None
        self.config = config or {}
        if policies_config:
            self.load_policies(policies_config)

    def load_policies(self, config: Dict):
        legacy_rules = config.get('rules', [])
        if legacy_rules:
            self.policies = [create_policy_from_legacy_rule(r, self.config) for r in legacy_rules]
        policies_data = config.get('policies', [])
        if policies_data:
            self.policies = [SkillPolicy.from_dict(p) for p in policies_data]
        self._update_default()

    def set_policies(self, policies: List[SkillPolicy]):
        """直接设置策略列表"""
        self.policies = policies
        self._update_default()

    def retrieve(self, state: Dict[str, float]) -> Optional[SkillPolicy]:
        """
        ★ 软路由：基于相似度选择最佳策略
        替代原来的硬匹配（is_applicable）
        """
        if not self.policies:
            return self.default_policy

        # 计算每个策略的适用性分数
        scored_policies = []
        for policy in self.policies:
            # 使用 numeric 特征计算软条件分数
            score = policy.compute_applicability_score(state)
            scored_policies.append((policy, score))

        # 按分数降序排列
        scored_policies.sort(key=lambda x: x[1], reverse=True)

        # 取最高分策略
        if scored_policies and scored_policies[0][1] > 0.3:  # 阈值可配置
            return scored_policies[0][0]

        return self.default_policy

    def retrieve_top_k(self, state: Dict[str, float], k: int = 3) -> List[SkillPolicy]:
        """
        ★ 返回 top-k 个最相似策略（用于 Soft Mixture）
        """
        if not self.policies:
            return []

        scored_policies = []
        for policy in self.policies:
            score = policy.compute_applicability_score(state)
            scored_policies.append((policy, score))

        scored_policies.sort(key=lambda x: x[1], reverse=True)
        return [p for p, _ in scored_policies[:k]]

    def retrieve_with_scores(self, state: Dict[str, float]) -> List[tuple]:
        """
        返回所有策略及其适用性分数（用于诊断）
        """
        if not self.policies:
            return []

        scored_policies = []
        for policy in self.policies:
            score = policy.compute_applicability_score(state)
            scored_policies.append((policy, score))

        scored_policies.sort(key=lambda x: x[1], reverse=True)
        return scored_policies

    def execute(self, policy: SkillPolicy, history: np.ndarray,
                horizon: int, period: int) -> Optional[np.ndarray]:
        return policy.execute(history, horizon, period)

    def get_all_policies(self) -> List[SkillPolicy]:
        return self.policies

    def add_policy(self, policy: SkillPolicy):
        self.policies.append(policy)
        self._update_default()

    def remove_policy(self, policy_id: str) -> bool:
        for i, p in enumerate(self.policies):
            if p.policy_id == policy_id:
                self.policies.pop(i)
                self._update_default()
                return True
        return False

    def _update_default(self):
        """更新默认策略（取 utility 最高的）"""
        if not self.policies:
            self.default_policy = None
            return
        # 取 utility_score 最高的策略作为默认
        self.default_policy = max(self.policies, key=lambda p: p.utility_score)

    def to_dict(self) -> Dict:
        return {
            'policies': [p.to_dict() for p in self.policies],
            'default_policy_id': self.default_policy.policy_id if self.default_policy else None
        }