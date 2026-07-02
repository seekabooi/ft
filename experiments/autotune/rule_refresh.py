# experiments/autotune/rule_refresh.py
"""
Rule Refresh（P15）
更新已有规则，不改变规则数量
"""

import numpy as np
from typing import Dict, List, Optional, Any


class RuleRefresher:
    """
    规则刷新器（P15）

    Refresh 定义：
        - ✅ 允许：更新 embedding、更新 state_condition、更新 skill_strategy 权重
        - ❌ 不允许：改变规则 ID、改变规则名称、改变规则数量
    """

    def __init__(self, config: Dict, logger):
        self.config = config
        self.logger = logger

    def refresh(self, policy, new_embedding: List[float],
                new_condition: Dict, new_strategy: Optional[Dict] = None,
                current_step: int = 0) -> Dict:
        """
        刷新规则

        Args:
            policy: 要刷新的策略
            new_embedding: 新的 embedding
            new_condition: 新的状态条件
            new_strategy: 新的执行策略（可选）
            current_step: 当前步数

        Returns:
            {
                'refreshed': bool,
                'changes': list,
                'new_embedding': list,
                'new_condition': dict
            }
        """
        changes = []

        # 1. 更新 embedding（✅ 允许）
        if new_embedding and len(new_embedding) == len(policy.embedding):
            old_embedding = policy.embedding.copy()
            policy.embedding = new_embedding
            changes.append('embedding_updated')

        # 2. 更新 state_condition（✅ 允许）
        if new_condition:
            old_condition = policy.state_condition.copy()
            policy.state_condition = new_condition
            changes.append('condition_updated')

        # 3. 更新 skill_strategy 权重（✅ 允许）
        if new_strategy:
            policy.skill_strategy = new_strategy
            changes.append('strategy_updated')

        # 4. 记录刷新（P13: 进入冷却）
        policy.refresh(new_embedding if new_embedding else policy.embedding,
                       new_condition if new_condition else policy.state_condition,
                       current_step)

        # 5. ✅ 不改变：policy_id、name、version

        return {
            'refreshed': len(changes) > 0,
            'changes': changes,
            'new_embedding': policy.embedding,
            'new_condition': policy.state_condition,
            'policy_id': policy.policy_id,
            'name': policy.name
        }

    def refresh_from_drift(self, policy, drift_center: List[float], current_step: int) -> Dict:
        """
        根据漂移中心刷新策略

        Args:
            policy: 要刷新的策略
            drift_center: 漂移中心
            current_step: 当前步数

        Returns:
            refresh结果
        """
        # 计算新 embedding（向漂移中心靠近）
        if policy.embedding and drift_center:
            old_emb = np.array(policy.embedding)
            new_center = np.array(drift_center)

            # 加权平均：保留部分原有特征，吸收漂移特征
            alpha = 0.3
            new_embedding = (1 - alpha) * old_emb + alpha * new_center
            new_embedding = new_embedding.tolist()

            # 保持归一化
            norm = np.linalg.norm(new_embedding)
            if norm > 0:
                new_embedding = (np.array(new_embedding) / norm).tolist()
        else:
            new_embedding = policy.embedding.copy()

        return self.refresh(policy, new_embedding, policy.state_condition, None, current_step)

    def refresh_from_candidate(self, policy, candidate_policy, current_step: int) -> Dict:
        """
        从候选策略刷新

        Args:
            policy: 要刷新的策略
            candidate_policy: 候选策略（TRIAL状态）
            current_step: 当前步数

        Returns:
            refresh结果
        """
        # 从候选策略吸收好的特征
        new_embedding = candidate_policy.embedding.copy()
        new_condition = candidate_policy.state_condition.copy()
        new_strategy = candidate_policy.skill_strategy.copy()

        # 候选策略转为ARCHIVE
        candidate_policy.status = 'ARCHIVE'

        return self.refresh(policy, new_embedding, new_condition, new_strategy, current_step)