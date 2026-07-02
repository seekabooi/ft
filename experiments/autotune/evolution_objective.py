# experiments/autotune/evolution_objective.py
"""
Evolution Objective（P10 + P12）
统一演化目标函数，所有演化行为最终优化同一个目标
"""

import numpy as np
from typing import Dict, Optional, List
from datetime import datetime


class EvolutionObjective:
    """
    演化目标函数

    evolution_score =
        0.5 × accuracy_gain
        + 0.2 × coverage_gain
        + 0.2 × stability_gain
        - 0.1 × complexity_penalty
        - 0.1 × evolution_cost

    所有演化行为看 delta_evolution_score
    """

    def __init__(self, config: Dict, logger):
        self.config = config
        self.logger = logger

        obj_cfg = config.get('evolution_objective', {})
        weights = obj_cfg.get('weights', {})

        self.w_accuracy = weights.get('accuracy_gain', 0.5)
        self.w_coverage = weights.get('coverage_gain', 0.2)
        self.w_stability = weights.get('stability_gain', 0.2)
        self.w_complexity = weights.get('complexity_penalty', 0.1)
        self.w_cost = weights.get('evolution_cost', 0.1)

        # 目标策略数（P4）
        self.target_policies = config.get('policy_pool', {}).get('target_policies', 8)

        # 历史记录
        self.history = []

    def compute_evolution_score(self,
                                accuracy_before: float,
                                accuracy_after: float,
                                coverage_before: float,
                                coverage_after: float,
                                stability_before: float,
                                stability_after: float,
                                policy_count_before: int,
                                policy_count_after: int,
                                evolution_cost: float = 0.0) -> Dict:
        """
        计算演化分数

        Args:
            accuracy_before/after: 验证集 MASE（越小越好，转化为增益）
            coverage_before/after: 覆盖率（越大越好）
            stability_before/after: 稳定性（1/方差，越大越好）
            policy_count_before/after: 策略数量
            evolution_cost: 演化成本（0~1）

        Returns:
            {
                'score': 总分数,
                'accuracy_gain': 准确率增益,
                'coverage_gain': 覆盖率增益,
                'stability_gain': 稳定性增益,
                'complexity_penalty': 复杂度惩罚,
                'evolution_cost': 演化成本,
                'improved': 是否改善
            }
        """
        # 1. 准确率增益（MASE 下降为正）
        if accuracy_before > 0:
            accuracy_gain = (accuracy_before - accuracy_after) / accuracy_before
        else:
            accuracy_gain = 0.0

        # 2. 覆盖率增益
        coverage_gain = coverage_after - coverage_before

        # 3. 稳定性增益
        stability_gain = stability_after - stability_before

        # 4. 复杂度惩罚（策略数超出目标越多，惩罚越大）
        complexity_penalty = max(0, policy_count_after - self.target_policies) / max(1, self.target_policies)

        # 5. 演化成本
        cost = evolution_cost

        # 6. 总分数
        score = (self.w_accuracy * max(0, accuracy_gain) +
                 self.w_coverage * max(0, coverage_gain) +
                 self.w_stability * max(0, stability_gain) -
                 self.w_complexity * complexity_penalty -
                 self.w_cost * cost)

        improved = score > 0.001

        result = {
            'score': score,
            'accuracy_gain': accuracy_gain,
            'coverage_gain': coverage_gain,
            'stability_gain': stability_gain,
            'complexity_penalty': complexity_penalty,
            'evolution_cost': cost,
            'improved': improved,
            'accuracy_before': accuracy_before,
            'accuracy_after': accuracy_after,
            'coverage_before': coverage_before,
            'coverage_after': coverage_after,
            'policy_count_before': policy_count_before,
            'policy_count_after': policy_count_after
        }

        self.history.append(result)
        return result

    def compute_evolution_cost(self, action_type: str, action_impact: float) -> float:
        """
        计算演化成本（P12）

        Args:
            action_type: 'merge' | 'retire' | 'patch' | 'refresh' | 'reinduction'
            action_impact: 动作影响程度（0~1）

        Returns:
            cost: 0~1 的演化成本
        """
        # 基础成本（不同动作风险不同）
        base_costs = {
            'merge': 0.3,
            'retire': 0.2,
            'patch': 0.15,
            'refresh': 0.1,
            'reinduction': 0.25,
        }

        base = base_costs.get(action_type, 0.2)

        # 最终成本 = 基础成本 × 影响程度
        cost = base * min(1.0, action_impact * 1.5)

        return min(1.0, cost)

    def should_evolve(self, current_score: float, candidate_score: float,
                      min_improvement: float = 0.01) -> bool:
        """
        判断是否应该执行演化

        Args:
            current_score: 当前演化分数
            candidate_score: 候选演化分数
            min_improvement: 最小改善阈值

        Returns:
            True 如果候选分数显著改善
        """
        return (candidate_score - current_score) > min_improvement

    def get_status(self) -> Dict:
        """获取演化目标状态"""
        if not self.history:
            return {'total_evolutions': 0, 'last_score': 0.0}

        recent = self.history[-10:] if len(self.history) >= 10 else self.history
        return {
            'total_evolutions': len(self.history),
            'last_score': self.history[-1]['score'],
            'avg_recent_score': np.mean([h['score'] for h in recent]),
            'improvement_rate': sum(1 for h in recent if h['improved']) / max(1, len(recent))
        }


def compute_complexity_penalty(policy_count: int, target: int) -> float:
    """计算复杂度惩罚"""
    if policy_count <= target:
        return 0.0
    return (policy_count - target) / max(1, target)