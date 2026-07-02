# experiments/autotune/causal_skill_graph.py
"""
Policy Influence Graph

不是统计共现，而是 policy substitution effect graph
"""

from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
import numpy as np


class PolicyInfluenceGraph:
    """
    Policy Influence Graph

    Edge 定义：E(i,j) = Δ performance when using i vs j in same state

    判定逻辑：
    - E(i,j) > 0.05 → i 对 j 有增强影响
    - E(i,j) < -0.05 → j 对 i 有增强影响
    - 互有胜负 → 竞争关系
    """

    def __init__(self):
        self.nodes: Set[str] = set()
        self.enhancement_edges: Dict[str, List[str]] = defaultdict(list)
        self.suppression_edges: Dict[str, List[str]] = defaultdict(list)
        self.competition_edges: Set[Tuple[str, str]] = set()
        self.cooccurrence_matrix: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.performance_diff_matrix: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

    def add_interaction(self, policy_a_id: str, policy_b_id: str,
                        mase_a: float, mase_b: float, state: Dict):
        """记录一次交互"""
        self.nodes.add(policy_a_id)
        self.nodes.add(policy_b_id)

        self.cooccurrence_matrix[policy_a_id][policy_b_id] += 1
        self.cooccurrence_matrix[policy_b_id][policy_a_id] += 1

        diff = mase_a - mase_b
        self.performance_diff_matrix[policy_a_id][policy_b_id] += diff
        self.performance_diff_matrix[policy_b_id][policy_a_id] -= diff

    def update(self):
        """更新图结构"""
        self.enhancement_edges.clear()
        self.suppression_edges.clear()
        self.competition_edges.clear()

        for a_id in self.nodes:
            for b_id in self.nodes:
                if a_id >= b_id:
                    continue

                count = self.cooccurrence_matrix[a_id].get(b_id, 0)
                if count < 2:
                    continue

                total_diff = self.performance_diff_matrix[a_id].get(b_id, 0)
                avg_diff = total_diff / count

                if avg_diff > 0.05:
                    self.enhancement_edges[a_id].append(b_id)
                    self.suppression_edges[b_id].append(a_id)
                elif avg_diff < -0.05:
                    self.enhancement_edges[b_id].append(a_id)
                    self.suppression_edges[a_id].append(b_id)
                else:
                    self.competition_edges.add((a_id, b_id))

    def suggest_retire(self) -> List[str]:
        """建议退休的策略"""
        self.update()

        candidates = []
        for policy_id in self.nodes:
            is_suppressed = any(
                policy_id in self.suppression_edges.get(enhancer, [])
                for enhancer in self.nodes
            )
            if is_suppressed:
                candidates.append(policy_id)

        return candidates

    def get_enhancers(self, policy_id: str) -> List[str]:
        """获取增强该策略的其它策略"""
        return self.enhancement_edges.get(policy_id, [])

    def get_competitors(self, policy_id: str) -> List[str]:
        """获取竞争策略"""
        competitors = []
        for a, b in self.competition_edges:
            if a == policy_id:
                competitors.append(b)
            elif b == policy_id:
                competitors.append(a)
        return competitors

    def to_dict(self) -> Dict:
        return {
            'nodes': list(self.nodes),
            'enhancement_edges': dict(self.enhancement_edges),
            'suppression_edges': dict(self.suppression_edges),
            'competition_edges': list(self.competition_edges)
        }