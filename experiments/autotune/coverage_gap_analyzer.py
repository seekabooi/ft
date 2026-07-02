# experiments/autotune/coverage_gap_analyzer.py
"""
Coverage Gap Analyzer（P8 + P15）
检测新区域 + gap_support
"""

import numpy as np
from typing import Dict, List, Optional, Any
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class CoverageGapAnalyzer:
    """
    覆盖缺口分析器（P8 + P15）

    功能：
    1. 检测新区域（状态空间未覆盖的区域）
    2. gap_support：连续出现 + 最近100窗口至少15个窗口落在同一区域
    3. 为 Re-Induction 提供前置条件

    P15 边界：
        - 局部异常（<5%）→ Patch
        - 中心漂移 → Refresh
        - 新区域出现（>15%）→ Re-Induction
    """

    def __init__(self, config: Dict, logger):
        self.config = config
        self.logger = logger

        gap_cfg = config.get('coverage_gap', {})
        self.min_support = gap_cfg.get('min_support_windows', 15)
        self.support_window = gap_cfg.get('support_window_size', 100)
        self.gap_threshold = gap_cfg.get('gap_threshold', 0.15)

        self._gap_history = []
        self._detected_gaps = []

    def detect_gap(self,
                   state_embeddings: List[np.ndarray],
                   policy_embeddings: List[np.ndarray],
                   hard_windows: List[int]) -> Dict:
        """
        检测覆盖缺口

        Args:
            state_embeddings: 状态嵌入列表
            policy_embeddings: 策略嵌入列表（作为聚类中心）
            hard_windows: 困难窗口索引列表

        Returns:
            {
                'has_gap': bool,
                'gap_ratio': float,
                'gap_indices': list,
                'gap_centers': list,
                'support_confirmed': bool,
                'recommended_action': 'patch' | 'reinduction' | 'none'
            }
        """
        if not state_embeddings or not policy_embeddings:
            return {'has_gap': False, 'gap_ratio': 0.0, 'recommended_action': 'none'}

        # 1. 计算每个状态到最近策略中心的距离
        distances = []
        for emb in state_embeddings:
            min_dist = min(np.linalg.norm(emb - pe) for pe in policy_embeddings)
            distances.append(min_dist)

        # 2. 找出距离远的状态（潜在缺口）
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        threshold = mean_dist + 1.5 * std_dist

        gap_indices = [i for i, d in enumerate(distances) if d > threshold]

        if not gap_indices:
            return {'has_gap': False, 'gap_ratio': 0.0, 'recommended_action': 'none'}

        # 3. 计算缺口比例
        gap_ratio = len(gap_indices) / max(1, len(state_embeddings))

        # 4. 检查支持度（P8: gap_support）
        # 检查这些缺口是否集中在最近的窗口中
        recent_gap_indices = [i for i in gap_indices if i >= max(0, len(state_embeddings) - self.support_window)]
        support_confirmed = len(recent_gap_indices) >= self.min_support

        # 5. 判断动作（P15 边界）
        if support_confirmed:
            if gap_ratio < 0.05:
                recommended_action = 'patch'
            elif gap_ratio > self.gap_threshold:
                recommended_action = 'reinduction'
            else:
                recommended_action = 'observe'
        else:
            recommended_action = 'none'

        # 6. 聚类找到缺口中心
        gap_centers = []
        if support_confirmed and len(gap_indices) >= 5:
            gap_embeddings = [state_embeddings[i] for i in gap_indices]
            try:
                kmeans = KMeans(n_clusters=min(3, len(gap_embeddings)), random_state=42, n_init=10)
                labels = kmeans.fit_predict(gap_embeddings)
                gap_centers = kmeans.cluster_centers_.tolist()
            except:
                gap_centers = [np.mean(gap_embeddings, axis=0).tolist()]

        result = {
            'has_gap': support_confirmed,
            'gap_ratio': gap_ratio,
            'gap_indices': gap_indices,
            'gap_centers': gap_centers,
            'support_confirmed': support_confirmed,
            'recommended_action': recommended_action,
            'num_gaps': len(gap_indices),
            'num_recent_gaps': len(recent_gap_indices)
        }

        if support_confirmed:
            self._detected_gaps.append(result)

        return result

    def compute_gap_ratio(self, state_embeddings: List[np.ndarray],
                          policy_embeddings: List[np.ndarray]) -> float:
        """计算缺口比例"""
        if not state_embeddings or not policy_embeddings:
            return 0.0

        distances = []
        for emb in state_embeddings:
            min_dist = min(np.linalg.norm(emb - pe) for pe in policy_embeddings)
            distances.append(min_dist)

        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        threshold = mean_dist + 1.5 * std_dist

        gap_count = sum(1 for d in distances if d > threshold)
        return gap_count / max(1, len(state_embeddings))

    def get_gap_history(self) -> List:
        """获取缺口历史"""
        return self._detected_gaps