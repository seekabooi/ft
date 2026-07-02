# experiments/autotune/regime_drift_monitor.py
"""
Regime Drift Monitor（P8 + P15）
检测策略中心漂移，触发 Rule Refresh
"""

import numpy as np
from typing import Dict, List, Optional, Any
from collections import deque


class RegimeDriftMonitor:
    """
    状态漂移监控器（P8 + P15）

    功能：
    1. 记录 policy embedding_center 随时间变化
    2. 计算 drift_distance
    3. 超过阈值触发 Rule Refresh

    Refresh 定义（P15）：
        - ✅ 允许：更新 embedding、更新 state_condition、更新 skill_strategy 权重
        - ❌ 不允许：改变规则 ID、改变规则名称、改变规则数量
    """

    def __init__(self, config: Dict, logger):
        self.config = config
        self.logger = logger

        drift_cfg = config.get('regime_drift', {})
        self.drift_threshold = drift_cfg.get('drift_threshold', 0.3)
        self.monitor_window = drift_cfg.get('monitor_window', 100)

        # 历史记录
        self._embedding_history = deque(maxlen=self.monitor_window)
        self._center_history = deque(maxlen=10)
        self._drift_alerts = []

    def update(self, state_embeddings: List[np.ndarray], policy_embeddings: List[np.ndarray]):
        """更新状态记录"""
        if state_embeddings:
            # 计算当前状态中心
            current_center = np.mean(state_embeddings, axis=0)
            self._embedding_history.append(current_center)

        if policy_embeddings:
            # 计算策略中心
            policy_center = np.mean(policy_embeddings, axis=0)
            self._center_history.append(policy_center)

    def detect_drift(self) -> Dict:
        """
        检测漂移

        Returns:
            {
                'has_drift': bool,
                'drift_distance': float,
                'drift_direction': str,
                'should_refresh': bool,
                'refresh_candidates': list
            }
        """
        if len(self._center_history) < 5:
            return {'has_drift': False, 'drift_distance': 0.0, 'should_refresh': False}

        # 计算当前中心 vs 历史中心
        current_center = self._center_history[-1]
        historical_center = np.mean(list(self._center_history)[:max(1, len(self._center_history) - 5)], axis=0)

        drift_distance = np.linalg.norm(current_center - historical_center)

        # 判断方向
        if drift_distance > 0.01:
            drift_direction = 'positive' if current_center[0] > historical_center[0] else 'negative'
        else:
            drift_direction = 'stable'

        has_drift = drift_distance > self.drift_threshold

        result = {
            'has_drift': has_drift,
            'drift_distance': drift_distance,
            'drift_direction': drift_direction,
            'should_refresh': has_drift,
            'threshold': self.drift_threshold
        }

        if has_drift:
            self._drift_alerts.append(result)

        return result

    def get_refresh_suggestions(self, policies: List) -> List:
        """
        获取需要 Refresh 的策略

        Refresh 触发条件（P15）：
            1. 检测到漂移
            2. 策略的 embedding 与当前中心距离过大

        Returns:
            需要 Refresh 的策略列表
        """
        if not policies:
            return []

        drift_result = self.detect_drift()
        if not drift_result['should_refresh']:
            return []

        refresh_candidates = []

        for policy in policies:
            if policy.in_cooldown:
                continue

            if policy.embedding:
                policy_center = np.array(policy.embedding)
                current_center = self._center_history[-1] if self._center_history else None

                if current_center is not None:
                    distance = np.linalg.norm(policy_center - current_center)
                    if distance > self.drift_threshold:
                        refresh_candidates.append({
                            'policy': policy,
                            'distance': distance,
                            'threshold': self.drift_threshold
                        })

        refresh_candidates.sort(key=lambda x: x['distance'], reverse=True)
        return refresh_candidates

    def get_drift_history(self) -> List:
        """获取漂移历史"""
        return self._drift_alerts