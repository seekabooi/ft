# experiments/autotune/rl_components.py
"""
RL 组件：Reward、Advantage、Baseline 计算
"""

import numpy as np
from typing import Dict, Optional


def compute_reward(mase: float, mase_weight: float = 1.0,
                   stability_penalty: float = 0.0,
                   consistency_bonus: float = 0.0) -> float:
    """
    计算 Reward

    reward = -mase_weight * mase - stability_penalty + consistency_bonus
    """
    return -mase_weight * mase - stability_penalty + consistency_bonus


def compute_advantage(reward: float, baseline: float) -> float:
    """
    计算 Advantage

    A = reward - baseline
    """
    return reward - baseline


class BaselineTracker:
    """Baseline 跟踪器（EMA）"""

    def __init__(self, initial: float = 0.0, decay: float = 0.9):
        self.baseline = initial
        self.decay = decay
        self._count = 0

    def update(self, reward: float):
        """更新 baseline（EMA）"""
        if self._count == 0:
            self.baseline = reward
        else:
            self.baseline = self.decay * self.baseline + (1 - self.decay) * reward
        self._count += 1

    def get(self) -> float:
        return self.baseline

    def reset(self, initial: float = 0.0):
        self.baseline = initial
        self._count = 0

    def to_dict(self) -> Dict:
        return {
            'baseline': self.baseline,
            'decay': self.decay,
            'count': self._count
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'BaselineTracker':
        tracker = cls(
            initial=data.get('baseline', 0.0),
            decay=data.get('decay', 0.9)
        )
        tracker._count = data.get('count', 0)
        return tracker


def normalize_advantage(advantages: np.ndarray) -> np.ndarray:
    """归一化 Advantage（Z-score）"""
    mean = np.mean(advantages)
    std = np.std(advantages) + 1e-8
    return (advantages - mean) / std


def compute_entropy(probs: np.ndarray) -> float:
    """计算熵"""
    probs = np.array(probs)
    return -np.sum(probs * np.log(probs + 1e-8))


def compute_kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """计算 KL 散度"""
    p = np.array(p) + 1e-8
    q = np.array(q) + 1e-8
    return np.sum(p * np.log(p / q))