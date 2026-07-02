# experiments/autotune/policy_distribution.py
"""
Policy Distribution Model - 策略分布模型

管理可学习的 logit 权重 θ，提供 softmax 分布，执行梯度更新。
★ ★ 2026-06-27 增加 adaptive decay（Patch 2）
★ ★ ★ 2026-06-28 增加梯度裁剪（Gradient Clipping）
★ ★ ★ ★ 2026-06-28 增加概率补偿（方案 E）
"""

import numpy as np
import math
from typing import Dict, List, Optional
from experiments.autotune.skill_policy import SkillPolicy


class PolicyDistributionModel:
    """策略分布模型，支持 Policy Gradient 在线学习"""

    def __init__(self, learning_rate: float = 0.01, temperature: float = 1.0, theta_decay: float = 0.01,
                 grad_clip_value: float = 1.0):
        self.theta: Dict[str, float] = {}  # policy_id -> logit weight
        self.learning_rate = learning_rate
        self.temperature = temperature
        self.theta_decay = theta_decay  # ★★★ Patch 2：权重衰减基准值
        self.grad_clip_value = grad_clip_value  # ★★★ 梯度裁剪阈值
        self._step = 0
        self._theta_clip_min = -5.0  # θ 下限
        self._theta_clip_max = 5.0  # θ 上限

    def get_distribution(self, state: Dict[str, float], policies: List[SkillPolicy]) -> Dict[str, float]:
        """计算策略分布 π(p|s)"""
        logits = {}
        for p in policies:
            if p.status in ['ARCHIVE', 'DELETE']:
                continue
            # 基础权重（可学习） + 状态相似度（先验）
            base = self.theta.get(p.policy_id, 0.0)
            similarity = p.compute_applicability_score(state)
            logits[p.policy_id] = base + similarity

        # 应用温度
        if self.temperature != 1.0:
            logits = {k: v / self.temperature for k, v in logits.items()}

        # softmax
        exp_values = np.exp(list(logits.values()))
        sum_exp = np.sum(exp_values) + 1e-8
        probs = {pid: exp_values[i] / sum_exp for i, pid in enumerate(logits.keys())}
        return probs

    # ★★★ 修改：update 方法增加 adaptive decay（Patch 2）和梯度裁剪 ★★★
    def update(self, policy_id: str, advantage: float, policy: Optional[SkillPolicy] = None,
               grad_clip_value: Optional[float] = None):
        """
        策略梯度更新（带 adaptive decay + 梯度裁剪）

        θ = (1 - decay) * θ + lr * clipped_advantage

        其中 decay = theta_decay / sqrt(selection_count + 1)
        高频访问策略 → 更小 decay（保持记忆）
        低频策略 → 更大 decay（加速遗忘）
        """
        # 获取当前 θ
        current_theta = self.theta.get(policy_id, 0.0)

        # 计算 adaptive decay
        if policy is not None:
            visit_count = policy.selection_count
        else:
            visit_count = 0

        # adaptive_decay = base_decay / sqrt(visit_count + 1)
        # 高频访问 → decay 小，低频 → decay 大
        adaptive_decay = self.theta_decay / (math.sqrt(visit_count + 1) + 1e-8)
        adaptive_decay = max(0.001, min(0.05, adaptive_decay))  # 限幅

        # ★★★ 梯度裁剪 ★★★
        clip_val = grad_clip_value if grad_clip_value is not None else self.grad_clip_value
        if clip_val > 0:
            clipped_advantage = np.clip(advantage, -clip_val, clip_val)
        else:
            clipped_advantage = advantage

        # L2-regularized policy gradient
        # θ = (1 - decay) * θ + lr * advantage
        new_theta = (1.0 - adaptive_decay) * current_theta + self.learning_rate * clipped_advantage

        # 限幅防止极端值
        new_theta = max(self._theta_clip_min, min(self._theta_clip_max, new_theta))

        self.theta[policy_id] = new_theta
        self._step += 1

    # ★★★ ★★★ ★★★ 概率补偿（方案 E） ★★★ ★★★ ★★★
    def compensate_old_policies(self, old_policy_ids: List[str], new_policy_ids: List[str]):
        """
        当新策略加入时，补偿旧策略的 θ，使其采样概率保持不变。

        补偿公式：θ_old_new = θ_old + log((N + K) / N)
        其中 N = len(old_policy_ids)，K = len(new_policy_ids)

        这确保了旧策略的总概率质量在加入新策略后保持不变。
        """
        if not old_policy_ids or not new_policy_ids:
            return

        N = len(old_policy_ids)
        K = len(new_policy_ids)
        compensation = math.log((N + K) / N)

        for pid in old_policy_ids:
            if pid in self.theta:
                new_theta = self.theta[pid] + compensation
                new_theta = max(self._theta_clip_min, min(self._theta_clip_max, new_theta))
                self.theta[pid] = new_theta

    def set_theta(self, policy_id: str, value: float):
        """直接设置 θ 值（用于强制休眠等场景）"""
        clipped = max(self._theta_clip_min, min(self._theta_clip_max, value))
        self.theta[policy_id] = clipped

    def get_theta(self, policy_id: str) -> float:
        return self.theta.get(policy_id, 0.0)

    def set_learning_rate(self, lr: float):
        self.learning_rate = lr

    def set_temperature(self, temp: float):
        self.temperature = temp

    def set_theta_decay(self, decay: float):
        self.theta_decay = decay

    def set_grad_clip_value(self, clip: float):
        self.grad_clip_value = clip

    def to_dict(self) -> Dict:
        return {
            'theta': self.theta,
            'learning_rate': self.learning_rate,
            'temperature': self.temperature,
            'theta_decay': self.theta_decay,
            'grad_clip_value': self.grad_clip_value,
            'step': self._step
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'PolicyDistributionModel':
        model = cls(
            learning_rate=data.get('learning_rate', 0.01),
            temperature=data.get('temperature', 1.0),
            theta_decay=data.get('theta_decay', 0.01),
            grad_clip_value=data.get('grad_clip_value', 1.0)
        )
        model.theta = data.get('theta', {})
        model._step = data.get('step', 0)
        return model