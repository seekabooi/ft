# experiments/autotune/stability_controller.py
"""
系统稳定性控制器 - 新增模块
实现：结构变化监控 + 变化预算控制
"""

import numpy as np
from typing import Dict, List, Optional, Any
from collections import deque
import time


class StabilityController:
    """
    稳定性控制器

    功能：
    1. 监控结构变化：policy 数量、embedding drift、entropy drift
    2. 执行变化预算：限制每轮的 merge/split/patch 次数
    3. 超预算时冻结 evolution（但不冻结 learning）
    """

    def __init__(self, config: Dict, logger):
        self.config = config
        self.logger = logger
        self.stab_config = config.get('stability', {})

        # 预算
        budget_cfg = self.stab_config.get('budget', {})
        self.max_merge = budget_cfg.get('max_merge', 1)
        self.max_split = budget_cfg.get('max_split', 1)
        self.max_patch = budget_cfg.get('max_patch', 2)

        # 状态
        self.merge_count = 0
        self.split_count = 0
        self.patch_count = 0
        self.evolution_frozen = False
        self.freeze_reason = ""

        # 历史记录（用于 drift 计算）
        self._policy_count_history = deque(maxlen=20)
        self._embedding_history = deque(maxlen=10)   # 存储每轮所有策略的 embedding 均值
        self._entropy_history = deque(maxlen=20)

        # 当前轮次
        self._round = 0

    def reset_budget(self):
        """每轮开始时重置计数"""
        self.merge_count = 0
        self.split_count = 0
        self.patch_count = 0
        self._round += 1

    def can_merge(self) -> bool:
        """是否允许执行 merge"""
        if self.evolution_frozen:
            return False
        return self.merge_count < self.max_merge

    def can_split(self) -> bool:
        if self.evolution_frozen:
            return False
        return self.split_count < self.max_split

    def can_patch(self) -> bool:
        if self.evolution_frozen:
            return False
        return self.patch_count < self.max_patch

    def record_merge(self):
        self.merge_count += 1
        self._check_budget()

    def record_split(self):
        self.split_count += 1
        self._check_budget()

    def record_patch(self):
        self.patch_count += 1
        self._check_budget()

    def _check_budget(self):
        """检查是否超预算，若超则冻结 evolution"""
        if (self.merge_count > self.max_merge or
            self.split_count > self.max_split or
            self.patch_count > self.max_patch):
            self.evolution_frozen = True
            self.freeze_reason = f"超预算: merge={self.merge_count}/{self.max_merge}, split={self.split_count}/{self.max_split}, patch={self.patch_count}/{self.max_patch}"
            self.logger.log(f"   ⚠️ evolution 已冻结: {self.freeze_reason}")

    def update_monitoring(self, policies: List, entropy: float):
        """
        更新监控指标，检测结构变化率
        """
        # 1. Policy 数量变化
        self._policy_count_history.append(len(policies))

        # 2. Embedding drift（计算所有策略 embedding 的均值）
        if policies and hasattr(policies[0], 'embedding'):
            embeddings = [p.embedding for p in policies if p.embedding]
            if embeddings:
                mean_emb = np.mean(embeddings, axis=0)
                self._embedding_history.append(mean_emb)

        # 3. Entropy drift
        self._entropy_history.append(entropy)

    def get_structure_change_rate(self) -> Dict[str, float]:
        """
        计算结构变化率
        """
        rates = {}

        # Policy 数量变化率（最近5轮 vs 之前5轮）
        if len(self._policy_count_history) >= 10:
            recent = np.mean(list(self._policy_count_history)[-5:])
            earlier = np.mean(list(self._policy_count_history)[:5])
            rates['policy_count_change'] = abs(recent - earlier) / (earlier + 0.001)

        # Embedding drift
        if len(self._embedding_history) >= 5:
            # 计算最近两个 embedding 向量的余弦距离变化
            recent = np.mean(list(self._embedding_history)[-3:], axis=0)
            earlier = np.mean(list(self._embedding_history)[:3], axis=0)
            norm_recent = np.linalg.norm(recent)
            norm_earlier = np.linalg.norm(earlier)
            if norm_recent > 0 and norm_earlier > 0:
                cos_sim = np.dot(recent, earlier) / (norm_recent * norm_earlier)
                rates['embedding_drift'] = 1.0 - cos_sim
            else:
                rates['embedding_drift'] = 0.0

        # Entropy drift
        if len(self._entropy_history) >= 10:
            recent = np.mean(list(self._entropy_history)[-5:])
            earlier = np.mean(list(self._entropy_history)[:5])
            rates['entropy_drift'] = abs(recent - earlier) / (earlier + 0.001)

        return rates

    def should_cooldown(self) -> bool:
        """
        判断是否需要进入冷却（基于结构变化率）
        如果变化率低于阈值，解除冻结
        """
        rates = self.get_structure_change_rate()
        change_rate = rates.get('policy_count_change', 1.0)
        threshold = self.config.get('evolution', {}).get('cooldown', {}).get('change_rate_threshold', 0.01)

        # 如果变化率低且已经冻结，解除冻结
        if self.evolution_frozen and change_rate < threshold:
            self.evolution_frozen = False
            self.freeze_reason = ""
            self.logger.log(f"   ✅ evolution 已解冻 (变化率={change_rate:.4f} < {threshold})")
            return False

        # 如果变化率高且未冻结，不进入冷却
        if change_rate >= threshold:
            return False

        # 其他情况：保持当前状态
        return self.evolution_frozen

    def is_frozen(self) -> bool:
        return self.evolution_frozen

    def get_status(self) -> Dict:
        return {
            'round': self._round,
            'merge_count': self.merge_count,
            'split_count': self.split_count,
            'patch_count': self.patch_count,
            'max_merge': self.max_merge,
            'max_split': self.max_split,
            'max_patch': self.max_patch,
            'evolution_frozen': self.evolution_frozen,
            'freeze_reason': self.freeze_reason,
            'policy_count_history': list(self._policy_count_history),
            'entropy_history': list(self._entropy_history),
        }