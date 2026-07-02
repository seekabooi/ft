# experiments/autotune/freeze_guard.py
"""
Freeze Guard（P9 + P10）
系统冻结保护：连续3轮Validation Score下降 + 无Coverage Gap + 无Hard Window增加 → 冻结
"""

import numpy as np
from typing import Dict, List, Optional, Any
from collections import deque


class FreezeGuard:
    """
    系统冻结保护（P9 + P10）

    触发条件：
        1. 连续3轮 Validation Score 下降
        2. 且 Coverage Gap 不存在
        3. 且 Hard Window 没有增加

    如果满足，进入 freeze_mode：
        - 只训练，不演化
        - 持续1000窗口后恢复
    """

    def __init__(self, config: Dict, logger):
        self.config = config
        self.logger = logger

        fg_cfg = config.get('freeze_guard', {})
        self.consecutive_decline_threshold = fg_cfg.get('consecutive_decline_threshold', 3)
        self.freeze_duration = fg_cfg.get('freeze_duration', 1000)

        # 状态追踪
        self._score_history = deque(maxlen=10)
        self._hard_window_history = deque(maxlen=10)
        self._freeze_start_step = 0
        self._freeze_count = 0
        self._is_frozen = False
        self._decline_count = 0

    def check(self,
              current_step: int,
              validation_score: float,
              hard_window_ratio: float,
              coverage_gap_exists: bool) -> Dict:
        """
        检查是否需要冻结

        Returns:
            {
                'should_freeze': bool,
                'should_unfreeze': bool,
                'is_frozen': bool,
                'reason': str,
                'freeze_remaining': int
            }
        """
        # 如果当前已冻结，检查是否应该解冻
        if self._is_frozen:
            freeze_elapsed = current_step - self._freeze_start_step

            if freeze_elapsed >= self.freeze_duration:
                self._is_frozen = False
                self._freeze_start_step = 0
                self._decline_count = 0
                self.logger.log(f"   ✅ Freeze guard: 冻结期结束 ({self.freeze_duration}窗口)，已解冻")
                return {
                    'should_freeze': False,
                    'should_unfreeze': True,
                    'is_frozen': False,
                    'reason': 'Freeze duration expired',
                    'freeze_remaining': 0
                }

            return {
                'should_freeze': False,
                'should_unfreeze': False,
                'is_frozen': True,
                'reason': f'Freeze active, {self.freeze_duration - freeze_elapsed} windows remaining',
                'freeze_remaining': self.freeze_duration - freeze_elapsed
            }

        # 记录历史
        self._score_history.append(validation_score)
        self._hard_window_history.append(hard_window_ratio)

        if len(self._score_history) < self.consecutive_decline_threshold:
            return {
                'should_freeze': False,
                'should_unfreeze': False,
                'is_frozen': False,
                'reason': 'Insufficient history',
                'freeze_remaining': 0
            }

        # 检查是否连续下降（P10）
        recent_scores = list(self._score_history)[-self.consecutive_decline_threshold:]
        is_declining = all(
            recent_scores[i] < recent_scores[i-1]
            for i in range(1, len(recent_scores))
        )

        if not is_declining:
            self._decline_count = 0
            return {
                'should_freeze': False,
                'should_unfreeze': False,
                'is_frozen': False,
                'reason': 'No consecutive decline',
                'freeze_remaining': 0
            }

        # 检查 Hard Window 是否增加
        recent_hard = list(self._hard_window_history)[-self.consecutive_decline_threshold:]
        hard_increased = all(
            recent_hard[i] >= recent_hard[i-1]
            for i in range(1, len(recent_hard))
        )

        # 检查 Coverage Gap
        has_gap = coverage_gap_exists

        # 判断是否应该冻结（P10）
        # 只有在：下降 + 无Coverage Gap + Hard Window没有增加 时才冻结
        if is_declining and not has_gap and not hard_increased:
            self._is_frozen = True
            self._freeze_start_step = current_step
            self._freeze_count += 1

            self.logger.log(f"   ⚠️ Freeze guard: 系统冻结！")
            self.logger.log(f"      连续 {self.consecutive_decline_threshold} 轮下降，无Coverage Gap，无Hard Window增加")
            self.logger.log(f"      将冻结 {self.freeze_duration} 窗口")

            return {
                'should_freeze': True,
                'should_unfreeze': False,
                'is_frozen': True,
                'reason': 'Consecutive decline + no gap + no hard window increase',
                'freeze_remaining': self.freeze_duration
            }

        self._decline_count += 1

        return {
            'should_freeze': False,
            'should_unfreeze': False,
            'is_frozen': False,
            'reason': f'Declining but has gap or hard window increasing (decline_count={self._decline_count})',
            'freeze_remaining': 0
        }

    def is_frozen(self) -> bool:
        """检查是否冻结"""
        return self._is_frozen

    def get_status(self) -> Dict:
        """获取冻结状态"""
        return {
            'is_frozen': self._is_frozen,
            'freeze_count': self._freeze_count,
            'freeze_start_step': self._freeze_start_step,
            'decline_count': self._decline_count,
            'score_history': list(self._score_history),
            'hard_window_history': list(self._hard_window_history)
        }

    def force_unfreeze(self):
        """强制解冻"""
        if self._is_frozen:
            self._is_frozen = False
            self._freeze_start_step = 0
            self._decline_count = 0
            self.logger.log("   ✅ Freeze guard: 强制解冻")