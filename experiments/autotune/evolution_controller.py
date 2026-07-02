# experiments/autotune/evolution_controller.py
"""
Evolution Controller（P2 + P13）
演化触发控制器 + Global Cooldown
"""

import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime


class EvolutionController:
    """
    演化控制器

    功能：
    1. Event Trigger（P2）：双触发（时间 OR 异常）
    2. Global Cooldown（P13）：所有演化冷却200窗口
    3. 演化动作互斥（P15）
    """

    def __init__(self, config: Dict, logger):
        self.config = config
        self.logger = logger

        evo_cfg = config.get('evolution', {})
        trigger_cfg = evo_cfg.get('trigger', {})
        cooldown_cfg = evo_cfg.get('cooldown', {})

        # 触发配置
        self.base_interval = trigger_cfg.get('base_interval', 100)
        self.interval_per_policy = trigger_cfg.get('interval_per_policy', 20)
        self.hard_window_threshold = trigger_cfg.get('hard_window_threshold', 0.15)
        self.redundancy_threshold = trigger_cfg.get('redundancy_threshold', 0.6)

        # 冷却配置
        self.global_cooldown_period = cooldown_cfg.get('global_period', 200)
        self._last_global_evolution_step = -self.global_cooldown_period

        # 统计追踪
        self._hard_window_history = []
        self._redundancy_history = []
        self._trigger_count = 0

    def should_trigger(self,
                       current_step: int,
                       policy_count: int,
                       hard_window_ratio: float,
                       redundancy_score: float,
                       evolution_history: List) -> Dict:
        """
        判断是否应该触发演化（P2：双触发）

        触发条件：
            1. 时间触发：step > trigger_interval
            2. 异常触发：hard_window_ratio > 15% 或 redundancy_score > threshold

        Returns:
            {
                'should_trigger': bool,
                'reason': str,
                'trigger_type': 'time' | 'hard_window' | 'redundancy'
            }
        """
        # 计算触发间隔
        trigger_interval = self.base_interval + policy_count * self.interval_per_policy

        # 记录历史
        self._hard_window_history.append(hard_window_ratio)
        self._redundancy_history.append(redundancy_score)

        # 保持历史长度
        if len(self._hard_window_history) > 20:
            self._hard_window_history.pop(0)
        if len(self._redundancy_history) > 20:
            self._redundancy_history.pop(0)

        # 检查冷却
        if (current_step - self._last_global_evolution_step) < self.global_cooldown_period:
            return {'should_trigger': False, 'reason': 'Global cooldown active'}

        # 1. 时间触发
        time_elapsed = current_step - self._last_global_evolution_step
        if time_elapsed >= trigger_interval:
            self._last_global_evolution_step = current_step
            self._trigger_count += 1
            return {
                'should_trigger': True,
                'reason': f'Time trigger: interval={trigger_interval}',
                'trigger_type': 'time'
            }

        # 2. 异常触发：Hard Window Ratio
        if hard_window_ratio > self.hard_window_threshold:
            self._last_global_evolution_step = current_step
            self._trigger_count += 1
            return {
                'should_trigger': True,
                'reason': f'Hard window trigger: ratio={hard_window_ratio:.3f} > {self.hard_window_threshold}',
                'trigger_type': 'hard_window'
            }

        # 3. 异常触发：Redundancy Score
        if redundancy_score > self.redundancy_threshold:
            self._last_global_evolution_step = current_step
            self._trigger_count += 1
            return {
                'should_trigger': True,
                'reason': f'Redundancy trigger: score={redundancy_score:.3f} > {self.redundancy_threshold}',
                'trigger_type': 'redundancy'
            }

        return {'should_trigger': False, 'reason': 'No trigger condition met'}

    # ★ 新增：强制触发，忽略所有条件 ★
    def force_trigger(self) -> Dict:
        """强制触发演化，用于手动或强制模式"""
        return {
            'should_trigger': True,
            'reason': 'Force trigger (manual override)',
            'trigger_type': 'force'
        }

    def is_action_allowed(self, action_type: str, policy_status: str, current_step: int) -> bool:
        """
        检查动作是否允许（P15：动作互斥 + P13：冷却）
        """
        # 检查冷却
        if (current_step - self._last_global_evolution_step) < self.global_cooldown_period:
            return False

        # 不同动作的状态要求
        status_allowed = {
            'merge': ['ACTIVE', 'DEPRECATED'],
            'retire': ['ACTIVE', 'DEPRECATED', 'TRIAL'],
            'patch': ['ACTIVE'],
            'refresh': ['ACTIVE'],
            'reinduction': ['TRIAL']  # 新规则进入TRIAL
        }

        return policy_status in status_allowed.get(action_type, ['ACTIVE'])

    def get_cooldown_remaining(self, current_step: int) -> int:
        """获取剩余冷却窗口数"""
        elapsed = current_step - self._last_global_evolution_step
        remaining = self.global_cooldown_period - elapsed
        return max(0, remaining)

    def get_trigger_stats(self) -> Dict:
        """获取触发统计"""
        return {
            'total_triggers': self._trigger_count,
            'last_trigger_step': self._last_global_evolution_step,
            'cooldown_remaining': self.get_cooldown_remaining(0) if self._last_global_evolution_step > 0 else 0,
            'avg_hard_window_ratio': np.mean(self._hard_window_history[-10:]) if self._hard_window_history else 0,
            'avg_redundancy_score': np.mean(self._redundancy_history[-10:]) if self._redundancy_history else 0
        }

    def record_evolution(self, current_step: int):
        """记录演化发生（更新冷却）"""
        self._last_global_evolution_step = current_step

    def reset(self):
        """重置控制器（用于新数据集）"""
        self._last_global_evolution_step = -self.global_cooldown_period
        self._hard_window_history = []
        self._redundancy_history = []
        self._trigger_count = 0