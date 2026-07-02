# experiments/autotune/online_evolver.py
"""
Policy Feedback Trigger System - 稳定性修复版本（已禁用）
"""

import json
import re
from typing import Dict, List, Optional, Any
import numpy as np
from datetime import datetime

from experiments.autotune.utils import ProgressLogger
from experiments.autotune.skill_policy import SkillPolicy


class PolicyFeedbackTrigger:
    """策略反馈触发器 - 只做诊断，不触发演化"""

    def __init__(self, config: Dict, logger: ProgressLogger):
        self.config = config
        self.logger = logger
        self.oe_config = config.get('online_evolution', {})
        self.enabled = self.oe_config.get('enabled', False)
        self.history = []

    def should_trigger(self, current_mase: float, avg_mase: float) -> bool:
        return False

    def diagnose(self, window_data: Dict, policies: List[SkillPolicy]) -> Dict:
        features = window_data.get('features', {})
        current_mase = window_data.get('mase', 1.0)
        matched = None
        for policy in policies:
            if policy.is_applicable(features):
                matched = policy
                break
        return {
            'current_mase': current_mase,
            'matched_policy': matched.name if matched else None,
            'policy_count': len(policies),
            'trigger_disabled': True,
            'reason': 'Online evolution disabled for stability'
        }