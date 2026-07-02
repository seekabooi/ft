# experiments/autotune/hard_refiner.py
"""
Local Policy Repair Engine - 简化版（当前已冻结）
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from experiments.autotune.utils import load_window_data, extract_features, compute_mase
from experiments.autotune.skill_policy import SkillPolicy


class LocalPolicyRepairEngine:
    """本地策略修复引擎 - 当前已冻结"""

    def __init__(self, config: Dict, logger):
        self.config = config
        self.logger = logger

    def repair(self, collected_data: pd.DataFrame,
               hard_window_ids: List[int],
               policies: List[SkillPolicy]) -> List[SkillPolicy]:
        """当前已冻结，返回空列表"""
        self.logger.log("ℹ️ 策略修复已冻结，不执行补丁生成")
        return []