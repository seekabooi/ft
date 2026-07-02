# experiments/autotune/skill_lifecycle.py
"""
技能生命周期管理
Create → Use → Evaluate → Evolve → Retire
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np


class SkillLifecycle:
    """技能生命周期管理器"""

    def __init__(self, config: Dict, logger):
        self.config = config
        self.logger = logger
        self.retire_days_threshold = config.get('retire_days_threshold', 30)
        self.min_usage_threshold = config.get('min_usage_threshold', 5)
        self.degradation_threshold = config.get('degradation_threshold', 0.15)

    def should_retire(self, policy, recent_performance: List[float]) -> Tuple[bool, str]:
        """
        判断是否应该退休

        条件：
        1. 长期未使用（> retire_days_threshold 天）
        2. 使用次数过少（< min_usage_threshold）
        3. 性能持续恶化（最近N次MASE上升 > degradation_threshold）
        """
        # 条件1：长期未使用
        if policy.last_used:
            try:
                last = datetime.fromisoformat(policy.last_used)
                if (datetime.now() - last).days > self.retire_days_threshold:
                    return True, f"未使用超过 {self.retire_days_threshold} 天"
            except:
                pass

        # 条件2：使用次数过少
        if policy.usage_count < self.min_usage_threshold:
            return True, f"使用次数 {policy.usage_count} < {self.min_usage_threshold}"

        # 条件3：性能持续恶化
        if len(recent_performance) >= 3:
            # 检查最近3次是否连续上升（恶化）
            if all(recent_performance[i] > recent_performance[i - 1]
                   for i in range(1, len(recent_performance))):
                # 计算恶化幅度
                degradation = (recent_performance[-1] - recent_performance[0]) / (recent_performance[0] + 0.01)
                if degradation > self.degradation_threshold:
                    return True, f"性能恶化 {degradation:.1%}"

        return False, ""

    def should_refine(self, policy, performance_report: Dict) -> Tuple[bool, str]:
        """
        判断是否需要精炼

        条件：
        1. 在某些子场景下效果显著下降
        2. 条件过于宽泛（覆盖太多异质窗口）
        """
        window_details = performance_report.get('window_details', [])
        if not window_details:
            return False, ""

        # 检查窗口间的MASE方差
        mases = [w.get('mase', 1.0) for w in window_details if w.get('matched_rule') == policy.name]
        if len(mases) < 3:
            return False, ""

        variance = np.var(mases)
        if variance > 0.25:
            return True, f"窗口间MASE方差 {variance:.3f} > 0.25"

        return False, ""

    def transition(self, policy, new_stage: str) -> str:
        """
        状态转移

        Stages: created → active → deprecated → retired
        """
        valid_transitions = {
            'created': ['active', 'retired'],
            'active': ['active', 'deprecated', 'retired'],
            'deprecated': ['active', 'retired'],
            'retired': []
        }

        if new_stage not in valid_transitions.get(policy.stage, []):
            self.logger.log(f"   ⚠️ 无效状态转移: {policy.stage} → {new_stage}")
            return policy.stage

        policy.stage = new_stage
        self.logger.log(f"   🔄 技能 {policy.name}: {policy.stage} → {new_stage}")
        return new_stage