import numpy as np
from .base import BaseSkill

class AdditiveCombinerSkill(BaseSkill):
    def __init__(self):
        super().__init__()
        self.name = "additive_combiner"
        self.description = "加法合成器：将趋势、季节、残差预测相加（仅限 DAG 模式）"
        self.min_data_points = 1
        self.requires_full_history = False
        self.strength_tags = ["combiner"]
        self.model_family = "combiner"
        self.required_features = []
        self.decision_hint = (
            "将多个成分预测相加得到最终预测。此技能只能用于 DAG 流水线，"
            "不能单独用于加权求和。"
        )
        self.state_card = {
            "when_to_use": {"conditions": [], "logic": "AND"},
            "when_not_to_use": {"conditions": [], "logic": "OR"},
            "visible_cues": [],
            "verification_cue": "",
            "fallback_skill": "naive"
        }

    def execute(self, history, horizon, **kwargs):
        # 在加权求和模式下不可用，返回 NaN 触发回退
        raise NotImplementedError("AdditiveCombiner 只能在 DAG 流水线中使用")