import numpy as np
from .base import BaseSkill

class MultiplicativeCombinerSkill(BaseSkill):
    def __init__(self):
        super().__init__()
        self.name = "multiplicative_combiner"
        self.description = "乘法合成器：趋势 × 季节 + 残差（仅限 DAG 模式）"
        self.min_data_points = 1
        self.requires_full_history = False
        self.strength_tags = ["combiner"]
        self.model_family = "combiner"
        self.required_features = []
        self.decision_hint = (
            "将趋势与季节分量相乘，再加残差得到预测。仅用于 DAG 流水线，"
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
        raise NotImplementedError("MultiplicativeCombiner 只能在 DAG 流水线中使用")