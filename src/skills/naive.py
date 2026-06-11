import numpy as np
from .base import BaseSkill

class NaiveSkill(BaseSkill):
    def __init__(self):
        super().__init__()
        self.name = "naive"
        self.description = "简单移动平均（最后5点均值）"
        self.min_data_points = 3
        self.requires_full_history = False
        self.strength_tags = []
        self.model_family = "lightweight"
        self.required_features = ["data_length"]
        self.decision_hint = "简单基准，计算极快，适合任何平稳序列，可作为保底或平滑项，权重建议 0.1~0.3。"
        self.state_card = {
            "when_to_use": {"conditions": [], "logic": "AND"},
            "when_not_to_use": {"conditions": [], "logic": "OR"},
            "visible_cues": ["任意序列均可作为基准"],
            "verification_cue": "预测值接近历史均值",
            "fallback_skill": "naive_drift"
        }

    def execute(self, history, horizon, **kwargs):
        n = min(5, len(history))
        return np.full(horizon, np.mean(history[-n:]))