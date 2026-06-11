import numpy as np
from .base import BaseSkill

class NaiveDriftSkill(BaseSkill):
    def __init__(self):
        super().__init__()
        self.name = "naive_drift"
        self.description = "带漂移的朴素预测，适合无明显季节性但有缓慢漂移的序列"
        self.min_data_points = 5
        self.requires_full_history = False
        self.strength_tags = ["trend"]
        self.model_family = "lightweight"
        self.required_features = ["seasonal_strength", "trend_strength"]
        self.decision_hint = "适合缓慢漂移、无明显季节的序列，计算快。当趋势存在但季节性弱时可用，权重可 0.2~0.5。"
        self.state_card = {
            "when_to_use": {
                "conditions": [
                    {"field": "seasonal_strength", "op": "<", "value": 0.4},
                    {"field": "trend_strength", "op": "<", "value": 0.7}
                ],
                "logic": "AND"
            },
            "when_not_to_use": {
                "conditions": [
                    {"field": "seasonal_strength", "op": ">", "value": 0.5},
                    {"field": "trend_strength", "op": ">", "value": 0.8}
                ],
                "logic": "OR"
            },
            "visible_cues": ["序列无明显季节波动", "长期有缓慢漂移"],
            "verification_cue": "预测值在历史极值范围内",
            "available_views": ["series_with_drift_line"]
        }

    def execute(self, history: np.ndarray, horizon: int, **kwargs) -> np.ndarray:
        n = len(history)
        if n < 2:
            return np.full(horizon, history[-1])
        drift = (history[-1] - history[0]) / (n - 1)
        return np.array([history[-1] + drift * (i + 1) for i in range(horizon)])