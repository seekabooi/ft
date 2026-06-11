import numpy as np
from .base import BaseSkill

class LocalDriftSkill(BaseSkill):
    def __init__(self, window=5):
        super().__init__()
        self.name = "local_drift"
        self.description = "用最近 window 个点的线性斜率外推"
        self.window = window
        self.min_data_points = 5
        self.requires_full_history = False
        self.strength_tags = ["trend"]
        self.model_family = "lightweight"
        self.required_features = ["local_slope", "recent_volatility", "change_point_detected"]
        self.decision_hint = "局部趋势外推，适合近期出现明显上升或下降但无突变的场景。当局部斜率较大且波动平稳时可用，权重可 0.1~0.3。"
        self.state_card = {
            "when_to_use": {
                "conditions": [
                    {"field": "local_slope", "op": ">", "value": 0.01},
                    {"field": "recent_volatility", "op": "<", "value": 2.0}
                ],
                "logic": "AND"
            },
            "when_not_to_use": {
                "conditions": [{"field": "change_point_detected", "op": "==", "value": True}],
                "logic": "OR"
            },
            "visible_cues": ["局部斜率较大，近期波动平稳"],
            "verification_cue": "预测值沿趋势方向",
            "fallback_skill": "naive_drift"
        }

    def execute(self, history, horizon, **kwargs):
        w = min(self.window, len(history))
        x = np.arange(w)
        y = history[-w:]
        coeffs = np.polyfit(x, y, 1)
        last_x = w - 1
        future_x = np.arange(last_x + 1, last_x + 1 + horizon)
        return np.polyval(coeffs, future_x)