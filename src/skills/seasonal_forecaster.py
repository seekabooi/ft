import numpy as np
from .base import BaseSkill

class SeasonalForecasterSkill(BaseSkill):
    def __init__(self, period=12):
        super().__init__()
        self.name = "seasonal_forecaster"
        self.description = "季节预测器：傅里叶外推或日历基线"
        self.period = period
        self.min_data_points = 2 * period
        self.requires_full_history = True
        self.strength_tags = ["season", "forecast"]
        self.model_family = "lightweight"
        self.required_features = ["seasonal_strength", "data_length", "period"]
        self.decision_hint = (
            "专门预测季节分量，适合在 DAG 中接收去趋势后的季节序列。"
            "在加权求和模式下，对原始序列进行季节朴素预测。"
        )
        self.state_card = {
            "when_to_use": {
                "conditions": [
                    {"field": "seasonal_strength", "op": ">", "value": 0.2},
                    {"field": "data_length", "op": ">=", "value": 2 * self.period}
                ],
                "logic": "AND"
            },
            "when_not_to_use": {"conditions": [], "logic": "OR"},
            "visible_cues": ["季节模式稳定"],
            "verification_cue": "季节预测保持周期性",
            "fallback_skill": "seasonal_naive"
        }

    def execute(self, history, horizon, **kwargs):
        p = kwargs.get("period", self.period)
        if len(history) < p:
            return np.full(horizon, np.mean(history[-5:]))
        # 使用傅里叶级数拟合季节成分（仅季节，不含趋势）
        # 为简单起见，这里用季节朴素作为默认输出，实际可替换为傅里叶
        preds = []
        for i in range(horizon):
            idx = -(p - i % p)
            preds.append(history[idx])
        return np.array(preds)