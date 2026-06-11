import numpy as np
from .base import BaseSkill

class ETSSkill(BaseSkill):
    def __init__(self):
        super().__init__()
        self.name = "ets"
        self.description = "指数平滑（ETS），自动选择趋势/季节类型（支持季节性）"
        self.min_data_points = 10
        self.requires_full_history = True
        self.strength_tags = ["trend", "season"]
        self.model_family = "stat_model"
        self.required_features = ["seasonal_strength", "data_length"]
        self.decision_hint = "指数平滑，适合有趋势或季节性的序列。季节强时会自动启用加法季节性。"
        self.state_card = {
            "when_to_use": {
                "conditions": [
                    {"field": "seasonal_strength", "op": "<", "value": 0.7},
                    {"field": "data_length", "op": ">=", "value": 10}
                ],
                "logic": "AND"
            },
            "when_not_to_use": {
                "conditions": [{"field": "data_length", "op": "<", "value": 10}],
                "logic": "OR"
            },
            "visible_cues": ["有趋势或弱季节性"],
            "verification_cue": "残差无自相关",
            "fallback_skill": "naive_drift"
        }

    def execute(self, history: np.ndarray, horizon: int, **kwargs) -> np.ndarray:
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            n = len(history)
            period = kwargs.get('period', 12)
            seasonal_strength = kwargs.get('seasonal_strength', 0.0)
            if seasonal_strength > 0.5 and n >= 2 * period:
                model = ExponentialSmoothing(history, trend='add', seasonal='add', seasonal_periods=period)
            else:
                model = ExponentialSmoothing(history, trend='add', seasonal=None)
            fit = model.fit()
            return fit.forecast(horizon)
        except:
            return np.full(horizon, np.mean(history[-5:]))