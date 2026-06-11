import numpy as np
from .base import BaseSkill

class ExponentialSmoothingSkill(BaseSkill):
    def __init__(self):
        super().__init__()
        self.name = "ets"
        self.description = "指数平滑模型（ETS），适合有趋势但无明显季节性的序列"
        self.state_card = {
            "when_to_use": {
                "conditions": [
                    {"field": "trend_strength", "op": ">", "value": 0.4},
                    {"field": "seasonal_strength", "op": "<", "value": 0.3}
                ],
                "logic": "AND"
            },
            "when_not_to_use": {
                "conditions": [
                    {"field": "seasonal_strength", "op": ">", "value": 0.5}
                ],
                "logic": "OR"
            },
            "visible_cues": ["序列有趋势但无周期性", "波动平稳"],
            "verification_cue": "残差无自相关",
            "available_views": ["fitted_values_overlay"]
        }

    def execute(self, history: np.ndarray, horizon: int, **kwargs) -> np.ndarray:
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            # 自动选择趋势（additive）
            model = ExponentialSmoothing(history, trend='add', seasonal=None)
            fit = model.fit()
            return fit.forecast(horizon)
        except Exception:
            return np.full(horizon, np.mean(history))