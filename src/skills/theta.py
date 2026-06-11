import numpy as np
from .base import BaseSkill

class ThetaSkill(BaseSkill):
    def __init__(self):
        super().__init__()
        self.name = "theta"
        self.description = "Theta 方法，适合短期波动预测"
        self.min_data_points = 10
        self.requires_full_history = False
        self.strength_tags = ["trend"]
        self.model_family = "lightweight"
        self.required_features = ["trend_strength", "data_length"]
        self.decision_hint = "短期趋势预测，适合有趋势但季节弱的数据。当趋势强度>0.3 时可用，权重可 0.2~0.4。"
        self.state_card = {
            "when_to_use": {
                "conditions": [{"field": "trend_strength", "op": ">", "value": 0.3},
                               {"field": "data_length", "op": ">=", "value": 10}],
                "logic": "AND"
            },
            "when_not_to_use": {
                "conditions": [{"field": "seasonal_strength", "op": ">", "value": 0.5}],
                "logic": "OR"
            },
            "visible_cues": ["有趋势，短期波动"],
            "verification_cue": "预测误差小于简单移动平均",
            "fallback_skill": "naive"
        }

    def execute(self, history, horizon, **kwargs):
        try:
            from statsmodels.tsa.forecasting.theta import ThetaModel
            model = ThetaModel(history)
            fit = model.fit()
            return fit.forecast(horizon)
        except:
            x = np.arange(len(history))
            coeffs = np.polyfit(x[-20:], history[-20:], 1)
            future_x = np.arange(len(history), len(history) + horizon)
            return np.polyval(coeffs, future_x)