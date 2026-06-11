import numpy as np
from .base import BaseSkill

class HoltWintersSkill(BaseSkill):
    def __init__(self, period=12):
        super().__init__()
        self.name = "holt_winters"
        self.description = "Holt-Winters 三指数平滑，支持趋势和季节"
        self.period = period
        self.min_data_points = 2 * period
        self.requires_full_history = True
        self.strength_tags = ["trend", "season"]
        self.model_family = "stat_model"
        self.required_features = ["seasonal_strength", "trend_strength", "data_length", "period"]
        self.decision_hint = (
            "擅长趋势+季节建模，是强季节序列的首选统计模型。"
            "当季节强度>0.5 且数据充足时应作为主模型，权重 0.5~0.8；若近期波动大则降低。"
        )
        self.state_card = {
            "when_to_use": {
                "conditions": [
                    {"field": "seasonal_strength", "op": ">", "value": 0.3},
                    {"field": "data_length", "op": ">=", "value": 2 * self.period}
                ],
                "logic": "AND"
            },
            "when_not_to_use": {
                "conditions": [
                    {"field": "adf_pvalue", "op": "<", "value": 0.01}
                ],
                "logic": "OR"
            },
            "visible_cues": ["有季节性，趋势平稳"],
            "verification_cue": "残差为白噪声",
            "fallback_skill": "seasonal_naive"
        }

    def execute(self, history, horizon, **kwargs):
        p = kwargs.get("period", self.period)
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            model = ExponentialSmoothing(history, trend='add', seasonal='add', seasonal_periods=p)
            fit = model.fit()
            return fit.forecast(horizon)
        except:
            return np.full(horizon, np.mean(history[-5:]))