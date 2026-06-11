import numpy as np
from .base import BaseSkill

class ARIMASkill(BaseSkill):
    def __init__(self):
        super().__init__()
        self.name = "arima"
        self.description = "ARIMA(1,1,1)，适合平稳或低季节性序列"
        self.min_data_points = 10
        self.requires_full_history = True
        self.strength_tags = ["trend"]
        self.model_family = "stat_model"
        self.required_features = ["adf_pvalue", "seasonal_strength"]
        self.decision_hint = "适合平稳或低季节性序列，固定阶数(1,1,1)快速拟合。当 ADF p<0.05 且季节弱时适用，权重可 0.3~0.6。"
        self.state_card = {
            "when_to_use": {
                "conditions": [
                    {"field": "adf_pvalue", "op": "<", "value": 0.05},
                    {"field": "seasonal_strength", "op": "<", "value": 0.4}
                ], "logic": "AND"
            },
            "when_not_to_use": {
                "conditions": [
                    {"field": "seasonal_strength", "op": ">", "value": 0.7}
                ], "logic": "OR"
            },
            "visible_cues": ["序列无明显周期性", "ACF 在 lag=1 后衰减"],
            "verification_cue": "残差 Ljung-Box 检验 p>0.05",
            "available_views": ["raw_series_last30", "acf_plot"]
        }

    def execute(self, history, horizon, **kwargs):
        try:
            from statsmodels.tsa.arima.model import ARIMA
            model = ARIMA(history, order=(1,1,1))
            fit = model.fit()
            return fit.forecast(steps=horizon)
        except:
            return np.full(horizon, np.mean(history))