import numpy as np
from scipy import stats
from .base import BaseSkill

class TrendForecasterSkill(BaseSkill):
    def __init__(self):
        super().__init__()
        self.name = "trend_forecaster"
        self.description = "趋势预测器：阻尼 Holt 或线性外推"
        self.min_data_points = 10
        self.requires_full_history = True
        self.strength_tags = ["trend", "forecast"]
        self.model_family = "lightweight"
        self.required_features = ["trend_strength", "data_length"]
        self.decision_hint = (
            "专门预测趋势分量，适合在 DAG 中接收去季节后的趋势序列。"
            "在加权求和模式下，对原始序列进行趋势预测（忽略季节）。"
        )
        self.state_card = {
            "when_to_use": {
                "conditions": [{"field": "trend_strength", "op": ">", "value": 0.15}],
                "logic": "AND"
            },
            "when_not_to_use": {"conditions": [], "logic": "OR"},
            "visible_cues": ["趋势明显"],
            "verification_cue": "趋势线外推合理",
            "fallback_skill": "naive_drift"
        }

    def execute(self, history, horizon, **kwargs):
        n = len(history)
        if n < 10:
            return np.full(horizon, np.mean(history[-5:]))
        # 使用阻尼 Holt 线性趋势（简化版）
        from statsmodels.tsa.holtwinters import Holt
        try:
            model = Holt(history, damped_trend=True).fit()
            return model.forecast(horizon)
        except:
            # 回退线性
            x = np.arange(n)
            slope, intercept, _, _, _ = stats.linregress(x, history)
            future_x = np.arange(n, n + horizon)
            return slope * future_x + intercept