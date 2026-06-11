import numpy as np
from .base import BaseSkill

class FourierSkill(BaseSkill):
    def __init__(self, period=12, n_harmonics=3):
        super().__init__()
        self.name = "fourier"
        self.description = "傅里叶级数拟合季节成分，使用最小二乘估计"
        self.period = period
        self.n_harmonics = n_harmonics
        self.min_data_points = 2 * period
        self.requires_full_history = True
        self.strength_tags = ["season", "trend"]
        self.model_family = "lightweight"
        self.required_features = ["seasonal_strength", "data_length", "period"]
        self.decision_hint = "利用傅里叶项捕捉复杂周期，适合固定周期的季节数据。"
        self.state_card = {
            "when_to_use": {
                "conditions": [
                    {"field": "seasonal_strength", "op": ">", "value": 0.4},
                    {"field": "data_length", "op": ">=", "value": 2 * self.period}
                ],
                "logic": "AND"
            },
            "when_not_to_use": {
                "conditions": [{"field": "data_length", "op": "<", "value": 2 * self.period}],
                "logic": "OR"
            },
            "visible_cues": ["有明显的周期波动"],
            "verification_cue": "残差无显著自相关",
            "fallback_skill": "seasonal_naive"
        }

    def execute(self, history, horizon, **kwargs):
        p = kwargs.get("period", self.period)
        n = len(history)
        if n < 2 * p:
            return np.full(horizon, np.mean(history[-5:]))
        t = np.arange(n)
        X = np.ones((n, 1 + 2 * self.n_harmonics))
        for i in range(1, self.n_harmonics + 1):
            X[:, 2*i-1] = np.sin(2 * np.pi * i * t / p)
            X[:, 2*i]   = np.cos(2 * np.pi * i * t / p)
        try:
            coeff = np.linalg.lstsq(X, history, rcond=None)[0]
        except:
            return np.full(horizon, np.mean(history[-5:]))
        t_future = np.arange(n, n + horizon)
        X_future = np.ones((horizon, 1 + 2 * self.n_harmonics))
        for i in range(1, self.n_harmonics + 1):
            X_future[:, 2*i-1] = np.sin(2 * np.pi * i * t_future / p)
            X_future[:, 2*i]   = np.cos(2 * np.pi * i * t_future / p)
        return X_future @ coeff