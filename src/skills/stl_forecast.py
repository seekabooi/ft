import numpy as np
from .base import BaseSkill

class STLForecastSkill(BaseSkill):
    def __init__(self, period=12):
        super().__init__()
        self.name = "stl_forecast"
        self.description = "STL 分解后分别预测趋势和季节分量"
        self.period = period
        self.state_card = {
            "when_to_use": {
                "conditions": [
                    {"field": "seasonal_strength", "op": ">", "value": 0.5},
                    {"field": "trend_strength", "op": ">", "value": 0.3}
                ],
                "logic": "AND"
            },
            "when_not_to_use": {
                "conditions": [
                    {"field": "data_length", "op": "<", "value": 2 * self.period}
                ],
                "logic": "OR"
            },
            "visible_cues": ["季节性明显且趋势持续"],
            "verification_cue": "重组后的残差白噪声",
            "available_views": ["stl_decomposition_plot"],
            "failure_mode": "分解失败或趋势外推发散",
            "fallback_skill": "seasonal_naive"
        }

    def execute(self, history: np.ndarray, horizon: int, **kwargs) -> np.ndarray:
        p = kwargs.get("period", self.period)
        try:
            from statsmodels.tsa.seasonal import STL
            stl = STL(history, period=p)
            res = stl.fit()
            trend = res.trend
            seasonal = res.seasonal
            x = np.arange(len(trend))
            valid = ~np.isnan(trend)
            if valid.sum() < 2:
                trend_pred = np.full(horizon, np.mean(history))
            else:
                x_v = x[valid]
                t_v = trend[valid]
                coeffs = np.polyfit(x_v, t_v, 1)
                future_x = np.arange(len(history), len(history) + horizon)
                trend_pred = np.polyval(coeffs, future_x)
            if len(history) >= p:
                last_period = seasonal[-p:]
                repeats = horizon // p + 1
                seasonal_pred = np.tile(last_period, repeats)[:horizon]
            else:
                seasonal_pred = np.zeros(horizon)
            return trend_pred + seasonal_pred
        except Exception:
            return np.full(horizon, np.mean(history))