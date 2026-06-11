import numpy as np
from .base import BaseSkill

class DecomposeEnsembleSkill(BaseSkill):
    """STL 分解 → 趋势用 ARIMA/线性 → 季节用 SeasonalNaive → 残差用 Naive → 重组"""
    def __init__(self, period=12):
        super().__init__()
        self.name = "decompose_ensemble"
        self.description = "STL分解后分别预测各分量再重组，适合强季节性与趋势混合"
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
            "verification_cue": "重组残差白噪声",
            "available_views": ["stl_decomposition_plot"],
            "failure_mode": "STL分解失败或分量预测异常",
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
            residual = res.resid

            # 趋势外推：线性回归
            x = np.arange(len(trend))
            valid = ~np.isnan(trend)
            if valid.sum() < 2:
                trend_pred = np.full(horizon, np.nanmean(trend) if not np.all(np.isnan(trend)) else np.mean(history))
            else:
                x_v = x[valid]
                t_v = trend[valid]
                coeffs = np.polyfit(x_v, t_v, 1)
                future_x = np.arange(len(history), len(history) + horizon)
                trend_pred = np.polyval(coeffs, future_x)

            # 季节预测：直接用最后一个完整周期
            if len(history) >= p:
                last_period = seasonal[-p:]
                repeats = horizon // p + 1
                seasonal_pred = np.tile(last_period, repeats)[:horizon]
            else:
                seasonal_pred = np.zeros(horizon)

            # 残差预测：用均值（或naive）
            resid_pred = np.full(horizon, np.nanmean(residual) if not np.all(np.isnan(residual)) else 0.0)

            return trend_pred + seasonal_pred + resid_pred
        except Exception:
            return np.full(horizon, np.mean(history))