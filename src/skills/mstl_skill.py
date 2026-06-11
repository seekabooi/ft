import numpy as np
from .base import BaseSkill

class MSTLSkill(BaseSkill):
    """
    多季节分解 (MSTL) 技能，自动处理多个周期（如周、年）。
    默认使用两个季节周期：第一个为传入的 period，第二个为 period*4（或由用户指定）。
    """
    def __init__(self, periods=None):
        super().__init__()
        self.name = "mstl"
        self.description = "多季节分解，适合具有多个周期（如周、年）的日频数据"
        self.periods = periods  # 例如 [7, 365]
        self.state_card = {
            "when_to_use": {
                "conditions": [
                    {"field": "seasonal_strength", "op": ">", "value": 0.5},
                    {"field": "data_length", "op": ">=", "value": 60}
                ],
                "logic": "AND"
            },
            "when_not_to_use": {
                "conditions": [
                    {"field": "data_length", "op": "<", "value": 30}
                ],
                "logic": "OR"
            },
            "visible_cues": ["序列呈现多个季节周期（如周内+年内）"],
            "verification_cue": "残差白噪声且无明显剩余周期",
            "available_views": ["mstl_decomposition_plot"],
            "failure_mode": "分解失败或季节外推不合理",
            "fallback_skill": "stl_forecast"
        }

    def execute(self, history: np.ndarray, horizon: int, **kwargs) -> np.ndarray:
        try:
            from statsmodels.tsa.seasonal import MSTL
            period = kwargs.get("period", 12)
            # 如果没有指定多周期，则使用 period 和 period*2 作为示例
            if self.periods is None:
                periods = [period, min(period * 2, len(history) // 2)]
            else:
                periods = self.periods
            # 过滤掉太长的周期
            periods = [p for p in periods if p < len(history) // 2]
            if not periods:
                periods = [period]

            mstl = MSTL(history, periods=periods)
            res = mstl.fit()
            trend = res.trend
            seasonal = res.seasonal  # 多列季节性分量
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

            # 季节预测：对于每个季节分量，用最后一个完整周期重复
            seasonal_sum = np.zeros(horizon)
            for i in range(seasonal.shape[1]):
                comp = seasonal[:, i]
                last_period = comp[-periods[i]:] if len(comp) >= periods[i] else comp[-periods[i]//2:]
                repeats = horizon // len(last_period) + 1
                seasonal_pred = np.tile(last_period, repeats)[:horizon]
                seasonal_sum += seasonal_pred

            # 残差预测：均值
            resid_pred = np.full(horizon, np.nanmean(residual) if not np.all(np.isnan(residual)) else 0.0)

            return trend_pred + seasonal_sum + resid_pred
        except Exception:
            return np.full(horizon, np.mean(history))