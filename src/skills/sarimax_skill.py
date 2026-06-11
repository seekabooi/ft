import numpy as np
from .base import BaseSkill

class SARIMAXSkill(BaseSkill):
    """
    季节 ARIMA 模型（支持外生变量，目前外生变量留空）。
    自动选择阶数 (1,1,1)x(1,1,1,period) 作为默认。
    """
    def __init__(self):
        super().__init__()
        self.name = "sarimax"
        self.description = "季节ARIMA模型，适合带有季节性的时间序列"
        self.state_card = {
            "when_to_use": {
                "conditions": [
                    {"field": "seasonal_strength", "op": ">", "value": 0.3},
                    {"field": "data_length", "op": ">=", "value": 2 * 12}  # 至少2个周期
                ],
                "logic": "AND"
            },
            "when_not_to_use": {
                "conditions": [
                    {"field": "data_length", "op": "<", "value": 2 * 12},
                    {"field": "adf_pvalue", "op": "<", "value": 0.01}   # 平稳数据不需要季节差分？
                ],
                "logic": "OR"
            },
            "visible_cues": ["序列呈现季节性波动，且至少包含两个完整周期"],
            "verification_cue": "残差为白噪声（Ljung-Box p>0.05）",
            "available_views": ["acf_plot", "seasonal_decompose"],
            "failure_mode": "模型无法收敛或预测值异常",
            "fallback_skill": "prophet"
        }

    def execute(self, history: np.ndarray, horizon: int, **kwargs) -> np.ndarray:
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            period = kwargs.get("period", 12)
            # 默认使用简单的季节阶数，可根据数据长度调整
            order = (1, 1, 1)
            seasonal_order = (1, 1, 1, period)
            model = SARIMAX(history, order=order, seasonal_order=seasonal_order,
                            enforce_stationarity=False, enforce_invertibility=False)
            fit = model.fit(disp=False, maxiter=50)
            forecast = fit.forecast(steps=horizon)
            return np.array(forecast)
        except Exception:
            # 回退到简单季节朴素
            return np.full(horizon, np.mean(history[-5:]))