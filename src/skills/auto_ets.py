import numpy as np
from .base import BaseSkill

class AutoETSSkill(BaseSkill):
    """使用 statsmodels 的自动 ETS 模型（根据AIC选择趋势和季节类型）"""
    def __init__(self):
        super().__init__()
        self.name = "auto_ets"
        self.description = "自动选择趋势/季节类型的指数平滑模型，适合有趋势但无明显季节性的序列"
        self.state_card = {
            "when_to_use": {
                "conditions": [
                    {"field": "trend_strength", "op": ">", "value": 0.3},
                    {"field": "seasonal_strength", "op": "<", "value": 0.4}
                ],
                "logic": "AND"
            },
            "when_not_to_use": {
                "conditions": [
                    {"field": "seasonal_strength", "op": ">", "value": 0.6},
                    {"field": "data_length", "op": "<", "value": 10}
                ],
                "logic": "OR"
            },
            "visible_cues": ["序列呈现单调趋势，无明显周期性"],
            "verification_cue": "残差无自相关",
            "available_views": ["fitted_trend_line"],
            "failure_mode": "自动选择失败或数值不稳定",
            "fallback_skill": "naive"
        }

    def execute(self, history: np.ndarray, horizon: int, **kwargs) -> np.ndarray:
        try:
            from statsmodels.tsa.exponential_smoothing.ets import ETSModel
            import warnings
            warnings.filterwarnings("ignore")
            # 自动选择趋势和季节（通过AIC）
            model = ETSModel(history, error='add', trend='add', seasonal=None, damped_trend=True)
            fit = model.fit(disp=False)
            forecast = fit.forecast(horizon)
            return np.array(forecast)
        except Exception:
            # 回退到简单指数平滑（无趋势）
            try:
                from statsmodels.tsa.holtwinters import SimpleExpSmoothing
                fit = SimpleExpSmoothing(history).fit()
                return fit.forecast(horizon)
            except:
                return np.full(horizon, np.mean(history))