import numpy as np
from .base import BaseSkill

class AutoARIMASkill(BaseSkill):
    """使用 pmdarima.auto_arima 自动选择参数的 ARIMA 模型"""
    def __init__(self):
        super().__init__()
        self.name = "auto_arima"
        self.description = "自动选择阶数的ARIMA模型，适合平稳或可差分平稳序列"
        self.min_data_points = 15
        self.requires_full_history = True
        self.strength_tags = ["trend", "season"]
        self.model_family = "stat_model"
        self.required_features = ["adf_pvalue", "seasonal_strength", "data_length"]
        self.decision_hint = "自动定阶的 ARIMA，适合中等长度非季节序列。当季节性较弱且数据>=30时可用，计算较慢，权重可 0.3~0.5。"
        self.state_card = {
            "when_to_use": {
                "conditions": [
                    {"field": "adf_pvalue", "op": "<", "value": 0.1},
                    {"field": "seasonal_strength", "op": "<", "value": 0.5},
                    {"field": "data_length", "op": ">=", "value": 30}
                ],
                "logic": "AND"
            },
            "when_not_to_use": {
                "conditions": [
                    {"field": "seasonal_strength", "op": ">", "value": 0.7},
                    {"field": "data_length", "op": "<", "value": 15},
                    # 新增：长序列且强季节性时禁用
                    {"field": "seasonal_strength", "op": ">", "value": 0.6, "data_length": ">300"}
                ],
                "logic": "OR"
            },
            "visible_cues": ["序列无明显周期性", "ACF衰减较慢"],
            "verification_cue": "残差Ljung-Box检验p>0.05",
            "available_views": ["acf_plot", "residual_diagnosis"],
            "failure_mode": "模型拟合失败或预测值异常",
            "fallback_skill": "naive"
        }

    def execute(self, history: np.ndarray, horizon: int, **kwargs) -> np.ndarray:
        try:
            import pmdarima as pm
            import warnings
            warnings.filterwarnings("ignore")
            model = pm.auto_arima(
                history,
                start_p=0, max_p=3,
                start_q=0, max_q=3,
                seasonal=False,
                stepwise=True,
                suppress_warnings=True,
                error_action='ignore',
                maxiter=10,
                trace=False
            )
            forecast = model.predict(n_periods=horizon)
            return np.array(forecast)
        except ImportError:
            return np.full(horizon, np.mean(history[-5:]))
        except Exception:
            return np.full(horizon, np.mean(history))