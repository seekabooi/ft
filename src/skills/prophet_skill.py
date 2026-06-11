import numpy as np
import pandas as pd
from .base import BaseSkill

class ProphetSkill(BaseSkill):
    def __init__(self):
        super().__init__()
        self.name = "prophet"
        self.description = "Facebook Prophet 模型，擅长捕捉多周期季节性和趋势"
        self.min_data_points = 30
        self.requires_full_history = True
        self.strength_tags = ["trend", "season"]
        self.model_family = "stat_model"
        self.required_features = ["seasonal_strength", "data_length", "adf_pvalue"]
        self.decision_hint = "适合趋势+季节的长序列，鲁棒性强，但计算较慢。当数据长度≥30且季节强度>0.2时推荐作为主模型，权重可 0.4~0.7。"
        self.state_card = {
            "when_to_use": {
                "conditions": [
                    {"field": "seasonal_strength", "op": ">", "value": 0.15},
                    {"field": "data_length", "op": ">=", "value": 20}
                ],
                "logic": "AND"
            },
            "when_not_to_use": {
                "conditions": [
                    {"field": "adf_pvalue", "op": "<", "value": 0.001}
                ],
                "logic": "OR"
            },
            "visible_cues": ["可能存在季节性"],
            "verification_cue": "残差为白噪声",
            "available_views": ["prophet_components_plot"]
        }

    def execute(self, history: np.ndarray, horizon: int, **kwargs) -> np.ndarray:
        try:
            from prophet import Prophet
            freq = 'MS' if len(history) <= 500 else 'D'
            df = pd.DataFrame({
                'ds': pd.date_range(end='today', periods=len(history), freq=freq),
                'y': history
            })
            #m = Prophet()
            m = Prophet(seasonality_mode='multiplicative')
            m.fit(df)
            future = m.make_future_dataframe(periods=horizon, freq=freq)
            forecast = m.predict(future)
            return forecast['yhat'].values[-horizon:]
        except ImportError:
            # 回退到季节性朴素
            period = kwargs.get('period', 12)
            if len(history) < period:
                return np.full(horizon, np.mean(history[-5:]))
            preds = [history[-(period - i % period)] for i in range(horizon)]
            return np.array(preds)
        except Exception:
            return np.full(horizon, np.mean(history[-5:]))