import numpy as np
from .base import BaseSkill

class LinearTrendSkill(BaseSkill):
    """简单线性趋势外推：拟合最近一段窗口的线性回归并外推"""
    def __init__(self, window=20):
        super().__init__()
        self.name = "linear_trend"
        self.description = "对最近窗口做线性回归外推，适合有明显单调趋势且无季节性的序列"
        self.window = window
        self.state_card = {
            "when_to_use": {
                "conditions": [
                    {"field": "trend_strength", "op": ">", "value": 0.6},
                    {"field": "seasonal_strength", "op": "<", "value": 0.3}
                ],
                "logic": "AND"
            },
            "when_not_to_use": {
                "conditions": [
                    {"field": "seasonal_strength", "op": ">", "value": 0.5}
                ],
                "logic": "OR"
            },
            "visible_cues": ["序列呈现持续上升或下降趋势"],
            "verification_cue": "残差无明显自相关",
            "available_views": ["trend_fit_plot"]
        }

    def execute(self, history: np.ndarray, horizon: int, **kwargs) -> np.ndarray:
        w = min(self.window, len(history))
        x = np.arange(w)
        y = history[-w:]
        coeffs = np.polyfit(x, y, 1)
        future_x = np.arange(w, w + horizon)
        return np.polyval(coeffs, future_x)