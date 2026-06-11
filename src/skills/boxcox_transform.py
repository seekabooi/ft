import numpy as np
from .base import BaseSkill
from scipy import stats

class BoxCoxTransformSkill(BaseSkill):
    """对非负序列做 Box-Cox 变换，预测后再反变换"""
    def __init__(self, base_skill=None):
        super().__init__()
        self.name = "boxcox_transform"
        self.description = "Box-Cox变换后使用基技能预测，再反变换，适合方差随水平增大的序列"
        self.base_skill = base_skill  # 由外部注入，如 ARIMA、ETS
        self.state_card = {
            "when_to_use": {
                "conditions": [
                    {"field": "recent_volatility", "op": ">", "value": 1.0},
                    {"field": "missing_rate", "op": "==", "value": 0.0}
                ],
                "logic": "AND"
            },
            "when_not_to_use": {
                "conditions": [
                    {"field": "data_length", "op": "<", "value": 10}
                ],
                "logic": "OR"
            },
            "visible_cues": ["序列方差随水平增大而增大"],
            "verification_cue": "变换后残差更稳定",
            "failure_mode": "Box-Cox变换失败或基技能异常",
            "fallback_skill": "naive"
        }

    def execute(self, history: np.ndarray, horizon: int, **kwargs) -> np.ndarray:
        if self.base_skill is None:
            return np.full(horizon, np.mean(history))
        try:
            # Box-Cox 需要严格正值，加上偏移量
            offset = 0.0
            min_val = np.min(history)
            if min_val <= 0:
                offset = abs(min_val) + 1.0
            shifted = history + offset
            # 自动寻找最佳 lambda
            transformed, lamb = stats.boxcox(shifted)
            # 用基技能预测变换后的序列
            base_forecast = self.base_skill.execute(transformed, horizon, **kwargs)
            # 反变换 (逆 Box-Cox)
            if lamb == 0:
                restored = np.exp(base_forecast) - offset
            else:
                restored = (lamb * base_forecast + 1) ** (1.0 / lamb) - offset
            return np.array(restored)
        except Exception:
            # 直接回退到基技能
            return self.base_skill.execute(history, horizon, **kwargs)