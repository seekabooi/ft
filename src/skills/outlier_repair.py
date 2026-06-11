import numpy as np
from .base import BaseSkill

class OutlierRepairSkill(BaseSkill):
    """检测异常点并用插值修正，再交给基技能预测"""
    def __init__(self, base_skill=None, method="iqr", threshold=3.0):
        super().__init__()
        self.name = "outlier_repair"
        self.description = "先修复异常点再预测，适合含离群值的序列"
        self.base_skill = base_skill
        self.method = method      # "iqr" 或 "zscore"
        self.threshold = threshold
        self.state_card = {
            "when_to_use": {
                "conditions": [
                    {"field": "recent_volatility", "op": ">", "value": 1.5}
                ],
                "logic": "AND"
            },
            "when_not_to_use": {
                "conditions": [
                    {"field": "data_length", "op": "<", "value": 10}
                ],
                "logic": "OR"
            },
            "visible_cues": ["序列存在突变点或离群值"],
            "verification_cue": "修复后序列的ACF更平滑",
            "failure_mode": "异常检测错误导致有用信息丢失",
            "fallback_skill": "naive"
        }

    def execute(self, history: np.ndarray, horizon: int, **kwargs) -> np.ndarray:
        if self.base_skill is None:
            return np.full(horizon, np.mean(history))
        try:
            repaired = self._repair_outliers(history)
            return self.base_skill.execute(repaired, horizon, **kwargs)
        except Exception:
            return self.base_skill.execute(history, horizon, **kwargs)

    def _repair_outliers(self, arr: np.ndarray) -> np.ndarray:
        arr = arr.copy()
        if self.method == "iqr":
            q1, q3 = np.percentile(arr, [25, 75])
            iqr = q3 - q1
            lower = q1 - self.threshold * iqr
            upper = q3 + self.threshold * iqr
            mask = (arr < lower) | (arr > upper)
        else:  # zscore
            z = np.abs((arr - np.mean(arr)) / np.std(arr))
            mask = z > self.threshold

        if not np.any(mask):
            return arr

        # 用线性插值修复异常位置
        x = np.arange(len(arr))
        valid_x = x[~mask]
        valid_y = arr[~mask]
        arr[mask] = np.interp(x[mask], valid_x, valid_y)
        return arr