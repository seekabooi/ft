import numpy as np
from src.skills.base import BaseSkill
from src.skills.data_profiler import DataProfiler

class SeasonalNaiveSkill(BaseSkill):
    def __init__(self, period=12):
        super().__init__()
        self.name = "seasonal_naive"
        self.description = "季节性朴素预测：使用上一个季节周期的值作为预测"
        self.min_data_points = 2 * period
        self.requires_full_history = False
        self.model_family = "lightweight"
        self.strength_tags = ["强季节性", "稳健", "基准模型"]
        self.required_features = ["period", "seasonal_strength", "data_length"]
        self._default_period = period
        self.decision_hint = "适用于具有稳定季节周期的序列，特别是数据量充足时。"

    def execute(self, history: np.ndarray, horizon: int, period=None, **kwargs) -> np.ndarray:
        # 已移除调试打印
        n = len(history)
        if period is None or period <= 0:
            freq = kwargs.get('freq', None)
            period = DataProfiler._auto_period(history, freq=freq)
            if period <= 0:
                period = self._default_period
        if period >= n:
            period = max(1, n // 2)
        if n >= period:
            last_cycle = history[-period:]
            if horizon <= period:
                forecast = last_cycle[:horizon]
            else:
                repeats = (horizon + period - 1) // period
                forecast = np.tile(last_cycle, repeats)[:horizon]
        else:
            forecast = np.full(horizon, history[-1])
        return forecast.astype(float)