import numpy as np
from statsmodels.tsa.seasonal import STL
from src.skills.base import BaseSkill
from src.skills.data_profiler import DataProfiler

class STLDecomposeSkill(BaseSkill):
    def __init__(self):
        super().__init__()
        self.name = "stl_decompose"
        self.description = "STL 分解后分别预测再组合"
        self.min_data_points = 30
        self.requires_full_history = False
        self.model_family = "heavy"
        self.strength_tags = ["长期趋势", "强季节性", "长序列"]
        self.required_features = ["seasonal_strength", "trend_strength", "period", "data_length"]
        self.preferred_length_range = (100, float('inf'))
        self.decision_hint = "适用于长度至少2个周期的强季节性或明显趋势序列，对长序列尤其有效。"

    def execute(self, history: np.ndarray, horizon: int, period=None, **kwargs) -> np.ndarray:
        n = len(history)
        if n < self.min_data_points:
            if period is None:
                period = 12
            if n >= period:
                return np.array([history[-period]] * horizon)
            return np.full(horizon, np.mean(history))

        # 确定周期
        if period is None or period <= 0:
            freq = kwargs.get('freq', None)
            period = DataProfiler._auto_period(history, freq=freq)
            if period <= 0:
                period = 12
        # 防止周期过大
        if period >= n:
            period = max(1, n // 2)

        try:
            stl = STL(history, period=period, robust=True)
            result = stl.fit()
            trend = result.trend
            seasonal = result.seasonal
            resid = result.resid
        except Exception:
            # 分解失败，回退到季节性朴素
            if n >= period:
                last_cycle = history[-period:]
                repeats = (horizon + period - 1) // period
                return np.tile(last_cycle, repeats)[:horizon]
            return np.full(horizon, history[-1])

        # 趋势预测：线性外推
        if len(trend) >= 5:
            x = np.arange(5)
            y = trend[-5:]
            slope, intercept = np.polyfit(x, y, 1)
            trend_forecast = np.array([intercept + slope * (len(trend) + i) for i in range(horizon)])
        else:
            trend_forecast = np.full(horizon, trend[-1])

        # 季节预测：最后一个周期模式
        full_cycles = n // period
        if full_cycles >= 1:
            seasonal_pattern = seasonal[-period:] if len(seasonal) >= period else seasonal
        else:
            seasonal_pattern = seasonal
        seasonal_forecast = np.tile(seasonal_pattern, (horizon // period + 1))[:horizon]

        # 残差：均值
        resid_mean = np.nanmean(resid) if not np.all(np.isnan(resid)) else 0.0
        resid_forecast = np.full(horizon, resid_mean)

        forecast = trend_forecast + seasonal_forecast + resid_forecast
        if np.min(history) >= 0:
            forecast = np.maximum(forecast, 0)
        return forecast