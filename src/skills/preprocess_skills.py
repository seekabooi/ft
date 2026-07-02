import numpy as np
from scipy import stats
import pandas as pd
from src.skills.base import BaseSkill


class FillMissing(BaseSkill):
    def __init__(self):
        super().__init__()
        self.name = "fill_missing"
        self.description = "线性插值填充缺失值"
        self.min_data_points = 3

    def execute(self, history: np.ndarray, horizon: int = 0, **kwargs) -> np.ndarray:
        if not np.isnan(history).any():
            return history.copy()
        s = pd.Series(history)
        s = s.interpolate(method='linear', limit_direction='both')
        return s.values

    def execute_with_context(self, history: np.ndarray):
        filled = self.execute(history)
        n_filled = int(np.isnan(history).sum())
        context = {"method": "fill_missing", "n_filled": n_filled}
        return filled, context


class ClipOutliers(BaseSkill):
    def __init__(self):
        super().__init__()
        self.name = "clip_outliers"
        self.description = "基于IQR截断异常值"
        self.min_data_points = 10

    def execute(self, history: np.ndarray, horizon: int = 0, **kwargs) -> np.ndarray:
        q1 = np.percentile(history, 25)
        q3 = np.percentile(history, 75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        return np.clip(history, lower, upper)

    def execute_with_context(self, history: np.ndarray):
        q1 = np.percentile(history, 25)
        q3 = np.percentile(history, 75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        clipped = np.clip(history, lower, upper)
        context = {"method": "clip", "lower": lower, "upper": upper}
        return clipped, context


class IdentityPre(BaseSkill):
    def __init__(self):
        super().__init__()
        self.name = "identity_pre"
        self.description = "不做任何变换"
        self.min_data_points = 1

    def execute(self, history: np.ndarray, horizon: int = 0, **kwargs) -> np.ndarray:
        return history.copy()

    def execute_with_context(self, history: np.ndarray):
        return history.copy(), {"method": "identity"}


class ZScoreNormalize(BaseSkill):
    def __init__(self):
        super().__init__()
        self.name = "zscore_normalize"
        self.description = "减去均值，除以标准差"
        self.min_data_points = 3

    def execute(self, history: np.ndarray, horizon: int = 0, **kwargs) -> np.ndarray:
        mean = np.mean(history)
        std = np.std(history)
        if std < 1e-10:
            std = 1.0
        return (history - mean) / std

    def execute_with_context(self, history: np.ndarray):
        mean = np.mean(history)
        std = np.std(history)
        if std < 1e-10:
            std = 1.0
        transformed = (history - mean) / std
        context = {"method": "zscore", "mean": mean, "std": std}
        return transformed, context


class LinearDetrend(BaseSkill):
    def __init__(self):
        super().__init__()
        self.name = "linear_detrend"
        self.description = "线性去趋势"
        self.min_data_points = 5

    def execute(self, history: np.ndarray, horizon: int = 0, **kwargs) -> np.ndarray:
        x = np.arange(len(history))
        slope, intercept = np.polyfit(x, history, 1)
        trend = slope * x + intercept
        return history - trend

    def execute_with_context(self, history: np.ndarray):
        x = np.arange(len(history))
        slope, intercept = np.polyfit(x, history, 1)
        trend = slope * x + intercept
        transformed = history - trend
        context = {"method": "detrend", "slope": slope, "intercept": intercept, "orig_len": len(history)}
        return transformed, context


class BoxCoxTransform(BaseSkill):
    def __init__(self):
        super().__init__()
        self.name = "boxcox_transform"
        self.description = "Box-Cox幂变换"
        self.min_data_points = 5

    def execute(self, history: np.ndarray, horizon: int = 0, **kwargs) -> np.ndarray:
        if np.min(history) <= 0:
            shift = abs(np.min(history)) + 1e-6
            data = history + shift
        else:
            shift = 0.0
            data = history
        transformed, lam = stats.boxcox(data)
        return transformed

    def execute_with_context(self, history: np.ndarray):
        if np.min(history) <= 0:
            shift = abs(np.min(history)) + 1e-6
            data = history + shift
        else:
            shift = 0.0
            data = history
        transformed, lam = stats.boxcox(data)
        context = {"method": "boxcox", "lam": lam, "shift": shift}
        return transformed, context