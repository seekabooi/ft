# experiments/autotune/meta_skills.py
"""
System Robustness Modules

★ 性能优化版本：
- 气温数据专项（seasonality-aware skill）
- 金价数据专项（regime switching skill）
- 安全网、专家投票、概念漂移
"""

import numpy as np
from typing import Dict, List, Optional, Any


class SafetyNet:
    """安全网：防崩溃"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.iqr_multiplier = config.get('iqr_multiplier', 3.0)

    def detect(self, prediction: np.ndarray, history: np.ndarray) -> bool:
        q1 = np.percentile(history, 25)
        q3 = np.percentile(history, 75)
        iqr = q3 - q1
        lower_bound = q1 - self.iqr_multiplier * iqr
        upper_bound = q3 + self.iqr_multiplier * iqr
        return np.any((prediction < lower_bound) | (prediction > upper_bound))

    def execute(self, prediction: np.ndarray, history: np.ndarray) -> np.ndarray:
        if not self.detect(prediction, history):
            return prediction
        fallback = np.full_like(prediction, history[-1] if len(history) > 0 else 0)
        return fallback


class ExpertVoting:
    """专家投票：高波动场景"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.std_threshold = config.get('std_threshold', 0.2)

    def detect(self, predictions: List[np.ndarray], history: np.ndarray) -> bool:
        if len(predictions) < 2:
            return False
        stacked = np.stack(predictions)
        mean_std = np.mean(np.std(stacked, axis=0))
        hist_std = np.std(history)
        return hist_std > 0 and mean_std > hist_std * self.std_threshold

    def execute(self, predictions: List[np.ndarray], history: np.ndarray) -> np.ndarray:
        if not self.detect(predictions, history):
            return np.mean(np.stack(predictions), axis=0)
        return np.median(np.stack(predictions), axis=0)


class ConceptDriftDetector:
    """概念漂移检测"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.window_size = config.get('drift_window_size', 10)
        self.threshold = config.get('drift_threshold', 1.5)
        self.consecutive_steps = config.get('drift_consecutive', 3)
        self._error_history = []
        self._drift_count = 0
        self._drift_detected = False

    def detect(self, prediction: np.ndarray, actual: np.ndarray) -> bool:
        if len(prediction) != len(actual):
            return False
        current_error = np.mean(np.abs(prediction - actual))
        self._error_history.append(current_error)
        if len(self._error_history) > 100:
            self._error_history.pop(0)
        if len(self._error_history) < self.window_size + 1:
            return False
        baseline = np.mean(self._error_history[:-self.window_size])
        if baseline == 0:
            return False
        recent = np.mean(self._error_history[-self.window_size:])
        if recent / baseline > self.threshold:
            self._drift_count += 1
        else:
            self._drift_count = 0
        if self._drift_count >= self.consecutive_steps:
            self._drift_detected = True
            return True
        return False

    def get_adapted_history(self, history: np.ndarray) -> np.ndarray:
        if not self._drift_detected:
            return history
        return history[-50:] if len(history) > 50 else history


# ==================== ★ 气温专项：Seasonality-Aware Skill ====================

class SeasonalityAwareSkill:
    """
    气温数据专项技能

    根据季节强度自动选择不同的预测策略
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.seasonal_thresholds = {
            'daily': 7,
            'weekly': 30,
            'seasonal': 90
        }

    def execute(self, history: np.ndarray, horizon: int, period: int) -> np.ndarray:
        """执行季节感知预测"""
        n = len(history)
        if n < period:
            return np.array([np.mean(history)] * horizon)

        # 检测当前季节模式
        seasonality = self._detect_seasonality(history, period)

        if seasonality == 'strong':
            # 使用季节性模型
            return self._seasonal_forecast(history, horizon, period)
        elif seasonality == 'medium':
            # 混合模型
            return self._mixed_forecast(history, horizon, period)
        else:
            # 简单模型
            return np.array([np.mean(history[-30:])] * horizon)

    def _detect_seasonality(self, history: np.ndarray, period: int) -> str:
        """检测季节强度"""
        if len(history) < 2 * period:
            return 'weak'

        try:
            from statsmodels.tsa.stattools import acf
            acf_vals = acf(history, nlags=period, fft=True)
            if len(acf_vals) > period:
                seasonal_acf = acf_vals[period]
                if abs(seasonal_acf) > 0.3:
                    return 'strong'
                elif abs(seasonal_acf) > 0.15:
                    return 'medium'
            return 'weak'
        except:
            return 'weak'

    def _seasonal_forecast(self, history: np.ndarray, horizon: int, period: int) -> np.ndarray:
        """季节性预测"""
        if len(history) < period:
            return np.array([np.mean(history)] * horizon)

        # 使用最近的周期模式
        recent_cycle = history[-period:]
        predictions = []
        for i in range(horizon):
            idx = i % period
            predictions.append(recent_cycle[idx])

        return np.array(predictions)

    def _mixed_forecast(self, history: np.ndarray, horizon: int, period: int) -> np.ndarray:
        """混合预测"""
        seasonal_pred = self._seasonal_forecast(history, horizon, period)
        mean_pred = np.array([np.mean(history[-30:])] * horizon)
        return 0.6 * seasonal_pred + 0.4 * mean_pred


# ==================== ★ 金价专项：Regime Switching Skill ====================

class RegimeSwitchingSkill:
    """
    金价数据专项技能

    根据区制自动切换预测策略
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.regime_threshold = config.get('regime_threshold', 0.15)
        self._current_regime = 'stable'
        self._regime_history = []

    def execute(self, history: np.ndarray, horizon: int, period: int) -> np.ndarray:
        """执行区制切换预测"""
        n = len(history)
        if n < 30:
            return np.array([np.mean(history)] * horizon)

        # 检测当前区制
        regime = self._detect_regime(history)
        self._current_regime = regime
        self._regime_history.append(regime)
        if len(self._regime_history) > 20:
            self._regime_history.pop(0)

        if regime == 'bullish':
            return self._bullish_forecast(history, horizon)
        elif regime == 'bearish':
            return self._bearish_forecast(history, horizon)
        elif regime == 'shock':
            return self._shock_forecast(history, horizon)
        else:
            return np.array([np.mean(history[-20:])] * horizon)

    def _detect_regime(self, history: np.ndarray) -> str:
        """检测当前区制"""
        n = len(history)
        if n < 30:
            return 'stable'

        # 趋势
        from scipy import stats
        x = np.arange(n)
        slope, _, r_value, _, _ = stats.linregress(x, history)

        # 波动性
        volatility = np.std(history[-20:]) / (np.mean(history[-20:]) + 0.001)

        if abs(slope) > 0.03 and abs(r_value) > 0.3:
            if slope > 0:
                return 'bullish'
            else:
                return 'bearish'
        elif volatility > 0.4:
            return 'shock'
        else:
            return 'stable'

    def _bullish_forecast(self, history: np.ndarray, horizon: int) -> np.ndarray:
        """上涨区制预测"""
        # 使用趋势外推 + 少量均值回归
        n = len(history)
        if n < 10:
            return np.array([np.mean(history)] * horizon)

        from scipy import stats
        x = np.arange(n)
        slope, intercept, _, _, _ = stats.linregress(x, history)

        predictions = []
        for i in range(horizon):
            pred = slope * (n + i) + intercept
            # 少量均值回归约束
            mean_regression = 0.1 * (np.mean(history) - pred)
            predictions.append(pred + mean_regression)

        return np.array(predictions)

    def _bearish_forecast(self, history: np.ndarray, horizon: int) -> np.ndarray:
        """下跌区制预测"""
        n = len(history)
        if n < 10:
            return np.array([np.mean(history)] * horizon)

        from scipy import stats
        x = np.arange(n)
        slope, intercept, _, _, _ = stats.linregress(x, history)

        predictions = []
        for i in range(horizon):
            pred = slope * (n + i) + intercept
            # 防止过度下跌
            mean_regression = 0.15 * (np.mean(history) - pred)
            predictions.append(pred + mean_regression)

        return np.array(predictions)

    def _shock_forecast(self, history: np.ndarray, horizon: int) -> np.ndarray:
        """冲击区制预测"""
        # 回退到稳健均值
        return np.array([np.median(history[-10:])] * horizon)