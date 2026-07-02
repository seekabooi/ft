import numpy as np
from scipy import stats
from src.skills.base import BaseSkill


# ==================== 第一类：强制逆变换（不由LLM选择，由系统绑定） ====================
class InvertIdentity(BaseSkill):
    def __init__(self):
        super().__init__()
        self.name = "invert_identity"
        self.required_signature = {"scale": "raw"}

    def execute(self, forecast: np.ndarray, context=None, **kwargs):
        return forecast


class InvertZScore(BaseSkill):
    def __init__(self):
        super().__init__()
        self.name = "invert_zscore"
        self.required_signature = {"scale": "zscore"}

    def execute(self, forecast: np.ndarray, context=None, **kwargs):
        if context is None:
            return forecast
        return forecast * context["std"] + context["mean"]


class InvertDetrend(BaseSkill):
    def __init__(self):
        super().__init__()
        self.name = "invert_detrend"
        self.required_signature = {"trend": "detrended"}

    def execute(self, forecast: np.ndarray, context=None, **kwargs):
        if context is None:
            return forecast
        # 外推趋势：预测步数 H 对应原序列之后的趋势
        # 这里需要知道预测的起始索引，从context中获取原始长度
        orig_len = context.get("orig_len", 0)
        slope = context["slope"]
        intercept = context["intercept"]
        # 趋势外推：x = orig_len, orig_len+1, ...
        x_forecast = np.arange(orig_len, orig_len + len(forecast))
        trend_future = slope * x_forecast + intercept
        return forecast + trend_future


class InvertBoxCox(BaseSkill):
    def __init__(self):
        super().__init__()
        self.name = "invert_boxcox"
        self.required_signature = {"scale": "boxcox"}

    def execute(self, forecast: np.ndarray, context=None, **kwargs):
        if context is None:
            return forecast
        lam = context["lam"]
        shift = context.get("shift", 0.0)
        # 逆Box-Cox
        if abs(lam) < 1e-8:
            inverted = np.exp(forecast)
        else:
            inverted = (forecast * lam + 1) ** (1 / lam)
        return inverted - shift


# ==================== 第二类：智能增强技能（由LLM选择） ====================
class IdentityEnhance(BaseSkill):
    def __init__(self):
        super().__init__()
        self.name = "enhance_identity"
        self.description = "不做任何增强"

    def execute(self, forecast: np.ndarray, **kwargs):
        return forecast


class ResidualARCorrection(BaseSkill):
    def __init__(self):
        super().__init__()
        self.name = "residual_ar"
        self.description = "AR(1)残差修正"
        self.min_data_points = 20

    def execute(self, forecast: np.ndarray, history=None, horizon=None, **kwargs):
        if history is None or len(history) < 15:
            return forecast
        # 1. 在训练集末尾做滚动预测，计算残差（这里假设基准预测是历史值本身，实际应用需调用核心模型）
        # 为了演示，我们只用 naive 基准计算残差，实际生产应使用 core_predictor
        # 注意：这里的残差是在原始尺度（已逆变换后）
        n = len(history)
        residuals = []
        for i in range(10, n - 1):
            pred = np.mean(history[i - 5:i])  # 简单基准，实际应替换为主干模型
            residuals.append(history[i] - pred)
        residuals = np.array(residuals)
        if len(residuals) < 5:
            return forecast

        # 2. 拟合 AR(1): e_t = phi * e_{t-1} + noise
        phi = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
        if np.isnan(phi) or abs(phi) < 0.05:
            return forecast  # 无自相关，不修正

        # 3. 外推残差：用最后一个残差递推
        e_last = residuals[-1]
        correction = []
        for k in range(1, len(forecast) + 1):
            e_pred = (phi ** k) * e_last
            correction.append(e_pred)
        correction = np.array(correction)

        # 4. 限制修正幅度（防止过度修正）
        max_correction = 0.3 * np.std(history)
        correction = np.clip(correction, -max_correction, max_correction)

        return forecast + correction


class QuantileCalibration(BaseSkill):
    def __init__(self):
        super().__init__()
        self.name = "quantile_calibration"
        self.description = "分位数校准（等温回归）"

    def execute(self, forecast: np.ndarray, history=None, **kwargs):
        # 简化版：基于历史误差的标准差进行缩放
        if history is None or len(history) < 10:
            return forecast
        # 假设存在系统偏差，用最近20点的中位数误差校准
        # 实际实现会更复杂，这里仅做示例
        return forecast  # 占位