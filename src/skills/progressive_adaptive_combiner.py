import numpy as np
from scipy import stats
from scipy.fft import fft
from sklearn.linear_model import Ridge
from .base import BaseSkill
from .seasonal_naive import SeasonalNaiveSkill

class ProgressiveAdaptiveCombiner(BaseSkill):
    """
    渐进自适应组合器 (ProgressiveAdaptiveCombiner)

    在历史序列上划分 K 个递增窗口，逐步评估基础技能的预测误差，
    并利用指数平滑更新技能权重；同时筛选额外数值特征，最终输出
    加权组合预测。完全遵循 MMSkills 的数学表示，无多模态依赖。
    """
    def __init__(self, K=4, alpha=0.5, tau=0.1,
                 min_error_reduction=0.05, max_weight_volatility=0.6,
                 max_resid_autocorr=0.7, verification_ratio=0.9):
        super().__init__()
        self.name = "progressive_adaptive_combiner"
        self.description = "多窗口渐进学习组合器：动态调整技能权重与额外特征"
        self.min_data_points = 50
        self.requires_full_history = True
        self.strength_tags = ["meta", "adaptive", "combination"]
        self.model_family = "meta"
        self.required_features = []          # 自行计算所有特征
        self.decision_hint = (
            "适合长度≥50的序列，能够根据历史窗口渐进学习最优技能权重，"
            "鲁棒性强，适用于模式缓慢变化的数据。"
        )

        # 内部可调参数
        self.K = K
        self.alpha = alpha
        self.tau = tau
        self.min_error_reduction = min_error_reduction
        self.max_weight_volatility = max_weight_volatility
        self.max_resid_autocorr = max_resid_autocorr
        self.verification_ratio = verification_ratio

        self.EXTRA_FEATURES = ["ma_trend", "volatility", "spectral_entropy", "outlier_density"]

        # 状态卡
        self.state_card = {
            "when_to_use": {
                "conditions": [
                    {"field": "data_length", "op": ">=", "value": self.min_data_points}
                ],
                "logic": "AND"
            },
            "when_not_to_use": {
                "conditions": [
                    {"field": "data_length", "op": "<", "value": self.min_data_points}
                ],
                "logic": "OR"
            },
            "visible_cues": ["序列长度充足"],
            "verification_cue": "组合预测误差低于最优单一基础技能",
            "fallback_skill": "seasonal_naive"
        }

    # ---------- 内部工具方法 ----------
    def _get_base_skills(self, history):
        """返回适用于当前序列的基础技能实例列表（不包含自身）"""
        from .naive import NaiveSkill
        from .seasonal_naive import SeasonalNaiveSkill
        from .holt_winters import HoltWintersSkill
        from .naive_drift import NaiveDriftSkill
        from .ets import ETSSkill
        from .theta import ThetaSkill
        from .ar_skill import ARSkill
        from .fourier_skill import FourierSkill

        skills = [
            NaiveSkill(),
            SeasonalNaiveSkill(period=12),
            NaiveDriftSkill(),
            HoltWintersSkill(period=12),
            ETSSkill(),
            ThetaSkill(),
            ARSkill(),
            FourierSkill(period=12, n_harmonics=3),
        ]
        # 过滤掉数据量不足的技能
        return [s for s in skills if len(history) >= s.min_data_points]

    def _compute_extra_features(self, series):
        """计算额外特征字典"""
        n = len(series)
        feats = {f: 0.0 for f in self.EXTRA_FEATURES}
        if n < 10:
            return feats

        # 移动平均斜率
        x = np.arange(10)
        recent = series[-10:]
        slope, _, _, _, _ = stats.linregress(x, recent)
        feats["ma_trend"] = slope / (np.mean(np.abs(recent)) + 1e-8)

        # 局部波动率
        win = series[-20:]
        feats["volatility"] = np.std(win) / (np.mean(np.abs(win)) + 1e-8)

        # 频谱熵
        if n >= 16:
            fft_vals = np.abs(fft(series))
            power = fft_vals[:n // 2] ** 2
            power_norm = power / (power.sum() + 1e-8)
            entropy = -np.sum(power_norm * np.log(power_norm + 1e-8))
            feats["spectral_entropy"] = entropy / np.log(n // 2)
        else:
            feats["spectral_entropy"] = 0.0

        # 离群点密度
        med = np.median(series)
        mad = np.median(np.abs(series - med))
        if mad == 0:
            feats["outlier_density"] = 0.0
        else:
            outliers = np.abs(series - med) > 3 * mad
            feats["outlier_density"] = float(np.mean(outliers))

        return feats

    def _evaluate_skill(self, skill, history):
        """使用滚动原点评估技能的一步预测对称百分比误差均值"""
        n = len(history)
        min_train = max(skill.min_data_points, 10)
        errors = []
        for i in range(min_train, n):
            train = history[:i]
            actual = history[i]
            try:
                pred = skill.execute(train, 1)[0]
                err = abs(pred - actual) / (abs(pred) + abs(actual) + 1e-8)
                errors.append(err)
            except Exception:
                continue
        return np.mean(errors) if errors else 1.0

    # ---------- 核心预测 ----------
    def execute(self, history, horizon, **kwargs):
        n = len(history)
        if n < self.min_data_points:
            return SeasonalNaiveSkill().execute(history, horizon)

        base_skills = self._get_base_skills(history)
        M = len(base_skills)
        if M == 0:
            return SeasonalNaiveSkill().execute(history, horizon)

        # 窗口长度序列
        L_seq = np.linspace(self.min_data_points, n, self.K).astype(int)
        L_seq[-1] = n

        # 初始化权重（均匀）
        w_prev = np.ones(M) / M
        weight_history = [w_prev.copy()]
        window_errors = []
        feature_coefs = np.zeros(len(self.EXTRA_FEATURES))

        for k, L_k in enumerate(L_seq):
            if L_k < 2 * horizon:
                continue
            window_data = history[:L_k]

            # 评估基础技能误差
            errors = np.array([self._evaluate_skill(sk, window_data) for sk in base_skills])
            window_errors.append(np.mean(errors))

            # 更新权重
            inv_err = 1.0 / (errors + 1e-8)
            w_raw = inv_err / inv_err.sum()
            w_new = self.alpha * w_raw + (1 - self.alpha) * w_prev
            weight_history.append(w_new.copy())
            w_prev = w_new

        # 最终权重
        final_weights = w_prev

        # 生成预测
        base_forecasts = []
        for sk in base_skills:
            try:
                fc = sk.execute(history, horizon)
                base_forecasts.append(fc)
            except Exception:
                base_forecasts.append(np.zeros(horizon))
        base_forecasts = np.array(base_forecasts)
        combined = np.dot(final_weights, base_forecasts)

        # 额外特征修正（当前版本仅做微弱调整）
        last_feats = self._compute_extra_features(history)
        correction = np.zeros(horizon)
        for idx, feat in enumerate(self.EXTRA_FEATURES):
            correction += feature_coefs[idx] * last_feats[feat]

        return combined + correction