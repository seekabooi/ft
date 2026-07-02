# experiments/autotune/state_encoder.py
"""
State Encoder - v4 全功能版（修复版）
★ 修复 np.polyfit 解包错误，改用 scipy.stats.linregress
★ 去硬阈值：全部连续值表达
★ 多窗口统计：短/中/长窗口的 mean/variance/delta
★ 随机投影：可学习的线性投影层
★ 统一 schema：numeric + embedding + regime_score
★ ★ 2026-06-26 增加 extract_regime() 和 get_regime_learning_rate() 方法（RL 支持）
"""

import numpy as np
from typing import Dict, Optional, List
import hashlib
from scipy import stats  # ★ 新增导入


class StateEncoder:
    """
    State Encoder - v4 全功能版（修复版）

    输出 schema:
        - numeric: 可解释特征（连续值，无硬分箱）
        - embedding: 不可解释特征（随机投影）
        - regime_score: 连续值（0~1），表示当前状态所处的 regime 位置
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._cache = {}

        # 从配置读取参数
        enc_cfg = self.config.get('encoder', {})
        self.windows = enc_cfg.get('windows', [7, 30, 90])
        self.statistics = enc_cfg.get('statistics', ['mean', 'variance', 'delta'])
        self.projection_dim = enc_cfg.get('projection_dim', 8)
        self.use_learned_projection = enc_cfg.get('use_learned_projection', True)
        self.regime_enabled = enc_cfg.get('regime_score_enabled', True)
        self.regime_temp = enc_cfg.get('regime_temperature', 1.0)

        # Regime 阈值（用于 extract_regime 返回标签）
        self.regime_thresholds = {
            'volatility': 1.2,      # volatility_ratio > 1.2 → 高波动
            'seasonality': 0.3,     # seasonal_strength > 0.3 → 强季节
            'trend': 0.4,           # trend_strength > 0.4 → 强趋势
            'stationary': 0.05      # adf_pvalue < 0.05 → 平稳
        }

        # 随机投影矩阵（使用固定种子初始化，后续可通过 EMA 更新）
        self._projection = None
        self._projection_init_seed = 42

        # 用于 regime_score 的参考 embedding（在第一次编码时初始化）
        self._regime_centers = None  # 存储见过的 regime 中心

    def encode(self, series: np.ndarray) -> Dict:
        """编码状态"""
        cache_key = self._compute_cache_key(series)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # 1. 提取可解释特征（全部连续值，无硬分箱）
        numeric = self._extract_continuous_features(series)

        # 2. 构建结构化特征向量（用于投影）
        feature_vector = self._build_feature_vector(series, numeric)

        # 3. 随机投影 → embedding
        embedding = self._apply_projection(feature_vector)

        # 4. 计算 regime_score（连续值）
        if self.regime_enabled:
            regime_score = self._compute_regime_score(embedding)
        else:
            regime_score = 0.5

        # 5. 组装输出
        state = {
            'numeric': numeric,
            'embedding': embedding.tolist() if embedding is not None else [],
            'regime_score': float(regime_score),
            'recommended_temperature': self._compute_temperature(series, numeric)
        }

        self._cache[cache_key] = state
        return state

    # ★★★ 新增：提取 Regime 标签 ★★★
    def extract_regime(self, numeric: Dict[str, float]) -> Dict[str, int]:
        """
        从 numeric 特征中提取 regime 标签

        Returns:
            {
                'volatility': 0/1,      # 高波动
                'seasonality': 0/1,     # 强季节
                'trend': 0/1,           # 强趋势
                'stationary': 0/1,      # 平稳
                'long_sequence': 0/1    # 长序列 (>400)
            }
        """
        regime = {}

        # 波动性
        vol_ratio = numeric.get('volatility_ratio', 1.0)
        regime['volatility'] = 1 if vol_ratio > self.regime_thresholds['volatility'] else 0

        # 季节性（从特征中获取，如果没有则用 0）
        seasonal = numeric.get('seasonal_strength', 0.0)
        regime['seasonality'] = 1 if seasonal > self.regime_thresholds['seasonality'] else 0

        # 趋势
        trend = numeric.get('trend_strength', 0.0)
        # 如果没有 trend_strength，尝试从 trend_slope 推断
        if trend == 0.0 and 'trend_slope' in numeric:
            trend = min(1.0, abs(numeric.get('trend_slope', 0.0)) * 5)
        regime['trend'] = 1 if trend > self.regime_thresholds['trend'] else 0

        # 平稳性
        adf_pvalue = numeric.get('adf_pvalue', 0.5)
        regime['stationary'] = 1 if adf_pvalue < self.regime_thresholds['stationary'] else 0

        # 长序列
        data_length = numeric.get('data_length', 0)
        regime['long_sequence'] = 1 if data_length > 400 else 0

        return regime

    # ★★★ 新增：根据 Regime 获取学习率调整系数 ★★★
    def get_regime_learning_rate(self, numeric: Dict[str, float], base_lr: float) -> float:
        """
        根据 regime 调整学习率

        Returns:
            调整后的学习率
        """
        regime = self.extract_regime(numeric)

        # 默认系数
        multiplier = 1.0

        # 高波动 → 增大学习率（更快适应变化）
        if regime.get('volatility', 0) == 1:
            multiplier *= 1.5

        # 强季节 → 适度增大
        if regime.get('seasonality', 0) == 1:
            multiplier *= 1.2

        # 平稳 → 减小学习率（更稳定）
        if regime.get('stationary', 0) == 1:
            multiplier *= 0.7

        # 长序列 → 适度减小（更稳定）
        if regime.get('long_sequence', 0) == 1:
            multiplier *= 0.9

        # 限制范围
        multiplier = max(0.3, min(2.0, multiplier))

        return base_lr * multiplier

    def _extract_continuous_features(self, series: np.ndarray) -> Dict[str, float]:
        """
        提取可解释特征，全部连续值，无硬分箱
        包含：全局统计 + 局部动态 + 多窗口统计
        """
        n = len(series)
        features = {}

        # ----- 全局统计（连续值） -----
        features['data_length'] = float(n)
        if n > 0:
            features['mean'] = float(np.mean(series))
            features['std'] = float(np.std(series))
            features['cv'] = features['std'] / (abs(features['mean']) + 1e-8)
            features['min'] = float(np.min(series))
            features['max'] = float(np.max(series))
            features['range'] = features['max'] - features['min']
            features['skewness'] = float(self._skewness(series))
            features['kurtosis'] = float(self._kurtosis(series))

            # 趋势强度（连续值，使用 scipy.stats.linregress）
            if n >= 5:
                x = np.arange(n)
                try:
                    # ★★★ 修复：使用 scipy.stats.linregress，直接返回 5 个值 ★★★
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, series)
                    features['trend_slope'] = float(slope)
                    features['trend_r2'] = float(r_value ** 2)
                    features['trend_pvalue'] = float(p_value)
                    # ★★★ 新增：趋势强度（基于 R² 归一化） ★★★
                    features['trend_strength'] = min(1.0, float(abs(r_value)))
                except Exception as e:
                    # 回退方案：使用 np.polyfit 获取斜率
                    coeffs = np.polyfit(x, series, 1)
                    features['trend_slope'] = float(coeffs[0])
                    features['trend_r2'] = 0.0
                    features['trend_pvalue'] = 0.5
                    features['trend_strength'] = 0.0

            # 平稳性（ADF p-value，连续值）
            try:
                from statsmodels.tsa.stattools import adfuller
                adf_result = adfuller(series, maxlag=min(12, n // 2))
                features['adf_pvalue'] = float(adf_result[1])
            except:
                features['adf_pvalue'] = 0.5

        # ----- 局部动态（连续值，最近30点） -----
        local_n = min(30, n)
        if local_n >= 3:
            local_series = series[-local_n:]
            features['local_mean'] = float(np.mean(local_series))
            features['local_std'] = float(np.std(local_series))
            features['local_cv'] = features['local_std'] / (abs(features['local_mean']) + 1e-8)
            # 局部斜率
            x = np.arange(len(local_series))
            try:
                slope, _, _, _, _ = stats.linregress(x, local_series)
                features['local_slope'] = float(slope)
            except:
                features['local_slope'] = 0.0
            # 局部变化率
            if local_series[0] != 0:
                features['local_change_rate'] = float((local_series[-1] - local_series[0]) / abs(local_series[0] + 1e-8))
            else:
                features['local_change_rate'] = 0.0

            # 局部波动率（近期 vs 全局）
            if features.get('std', 0) > 0:
                features['volatility_ratio'] = features['local_std'] / features['std']
            else:
                features['volatility_ratio'] = 1.0

        # ----- 多窗口统计（短/中/长） -----
        for window in self.windows:
            if n >= window:
                win_series = series[-window:]
                features[f'win_{window}_mean'] = float(np.mean(win_series))
                features[f'win_{window}_std'] = float(np.std(win_series))
                # Delta change: 窗口首尾差
                features[f'win_{window}_delta'] = float(win_series[-1] - win_series[0])
                # 窗口内变化率（相对于窗口均值）
                if features[f'win_{window}_mean'] != 0:
                    features[f'win_{window}_change_rate'] = (
                        features[f'win_{window}_delta'] / abs(features[f'win_{window}_mean'])
                    )
                else:
                    features[f'win_{window}_change_rate'] = 0.0
                # 窗口内方差
                features[f'win_{window}_variance'] = float(np.var(win_series))

        # ----- Regime 信息（连续值，不输出标签） -----
        if 'volatility_ratio' in features:
            features['regime_indicator'] = features['volatility_ratio']
        else:
            features['regime_indicator'] = 0.5

        return features

    def _build_feature_vector(self, series: np.ndarray, numeric: Dict) -> np.ndarray:
        """构建用于投影的结构化特征向量"""
        # 从 numeric 中提取关键特征
        keys = [
            'mean', 'std', 'cv', 'trend_slope', 'trend_r2',
            'adf_pvalue', 'local_mean', 'local_std', 'local_slope',
            'volatility_ratio', 'regime_indicator'
        ]
        # 加上多窗口统计
        for window in self.windows:
            keys.append(f'win_{window}_mean')
            keys.append(f'win_{window}_std')
            keys.append(f'win_{window}_delta')

        vector = []
        for key in keys:
            vector.append(numeric.get(key, 0.0))

        return np.array(vector, dtype=np.float32)

    def _apply_projection(self, feature_vector: np.ndarray) -> np.ndarray:
        """应用随机投影（可学习版）"""
        if self._projection is None:
            # 初始化投影矩阵
            rng = np.random.RandomState(self._projection_init_seed)
            self._projection = rng.randn(len(feature_vector), self.projection_dim) / np.sqrt(len(feature_vector))

        # 投影
        embedding = feature_vector @ self._projection

        # L2 归一化
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def update_projection(self, gradient: np.ndarray, lr: float = 0.01):
        """更新投影矩阵（可学习）"""
        if self._projection is not None and self.use_learned_projection:
            self._projection += lr * gradient
            # 保持列归一化
            for i in range(self._projection.shape[1]):
                col_norm = np.linalg.norm(self._projection[:, i])
                if col_norm > 0:
                    self._projection[:, i] /= col_norm

    def _compute_regime_score(self, embedding: np.ndarray) -> float:
        """计算 regime_score（连续值 0~1）"""
        if self._regime_centers is None:
            self._regime_centers = [embedding.copy()]
            return 0.5

        # 计算与最近 regime 中心的相似度（使用余弦相似度）
        similarities = []
        for center in self._regime_centers:
            cos_sim = np.dot(embedding, center) / (
                np.linalg.norm(embedding) * np.linalg.norm(center) + 1e-8
            )
            similarities.append(cos_sim)

        max_sim = max(similarities) if similarities else 0.0
        score = 1.0 / (1.0 + np.exp(-(max_sim - 0.5) / self.regime_temp))

        # 更新 regime 中心（EMA）
        if len(self._regime_centers) < 10:
            self._regime_centers.append(embedding.copy())
        else:
            distances = [np.linalg.norm(embedding - c) for c in self._regime_centers]
            farthest_idx = np.argmax(distances)
            if distances[farthest_idx] > 0.5:
                self._regime_centers[farthest_idx] = embedding.copy()

        return float(score)

    def _compute_temperature(self, series: np.ndarray, numeric: Dict) -> float:
        """根据状态推荐 temperature"""
        base_temp = 1.0
        cv = numeric.get('cv', 0.5)
        volatility_ratio = numeric.get('volatility_ratio', 1.0)
        temp = base_temp * (1.0 + 0.5 * cv * volatility_ratio)
        return min(1.5, max(0.5, temp))

    def _skewness(self, x: np.ndarray) -> float:
        try:
            from scipy import stats
            return float(stats.skew(x))
        except:
            return 0.0

    def _kurtosis(self, x: np.ndarray) -> float:
        try:
            from scipy import stats
            return float(stats.kurtosis(x))
        except:
            return 0.0

    def _compute_cache_key(self, series: np.ndarray) -> str:
        data = series[-200:] if len(series) >= 200 else series
        return hashlib.md5(data.tobytes()).hexdigest()

    def clear_cache(self):
        self._cache.clear()