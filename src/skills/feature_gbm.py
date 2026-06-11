import numpy as np
from .base import BaseSkill

class FeatureGBMSkill(BaseSkill):
    """使用滞后特征、滚动统计等训练 LightGBM 回归模型进行预测"""
    def __init__(self, n_lags=10, n_est=50):
        super().__init__()
        self.name = "feature_gbm"
        self.description = "LightGBM 构造滞后和滚动统计特征，适合复杂非线性模式"
        self.n_lags = n_lags
        self.n_est = n_est
        self.state_card = {
            "when_to_use": {
                "conditions": [
                    {"field": "data_length", "op": ">=", "value": 50}
                ],
                "logic": "AND"
            },
            "when_not_to_use": {
                "conditions": [
                    {"field": "data_length", "op": "<", "value": 30}
                ],
                "logic": "OR"
            },
            "visible_cues": ["序列非线性明显，传统模型误差较大"],
            "verification_cue": "交叉验证误差稳定",
            "failure_mode": "模型过拟合或数据量不足",
            "fallback_skill": "auto_arima"
        }

    def execute(self, history: np.ndarray, horizon: int, **kwargs) -> np.ndarray:
        try:
            import lightgbm as lgb
        except ImportError:
            return np.full(horizon, np.mean(history))
        # 构造特征
        n = len(history)
        if n < self.n_lags + 5:
            return np.full(horizon, np.mean(history))
        X, y = [], []
        for i in range(self.n_lags, n):
            feats = []
            # 滞后
            for lag in range(1, self.n_lags + 1):
                feats.append(history[i - lag])
            # 滚动统计
            feats.append(np.mean(history[i - self.n_lags : i]))
            feats.append(np.std(history[i - self.n_lags : i]))
            feats.append(history[i - 1] - history[i - 2])  # 差分
            X.append(feats)
            y.append(history[i])
        X = np.array(X)
        y = np.array(y)
        if len(X) < 10:
            return np.full(horizon, np.mean(history))
        # 训练 LightGBM
        model = lgb.LGBMRegressor(n_estimators=self.n_est, verbose=-1, random_state=42)
        model.fit(X, y)
        # 多步预测：迭代使用预测值
        last_feats = []
        for lag in range(1, self.n_lags + 1):
            last_feats.append(history[-lag])
        last_feats.append(np.mean(history[-self.n_lags:]))
        last_feats.append(np.std(history[-self.n_lags:]))
        last_feats.append(history[-1] - history[-2])
        current_feats = np.array(last_feats).reshape(1, -1)
        preds = []
        for _ in range(horizon):
            pred = model.predict(current_feats)[0]
            preds.append(pred)
            # 更新特征（简单递推，可能累积误差）
            new_feats = [pred] + last_feats[:self.n_lags - 1]
            rolling_mean = np.mean(new_feats)
            rolling_std = np.std(new_feats) if len(new_feats) > 1 else 0
            new_diff = pred - last_feats[0]
            current_feats = np.array(new_feats + [rolling_mean, rolling_std, new_diff]).reshape(1, -1)
            last_feats = new_feats
        return np.array(preds)