import numpy as np
from .base import BaseSkill

class IncrementalGBMSkill(BaseSkill):
    """增量/在线学习 LightGBM（需要 lightgbm 安装）"""
    def __init__(self, n_lags=10, n_est=50, update_freq=20):
        super().__init__()
        self.name = "incremental_gbm"
        self.description = "增量LightGBM：使用滞后特征和滚动统计，定期重训适应长序列"
        self.n_lags = n_lags
        self.n_est = n_est
        self.update_freq = update_freq
        self.model = None
        self._step = 0
        self.min_data_points = n_lags + 20
        self.requires_full_history = True
        self.strength_tags = ["ml", "incremental"]
        self.model_family = "ml"
        self.required_features = ["data_length"]
        self.decision_hint = "适合超长序列且模式复杂，需要一定计算资源。"
        self.state_card = {
            "when_to_use": {"conditions": [{"field": "data_length", "op": ">", "value": 500}], "logic": "AND"},
            "when_not_to_use": {"conditions": [], "logic": "OR"},
            "visible_cues": ["非线性模式，传统模型误差大"],
            "verification_cue": "交叉验证误差稳定",
            "fallback_skill": "naive_drift"
        }

    def _make_features(self, arr, start_idx):
        X, y = [], []
        for i in range(start_idx, len(arr)):
            if i < self.n_lags:
                continue
            feats = []
            for lag in range(1, self.n_lags+1):
                feats.append(arr[i - lag])
            feats.append(np.mean(arr[i-self.n_lags:i]))
            feats.append(np.std(arr[i-self.n_lags:i]))
            feats.append(arr[i-1] - arr[i-2] if i>=2 else 0)
            X.append(feats)
            y.append(arr[i])
        return np.array(X), np.array(y)

    def execute(self, history, horizon, **kwargs):
        try:
            import lightgbm as lgb
        except ImportError:
            return np.full(horizon, np.mean(history[-5:]))

        n = len(history)
        if n < self.min_data_points:
            return np.full(horizon, np.mean(history))

        # 定期重训
        self._step += 1
        if self.model is None or self._step % self.update_freq == 0:
            X, y = self._make_features(history, self.n_lags)
            if len(X) < 20:
                self.model = None
            else:
                self.model = lgb.LGBMRegressor(n_estimators=self.n_est, verbose=-1, random_state=42)
                self.model.fit(X, y)

        if self.model is None:
            return np.full(horizon, np.mean(history[-5:]))

        # 递归多步预测
        last_feats = []
        for lag in range(1, self.n_lags+1):
            last_feats.append(history[-lag])
        last_feats.append(np.mean(history[-self.n_lags:]))
        last_feats.append(np.std(history[-self.n_lags:]))
        last_feats.append(history[-1] - history[-2] if n>=2 else 0)

        preds = []
        current_feats = np.array(last_feats).reshape(1, -1)
        for _ in range(horizon):
            pred = self.model.predict(current_feats)[0]
            preds.append(pred)
            # 更新特征
            new_feats = [pred] + last_feats[:self.n_lags-1]
            rolling_mean = np.mean(new_feats)
            rolling_std = np.std(new_feats) if len(new_feats)>1 else 0
            new_diff = pred - last_feats[0]
            current_feats = np.array(new_feats + [rolling_mean, rolling_std, new_diff]).reshape(1, -1)
            last_feats = new_feats
        return np.array(preds)