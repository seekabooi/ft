import numpy as np
from .base import BaseSkill

class AdaptiveWeightedEnsemble(BaseSkill):
    """滑动窗口自适应权重：动态评估多个技能的最近误差，调整组合权重"""
    def __init__(self, skills=None, window=100, update_freq=20):
        super().__init__()
        self.name = "adaptive_weighted_ensemble"
        self.description = "自适应加权组合：基于滚动窗口的近期误差动态更新技能权重"
        self.skills = skills or []
        self.window = window
        self.update_freq = update_freq
        self.weights = None
        self._step = 0
        self.min_data_points = max(window, 50)
        self.requires_full_history = True
        self.strength_tags = ["adaptive", "ensemble"]
        self.model_family = "meta"
        self.required_features = ["data_length"]
        self.decision_hint = "适合长序列且模式可能缓慢变化的场景。"
        self.state_card = {
            "when_to_use": {"conditions": [{"field": "data_length", "op": ">", "value": 200}], "logic": "AND"},
            "when_not_to_use": {"conditions": [], "logic": "OR"},
            "visible_cues": ["序列非平稳，模式缓慢变化"],
            "verification_cue": "组合权重随时间平滑变化",
            "fallback_skill": "naive"
        }

    def execute(self, history, horizon, **kwargs):
        self._step += 1
        n = len(history)
        if not self.skills or n < self.min_data_points:
            return np.full(horizon, np.mean(history))
        # 定期更新权重
        if self.weights is None or self._step % self.update_freq == 0:
            recent = history[-self.window:]
            errors = []
            for sk in self.skills:
                try:
                    # 在 recent 上做一步滚动预测评估
                    errs = []
                    for i in range(max(10, len(recent)//2), len(recent)):
                        train = recent[:i]
                        pred = sk.execute(train, 1)[0]
                        errs.append(abs(pred - recent[i]))
                    mae = np.mean(errs)
                    errors.append(1.0 / (mae + 1e-8))
                except:
                    errors.append(0.0)
            total = sum(errors)
            if total > 0:
                self.weights = [e / total for e in errors]
            else:
                self.weights = [1.0 / len(self.skills)] * len(self.skills)
        # 加权预测
        preds = []
        for sk in self.skills:
            try:
                preds.append(sk.execute(history, horizon, **kwargs))
            except:
                preds.append(np.full(horizon, np.mean(history)))
        preds = np.array(preds)
        weighted = np.average(preds, axis=0, weights=self.weights)
        return weighted