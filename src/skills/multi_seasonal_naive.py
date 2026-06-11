import numpy as np
from .base import BaseSkill

class MultiSeasonalNaiveSkill(BaseSkill):
    def __init__(self, period=12):
        super().__init__()
        self.name = "multi_seasonal_naive"
        self.description = "乘法季节性朴素预测（对数变换）"
        self.period = period
        self.min_data_points = 2 * period
        self.requires_full_history = False
        self.strength_tags = ["season", "multiplicative"]
        self.model_family = "lightweight"
        self.required_features = ["seasonal_strength", "data_length", "period"]
        self.decision_hint = (
            "适合季节性幅度随趋势增长的数据，计算极快。"
            "当季节性强度>0.5且趋势明显上升时优先考虑，权重可占 0.3~0.6。"
        )
        self.state_card = {
            "when_to_use": {
                "conditions": [
                    {"field": "seasonal_strength", "op": ">", "value": 0.3},
                    {"field": "data_length", "op": ">=", "value": 2 * self.period}
                ],
                "logic": "AND"
            },
            "when_not_to_use": {
                "conditions": [
                    {"field": "seasonal_strength", "op": "<", "value": 0.2}
                ],
                "logic": "OR"
            },
            "visible_cues": ["季节性幅度随趋势增大"],
            "verification_cue": "对数还原后预测值在历史极值范围内",
            "fallback_skill": "seasonal_naive"
        }

    def execute(self, history, horizon, **kwargs):
        p = kwargs.get("period", self.period)
        if len(history) < p:
            return np.full(horizon, np.mean(history[-min(5, len(history)):]))
        # 对数变换（避免零值）
        log_history = np.log(np.maximum(history, 1e-8))
        preds = []
        for i in range(horizon):
            idx = -(p - i % p)
            preds.append(log_history[idx])
        return np.exp(np.array(preds))