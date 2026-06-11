import numpy as np
from .base import BaseSkill

class CrostonSkill(BaseSkill):
    def __init__(self):
        super().__init__()
        self.name = "croston"
        self.description = "Croston 方法，适合间歇性需求预测"
        self.min_data_points = 5
        self.requires_full_history = False
        self.strength_tags = ["intermittent"]
        self.model_family = "lightweight"
        self.required_features = ["missing_rate", "data_length"]
        self.decision_hint = "专为间歇性需求（较多零值或缺失）设计。当缺失率>0.2 时考虑，权重可 0.3~0.6。"
        # 加强条件：要求缺失率 > 0.2，避免在非间歇序列上误用
        self.state_card = {
            "when_to_use": {
                "conditions": [
                    {"field": "missing_rate", "op": ">", "value": 0.2},
                    {"field": "data_length", "op": ">=", "value": 20}
                ],
                "logic": "AND"
            },
            "when_not_to_use": {
                "conditions": [
                    {"field": "seasonal_strength", "op": ">", "value": 0.3},
                    {"field": "missing_rate", "op": "<", "value": 0.05}
                ],
                "logic": "OR"
            },
            "visible_cues": ["序列含有较多零值或缺失"],
            "verification_cue": "预测值非负",
            "fallback_skill": "naive"
        }

    def execute(self, history, horizon, **kwargs):
        nonzero = history[history > 1e-8]
        if len(nonzero) == 0:
            return np.zeros(horizon)
        intervals = np.diff(np.where(history > 1e-8)[0])
        mean_interval = np.mean(intervals) if len(intervals) > 0 else 1.0
        mean_nonzero = np.mean(nonzero)
        preds = []
        for i in range(horizon):
            if i % max(1, int(mean_interval)) == 0:
                preds.append(mean_nonzero)
            else:
                preds.append(0.0)
        return np.array(preds[:horizon])