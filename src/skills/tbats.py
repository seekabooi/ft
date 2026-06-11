import numpy as np
from .base import BaseSkill

class TBATSSkill(BaseSkill):
    def __init__(self):
        super().__init__()
        self.name = "tbats"
        self.description = "TBATS 模型，处理复杂季节性"
        self.min_data_points = 30
        self.requires_full_history = True
        self.strength_tags = ["season", "complex_season"]
        self.model_family = "stat_model"
        self.required_features = ["seasonal_strength", "data_length"]
        self.decision_hint = "处理复杂季节性（多周期、非整数周期）的强力模型，计算昂贵。仅当数据长度≥30且季节强度>0.3 时考虑，权重 0.3~0.6。"
        self.state_card = {
            "when_to_use": {
                "conditions": [{"field": "seasonal_strength", "op": ">", "value": 0.3},
                               {"field": "data_length", "op": ">=", "value": 30}],
                "logic": "AND"
            },
            "when_not_to_use": {
                "conditions": [{"field": "data_length", "op": "<", "value": 30}],
                "logic": "OR"
            },
            "visible_cues": ["复杂季节性模式"],
            "verification_cue": "残差白噪声",
            "fallback_skill": "seasonal_naive"
        }

    def execute(self, history, horizon, **kwargs):
        try:
            from tbats import TBATS
            estimator = TBATS(seasonal_periods=[12])
            model = estimator.fit(history)
            forecast = model.forecast(steps=horizon)
            return np.array(forecast)
        except:
            return np.full(horizon, np.mean(history[-5:]))