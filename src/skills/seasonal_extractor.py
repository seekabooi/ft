import numpy as np
from .base import BaseSkill

class SeasonalExtractorSkill(BaseSkill):
    def __init__(self, period=12):
        super().__init__()
        self.name = "seasonal_extractor"
        self.description = "季节成分提取器，用季节朴素外推预测"
        self.period = period
        self.min_data_points = 2 * period
        self.requires_full_history = True
        self.strength_tags = ["season", "decomposition"]
        self.model_family = "lightweight"
        self.required_features = ["seasonal_strength", "data_length", "period"]
        self.decision_hint = (
            "提取固定周期季节成分并外推。适合季节明显的序列，可作为 DAG 节点。"
            "在加权求和模式下输出季节朴素预测。"
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
                "conditions": [{"field": "seasonal_strength", "op": "<", "value": 0.2}],
                "logic": "OR"
            },
            "visible_cues": ["存在稳定周期波动"],
            "verification_cue": "季节成分在周期内稳定",
            "fallback_skill": "seasonal_naive"
        }

    def execute(self, history, horizon, **kwargs):
        p = kwargs.get("period", self.period)
        if len(history) < p:
            return np.full(horizon, np.mean(history[-5:]))
        # 朴素季节外推作为预测
        preds = []
        for i in range(horizon):
            idx = -(p - i % p)
            preds.append(history[idx])
        return np.array(preds)

    def decompose(self, history, period=None):
        """返回 (seasonal, deseasonalized)"""
        p = period if period is not None else self.period
        n = len(history)
        if n < 2 * p:
            return np.zeros_like(history), history
        # 使用经典移动平均分解
        seasonal = np.zeros(n)
        for i in range(p, n - p):
            seasonal[i] = np.mean(history[i - p:i + p + 1])
        # 简单季节成分
        seasonal_avg = np.array([np.mean(history[j::p]) for j in range(p)])
        seasonal = np.tile(seasonal_avg, n // p + 1)[:n]
        deseasonalized = history - seasonal
        return seasonal, deseasonalized