import numpy as np
from scipy import stats
from .base import BaseSkill

class DetrenderSkill(BaseSkill):
    def __init__(self):
        super().__init__()
        self.name = "detrender"
        self.description = "线性趋势分离器，同时外推趋势作为预测"
        self.min_data_points = 10
        self.requires_full_history = True
        self.strength_tags = ["trend", "decomposition"]
        self.model_family = "lightweight"
        self.required_features = ["trend_strength", "data_length"]
        self.decision_hint = (
            "提取线性趋势并外推。适合趋势明显的序列，可作为 DAG 的起始节点。"
            "在加权求和模式下，它直接输出趋势预测，可与其他技能组合。"
        )
        self.state_card = {
            "when_to_use": {"conditions": [], "logic": "AND"},
            "when_not_to_use": {
                "conditions": [{"field": "trend_strength", "op": "<", "value": 0.1}],
                "logic": "OR"
            },
            "visible_cues": ["序列存在明显上升或下降趋势"],
            "verification_cue": "趋势线能较好拟合历史数据",
            "fallback_skill": "naive"
        }

    def execute(self, history, horizon, **kwargs):
        n = len(history)
        if n < self.min_data_points:
            return np.full(horizon, np.mean(history[-5:]))
        x = np.arange(n)
        slope, intercept, _, _, _ = stats.linregress(x, history)
        # 返回趋势预测（保留残差供 DAG 使用，但在此仅返回趋势值）
        future_x = np.arange(n, n + horizon)
        trend_forecast = slope * future_x + intercept
        return trend_forecast

    def decompose(self, history):
        """供 DAG 调用的分解方法，返回 (trend, detrended)"""
        n = len(history)
        x = np.arange(n)
        slope, intercept, _, _, _ = stats.linregress(x, history)
        trend = slope * x + intercept
        detrended = history - trend
        return trend, detrended
