import numpy as np
from .base import BaseSkill

class BiasCorrectorSkill(BaseSkill):
    def __init__(self):
        super().__init__()
        self.name = "bias_corrector"
        self.description = "偏差修正：基于近期残差均值调整预测（兼容加权求和和 DAG）"
        self.min_data_points = 5
        self.requires_full_history = True
        self.strength_tags = ["corrector"]
        self.model_family = "lightweight"
        self.required_features = ["data_length"]
        self.decision_hint = (
            "根据最近几步的预测残差均值，对任意预测进行平移修正。"
            "在加权求和中可作为微调项，权重通常为 0.02~0.08。"
        )
        self.state_card = {
            "when_to_use": {"conditions": [], "logic": "AND"},
            "when_not_to_use": {"conditions": [], "logic": "OR"},
            "visible_cues": ["预测存在系统性偏差"],
            "verification_cue": "修正后残差白噪声",
            "fallback_skill": "naive"
        }

    def execute(self, history, horizon, **kwargs):
        # 在没有基础预测的情况下，返回零修正
        # DAG 中会传入初步预测，这里仅作为占位
        return np.zeros(horizon)

    def correct(self, raw_forecast, history, window=12):
        """真正的修正逻辑：用历史末尾 window 个点的残差均值调整 raw_forecast"""
        n = len(history)
        if n < 2:
            return raw_forecast
        # 计算最近 window 个点的朴素残差（用季节性朴素作为基准）
        from .seasonal_naive import SeasonalNaiveSkill
        naive = SeasonalNaiveSkill()
        try:
            # 生成过去 window 个点的朴素预测
            test_hist = history[:-1]
            # 简单起见，用最后几个点的误差
            errors = []
            for i in range(min(window, n - 1)):
                hist = history[:n - window + i]
                pred = naive.execute(hist, 1)[0]
                actual = history[n - window + i]
                errors.append(actual - pred)
            bias = np.mean(errors) if errors else 0.0
        except:
            bias = 0.0
        return raw_forecast + bias