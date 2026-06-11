import numpy as np
from .base import BaseSkill

class ResidualCorrectionSkill(BaseSkill):
    def __init__(self, base_skill=None, residual_window=20):
        super().__init__()
        self.name = "residual_correction"
        self.description = "基础预测 + 残差均值修正"
        self.base_skill = base_skill
        self.residual_window = residual_window
        self.min_data_points = 10
        self.requires_full_history = True
        self.strength_tags = []
        self.model_family = "stat_model"
        self.required_features = ["data_length"]   # 自身仅需长度，基础技能由调用方决定
        self.decision_hint = "修正系统性偏差的辅助技能，必须与另一个主技能搭配使用。当主模型存在持续偏差时加入，权重不超过 0.15，避免过度修正。"
        self.state_card = {
            "when_to_use": {"conditions": [], "logic": "AND"},
            "when_not_to_use": {"conditions": [], "logic": "OR"},
            "visible_cues": ["基础模型存在系统性偏差"],
            "verification_cue": "修正后残差白噪声",
            "fallback_skill": "naive"
        }

    def execute(self, history, horizon, **kwargs):
        if self.base_skill is None:
            return np.full(horizon, np.mean(history))
        base_forecast = self.base_skill.execute(history, horizon, **kwargs)
        w = min(self.residual_window, len(history) - 1)
        if w < 2:
            return base_forecast
        hist_fits, hist_actuals = [], history[-w:]
        for i in range(1, w + 1):
            hist_input = history[:len(history) - w + i - 1]
            try:
                pred = self.base_skill.execute(hist_input, 1, **kwargs)[0]
            except:
                pred = np.mean(hist_input[-5:])
            hist_fits.append(pred)
        hist_residuals = hist_actuals - np.array(hist_fits)
        future_residual = np.full(horizon, np.mean(hist_residuals))
        return base_forecast + future_residual