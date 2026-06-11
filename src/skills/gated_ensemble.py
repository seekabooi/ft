import numpy as np
from .base import BaseSkill
from src.skills.data_profiler import DataProfiler

class GatedEnsembleSkill(BaseSkill):
    """根据数据状态自动选择一组候选技能，并加权平均"""
    def __init__(self):
        super().__init__()
        self.name = "gated_ensemble"
        self.description = "根据季节性/趋势/波动强度动态选择候选技能池并加权平均"
        # 内置的技能池（由外部传入或默认使用已注册技能）
        self.available_skills = []  # 将在注册时由外部设置
        self.state_card = {
            "when_to_use": {"conditions": [], "logic": "AND"},
            "when_not_to_use": {"conditions": [], "logic": "OR"},
            "visible_cues": [],
            "verification_cue": "组合预测优于任意单个技能",
            "failure_mode": "候选池为空或所有技能失败",
            "fallback_skill": "naive"
        }

    def set_skills(self, skills: list):
        self.available_skills = skills

    def execute(self, history: np.ndarray, horizon: int, **kwargs) -> np.ndarray:
        if not self.available_skills:
            return np.full(horizon, np.mean(history))

        profile = DataProfiler.profile(history)
        seasonal = profile['seasonal_strength']
        trend = profile['trend_strength']
        vol = profile['recent_volatility']
        length = profile['data_length']

        # 定义门控规则：根据状态选择候选技能名
        selected_names = set()
        if seasonal > 0.5:
            selected_names.update(['seasonal_naive', 'prophet', 'stl_forecast', 'decompose_ensemble'])
        elif seasonal > 0.3:
            selected_names.update(['prophet', 'arima', 'auto_arima'])
        else:
            selected_names.update(['naive', 'naive_drift', 'arima', 'auto_ets'])

        if trend > 0.5:
            selected_names.update(['linear_trend', 'ets', 'theta'])
        elif trend > 0.3:
            selected_names.update(['arima', 'auto_arima', 'auto_ets'])

        if vol > 2.0:
            selected_names.update(['naive_drift', 'arima'])  # 高波动少用复杂模型

        if length < 30:
            selected_names.update(['naive', 'naive_drift'])

        # 从可用技能中挑选
        pool = [s for s in self.available_skills if s.name in selected_names]
        if not pool:
            pool = self.available_skills[:3]  # 保底选3个

        # 执行并加权平均（简单平均，可后续改进）
        preds = []
        for skill in pool:
            try:
                pred = skill.execute(history, horizon, **kwargs)
                preds.append(pred)
            except:
                continue
        if not preds:
            return np.full(horizon, np.mean(history))
        return np.mean(preds, axis=0)