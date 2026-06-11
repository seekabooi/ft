import numpy as np
from .base import BaseSkill

class EnsembleSkill(BaseSkill):
    def __init__(self, skills: list = None):
        super().__init__()
        self.name = "ensemble"
        self.description = "组合多个候选技能取平均，提高稳定性"
        self.skills = skills or []

        # === 新增元数据 ===
        self.min_data_points = 5                # 自身调度轻量，但内部技能会自行检查
        self.requires_full_history = False
        self.strength_tags = []
        self.model_family = "lightweight"

        self.state_card = {
            "when_to_use": {"conditions": [], "logic": "AND"},
            "when_not_to_use": {"conditions": [], "logic": "OR"},
            "visible_cues": [],
            "verification_cue": "",
            "available_views": []
        }

    def execute(self, history: np.ndarray, horizon: int, **kwargs) -> np.ndarray:
        if not self.skills:
            return np.full(horizon, np.mean(history[-5:]))
        preds = []
        for skill in self.skills:
            try:
                preds.append(skill.execute(history, horizon))
            except Exception:
                continue
        if not preds:
            return np.full(horizon, np.mean(history))
        return np.mean(preds, axis=0)