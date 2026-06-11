import numpy as np
from .base import BaseSkill

class ValidationWeightedEnsemble(BaseSkill):
    """基于最近验证窗口误差加权的组合技能"""
    def __init__(self, skills: list = None, validation_window: int = 10):
        super().__init__()
        self.name = "weighted_ensemble"
        self.description = "根据最近验证窗口的预测误差倒数加权组合多个技能"
        self.skills = skills or []
        self.validation_window = validation_window
        self.state_card = {
            "when_to_use": {"conditions": [], "logic": "AND"},
            "when_not_to_use": {"conditions": [], "logic": "OR"},
            "visible_cues": [],
            "verification_cue": "加权预测应优于任意单个技能",
            "failure_mode": "所有技能在验证窗口表现都很差",
            "fallback_skill": "naive"
        }

    def execute(self, history: np.ndarray, horizon: int, **kwargs) -> np.ndarray:
        if not self.skills:
            return np.full(horizon, np.mean(history))

        # 使用最后 validation_window 个点作为验证集
        val_len = min(self.validation_window, len(history) - horizon)
        if val_len < 2:
            # 验证集太小，回退简单平均
            preds = []
            for s in self.skills:
                try:
                    preds.append(s.execute(history, horizon))
                except:
                    continue
            if not preds:
                return np.full(horizon, np.mean(history))
            return np.mean(preds, axis=0)

        val_history = history[:-val_len]
        val_actuals = history[-val_len:]

        weights = []
        preds_list = []

        for skill in self.skills:
            try:
                # 在验证集上滚动预测
                val_preds = []
                for i in range(val_len):
                    cur_hist = np.concatenate([val_history, val_actuals[:i]])
                    pred = skill.execute(cur_hist, horizon=1)[0]
                    val_preds.append(pred)
                errors = np.abs(np.array(val_preds) - val_actuals)
                mae = np.mean(errors)
                weights.append(1.0 / (mae + 1e-8))
                # 同时保存该技能在全部历史下的预测
                full_pred = skill.execute(history, horizon)
                preds_list.append(full_pred)
            except:
                weights.append(0.0)
                preds_list.append(np.full(horizon, np.mean(history)))

        if sum(weights) == 0:
            return np.mean(preds_list, axis=0)

        weights = np.array(weights) / sum(weights)
        ensemble_pred = np.zeros(horizon)
        for w, p in zip(weights, preds_list):
            ensemble_pred += w * p
        return ensemble_pred