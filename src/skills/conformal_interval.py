import numpy as np
from .base import BaseSkill

class ConformalIntervalSkill(BaseSkill):
    """
    预测区间组合：基于多个技能的历史残差分布给出点预测和区间。
    skills: 参与组合的技能列表
    alpha: 区间显著性水平，默认0.1（即90%区间）
    """
    def __init__(self, skills=None, alpha=0.1):
        super().__init__()
        self.name = "conformal_interval"
        self.description = "多技能点预测+经验区间，提供不确定性估计"
        self.skills = skills or []
        self.alpha = alpha
        self.state_card = {
            "when_to_use": {"conditions": [], "logic": "AND"},
            "when_not_to_use": {"conditions": [], "logic": "OR"},
            "visible_cues": [],
            "verification_cue": "真实值以指定比例落在区间内",
            "failure_mode": "所有技能均失效",
            "fallback_skill": "naive"
        }

    def execute(self, history: np.ndarray, horizon: int, **kwargs) -> np.ndarray:
        """返回点预测（所有技能预测的平均值）"""
        if not self.skills:
            return np.full(horizon, np.mean(history))
        preds = []
        for skill in self.skills:
            try:
                p = skill.execute(history, horizon, **kwargs)
                preds.append(p)
            except:
                pass
        if not preds:
            return np.full(horizon, np.mean(history))
        # 点预测：所有技能的平均值
        point_forecast = np.mean(preds, axis=0)
        return point_forecast

    def predict_interval(self, history: np.ndarray, horizon: int, **kwargs):
        """返回 (点预测, 下界, 上界)"""
        point = self.execute(history, horizon, **kwargs)
        # 计算历史残差经验分位数
        residuals = []
        w = min(20, len(history)-1)
        for i in range(1, w+1):
            hist_input = history[:len(history)-w+i-1]
            actual = history[len(history)-w+i-1]
            # 用所有技能的均值作为历史拟合
            fits = []
            for skill in self.skills:
                try:
                    fits.append(skill.execute(hist_input, 1, **kwargs)[0])
                except:
                    fits.append(np.mean(hist_input[-5:]))
            if fits:
                fit_mean = np.mean(fits)
                residuals.append(actual - fit_mean)
        if residuals:
            lower_q = np.percentile(residuals, 100 * self.alpha / 2)
            upper_q = np.percentile(residuals, 100 * (1 - self.alpha / 2))
        else:
            lower_q, upper_q = -np.std(history)*2, np.std(history)*2
        lower_bound = point + lower_q
        upper_bound = point + upper_q
        return point, lower_bound, upper_bound