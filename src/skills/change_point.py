import numpy as np
from .base import BaseSkill

class ChangePointSkill(BaseSkill):
    """
    检测结构突变，如果突变存在则仅使用突变点之后的数据进行预测（默认使用最新段）。
    内部使用 ruptures 库或简单离差方法。
    """
    def __init__(self, base_skill=None, method='pelt', min_size=10):
        super().__init__()
        self.name = "change_point"
        self.description = "检测结构突变并分段建模，适合存在突变的序列"
        self.base_skill = base_skill
        self.method = method
        self.min_size = min_size
        self.state_card = {
            "when_to_use": {
                "conditions": [
                    {"field": "recent_volatility", "op": ">", "value": 1.5},
                    {"field": "data_length", "op": ">=", "value": 30}
                ],
                "logic": "AND"
            },
            "when_not_to_use": {
                "conditions": [
                    {"field": "data_length", "op": "<", "value": 20}
                ],
                "logic": "OR"
            },
            "visible_cues": ["序列存在明显的均值或方差突变"],
            "verification_cue": "突变点合理，且分段后预测误差降低",
            "failure_mode": "误检或无突变时浪费数据",
            "fallback_skill": "naive"
        }

    def execute(self, history: np.ndarray, horizon: int, **kwargs) -> np.ndarray:
        if self.base_skill is None:
            return np.full(horizon, np.mean(history))
        # 尝试检测突变点
        change_points = self._detect_changes(history)
        if change_points:
            # 取最后一个突变点之后的数据
            last_cp = change_points[-1]
            if len(history) - last_cp > self.min_size:
                recent = history[last_cp:]
            else:
                recent = history  # 分段太短则使用全部
        else:
            recent = history
        # 使用基技能在 recent 上预测
        return self.base_skill.execute(recent, horizon, **kwargs)

    def _detect_changes(self, arr):
        try:
            import ruptures as rpt
            # 使用 Pelt 方法检测均值变化
            algo = rpt.Pelt(model="rbf", min_size=self.min_size).fit(arr)
            change_points = algo.predict(pen=10)
            # 返回突变索引列表（排除最后一个即序列末尾）
            return [cp for cp in change_points if cp < len(arr)]
        except ImportError:
            # 回退到简单方法：找最大离差点
            if len(arr) < self.min_size * 2:
                return []
            mean = np.mean(arr)
            std = np.std(arr)
            anomalies = np.where(np.abs(arr - mean) > 2.5 * std)[0]
            if len(anomalies) == 0:
                return []
            # 简单分两组：异常点之前和之后
            split = anomalies[-1] + 1
            if split > self.min_size and len(arr) - split > self.min_size:
                return [split]
            return []