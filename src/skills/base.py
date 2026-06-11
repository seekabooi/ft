from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, List, Any

class BaseSkill(ABC):
    def __init__(self):
        self.name = "base"
        self.description = ""
        self.prototypes: List[Dict] = []
        self.state_card: Dict = self._default_state_card()

        # === 原有元数据 ===
        self.min_data_points = 3
        self.requires_full_history = False
        self.strength_tags: List[str] = []
        self.model_family = "lightweight"

        # === 按需特征列表 ===
        self.required_features: List[str] = [
            "seasonal_strength", "trend_strength", "adf_pvalue",
            "data_length", "period"
        ]

        # === LLM 决策提示 ===
        self.decision_hint: str = ""

        # === 新增：推荐序列长度范围（长序列优化） ===
        self.preferred_length_range: tuple = (0, float('inf'))  # (min_len, max_len)

    def _default_state_card(self) -> Dict:
        return {
            "when_to_use": {"conditions": [], "logic": "AND"},
            "when_not_to_use": {"conditions": [], "logic": "OR"},
            "visible_cues": [],
            "verification_cue": "",
            "failure_mode": "预测值超出历史波动3倍标准差",
            "fallback_skill": "naive"
        }

    @abstractmethod
    def execute(self, history: np.ndarray, horizon: int, **kwargs) -> np.ndarray:
        pass

    def evaluate_condition(self, condition: Dict, profile: Dict) -> bool:
        field = condition['field']
        op = condition['op']
        value = condition['value']
        actual = profile.get(field, 0)
        if op == '<': return actual < value
        elif op == '>': return actual > value
        elif op == '<=': return actual <= value
        elif op == '>=': return actual >= value
        elif op == '==': return actual == value
        return False

    def check_state_card(self, profile: Dict) -> Dict:
        wtu = self.state_card.get('when_to_use', {})
        wntu = self.state_card.get('when_not_to_use', {})
        wtu_cond = wtu.get('conditions', [])
        wntu_cond = wntu.get('conditions', [])
        wtu_logic = wtu.get('logic', 'AND')
        wntu_logic = wntu.get('logic', 'OR')

        if wtu_cond:
            res = [self.evaluate_condition(c, profile) for c in wtu_cond]
            wtu_pass = all(res) if wtu_logic == 'AND' else any(res)
        else:
            wtu_pass = True

        if wntu_cond:
            res = [self.evaluate_condition(c, profile) for c in wntu_cond]
            wntu_triggered = all(res) if wntu_logic == 'AND' else any(res)
        else:
            wntu_triggered = False

        applicable = wtu_pass and not wntu_triggered

        # 额外检查序列长度是否在推荐范围内
        if applicable and self.preferred_length_range[1] != float('inf'):
            data_len = profile.get('data_length', 0)
            if data_len < self.preferred_length_range[0] or data_len > self.preferred_length_range[1]:
                applicable = False

        return {
            'applicable': applicable,
            'visible_cues': self.state_card.get('visible_cues', []),
            'verification_cue': self.state_card.get('verification_cue', ''),
            'failure_mode': self.state_card.get('failure_mode', ''),
            'fallback_skill': self.state_card.get('fallback_skill', 'naive')
        }

    def verify_prediction(self, history: np.ndarray, prediction: float) -> Dict:
        hist_std = np.std(history) if len(history) > 1 else 1.0
        hist_mean = np.mean(history)
        z = (prediction - hist_mean) / hist_std if hist_std > 0 else 0
        return {'valid': abs(z) < 3.0, 'z_score': round(z, 2)}

    def to_function_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {"horizon": {"type": "integer"}},
                    "required": ["horizon"]
                }
            }
        }