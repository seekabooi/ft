# experiments/autotune/skill_card.py
"""
Legacy Adapter

只做序列化转换，不参与学习
"""

from typing import Dict, List, Optional, Any

from experiments.autotune.skill_policy import SkillPolicy, create_policy_from_legacy_rule


class SkillCard:
    """
    Legacy Adapter

    降级为适配器，只负责将旧格式转换为 SkillPolicy
    """

    def __init__(self, data: Dict):
        self._data = data
        self._policy = None

    def to_policy(self) -> SkillPolicy:
        """转换为 SkillPolicy"""
        if self._policy is None:
            rule = {
                'name': self._data.get('name', 'unnamed'),
                'condition': self._extract_condition(),
                'skill_strategy': {'stages': self._data.get('skill_chain', [])},
                'confidence': self._data.get('confidence', 0.5),
                'metadata': self._data.get('metadata', {})
            }
            self._policy = create_policy_from_legacy_rule(rule)
        return self._policy

    def _extract_condition(self) -> str:
        when_to_use = self._data.get('when_to_use', {})
        if not when_to_use:
            return 'True'
        conditions = []
        for k, v in when_to_use.items():
            if isinstance(v, str):
                conditions.append(f"{k} {v}")
            else:
                conditions.append(f"{k} == {v}")
        return ' and '.join(conditions)

    def to_dict(self) -> Dict:
        return self._data

    @classmethod
    def from_rule(cls, rule: Dict) -> 'SkillCard':
        """从规则创建 SkillCard"""
        return cls(rule)