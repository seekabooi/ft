# experiments/autotune/reflection.py
"""
反思机制（采集归纳合并）
从模仿学习升级为偏好学习
"""

import json
import numpy as np
from typing import Dict, List, Optional, Any


class ReflectionEngine:
    """
    反思引擎

    流程：
    1. 多候选生成：LLM生成K个候选策略
    2. 本地回测排序：代码计算MASE，按好/中/差排序
    3. 对比反思：反问LLM三个问题
    4. 生成增强SkillCard：诊断规则写入
    """

    def __init__(self, config: Dict, logger):
        self.config = config
        self.logger = logger
        self.model = config.get('llm', {}).get('model', 'glm-4')

    def reflect(self, window_data: Dict, candidates: List[Dict]) -> Dict:
        """
        执行反思

        Args:
            window_data: 窗口数据（特征、轨迹等）
            candidates: 候选策略列表，每个包含 {name, stages, mase}

        Returns:
            {
                'diagnostic_rules': [...],  # 诊断规则
                'best_strategy': {...},     # 最优策略
                'reflection': {...}         # 反思结果
            }
        """
        self.logger.log("   🧠 执行反思分析...")

        # 1. 按MASE排序
        sorted_candidates = sorted(candidates, key=lambda x: x.get('mase', float('inf')))
        best = sorted_candidates[0] if sorted_candidates else None
        worst = sorted_candidates[-1] if sorted_candidates else None

        if best is None or worst is None:
            return {'diagnostic_rules': [], 'best_strategy': None}

        # 2. 构建反思Prompt
        prompt = self._build_reflection_prompt(window_data, best, worst, candidates)

        # 3. 调用LLM
        try:
            result = self._call_llm(prompt)
            diagnostic_rules = result.get('diagnostic_rules', [])

            self.logger.log(f"      ✅ 生成 {len(diagnostic_rules)} 条诊断规则")

            return {
                'diagnostic_rules': diagnostic_rules,
                'best_strategy': best,
                'reflection': {
                    'best_why': result.get('best_why', ''),
                    'worst_why': result.get('worst_why', ''),
                    'key_insights': result.get('key_insights', [])
                }
            }
        except Exception as e:
            self.logger.log(f"      ⚠️ 反思失败: {e}")
            return {'diagnostic_rules': [], 'best_strategy': best}

    def _build_reflection_prompt(self, window_data: Dict, best: Dict,
                                 worst: Dict, all_candidates: List[Dict]) -> str:
        """构建反思Prompt"""
        # 提取特征
        features = window_data.get('features', {})
        feat_str = ', '.join([f"{k}: {v:.3f}" for k, v in features.items() if isinstance(v, (int, float))])

        # 构建候选摘要
        candidate_summary = []
        for i, c in enumerate(all_candidates):
            mase = c.get('mase', float('inf'))
            name = c.get('name', f'候选{i + 1}')
            stages = c.get('stages', [])
            stage_desc = []
            for st in stages:
                steps = st.get('steps', 0)
                weights = st.get('weights', {})
                w_str = ', '.join([f"{k}:{v:.2f}" for k, v in weights.items()])
                stage_desc.append(f"{steps}步{{{w_str}}}")
            candidate_summary.append(f"  {name}: MASE={mase:.4f}, 策略={' → '.join(stage_desc)}")

        prompt = f"""你是一个时序预测策略专家。分析以下候选策略的优劣，并总结诊断规则。

窗口特征：
{feat_str}

候选策略（按MASE排序）：
{chr(10).join(candidate_summary)}

请回答以下问题：
1. 最优策略做对了什么？（best_why）
2. 最差策略踩了什么坑？（worst_why）
3. 总结2-3条可复用的诊断规则（diagnostic_rules）

输出JSON格式：
{{
    "best_why": "最优策略的成功原因",
    "worst_why": "最差策略的失败原因",
    "key_insights": ["洞察1", "洞察2", ...],
    "diagnostic_rules": [
        {{
            "condition": "触发条件（如 seasonality > 0.6）",
            "action": "建议动作（如 避免使用残差校正）",
            "reason": "原因解释"
        }}
    ]
}}
只输出JSON，不要解释。"""
        return prompt

    def _call_llm(self, prompt: str) -> Dict:
        """调用LLM"""
        try:
            from src.agents.llm_client import LLMClient
            import sys, io, re
            old_out = sys.stdout
            sys.stdout = io.StringIO()
            client = LLMClient(model=self.model)
            resp = client.call_with_retry(prompt, max_retries=2)
            sys.stdout = old_out
            content = resp.choices[0].message.content
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                return json.loads(match.group())
            return {}
        except Exception as e:
            return {}

    def apply_reflection_to_policy(self, policy, reflection: Dict) -> Dict:
        """将反思结果应用到策略"""
        diagnostic_rules = reflection.get('diagnostic_rules', [])
        if not diagnostic_rules:
            return policy

        # 提取诊断规则中的条件
        for rule in diagnostic_rules:
            cond = rule.get('condition', '')
            action = rule.get('action', '')
            if '避免' in action or '不使用' in action:
                # 负向条件
                policy['negative_condition'] = policy.get('negative_condition', {})
                # 解析条件
                if '>' in cond:
                    k, v = cond.split('>')
                    policy['negative_condition'][k.strip()] = f"> {float(v.strip())}"
                elif '<' in cond:
                    k, v = cond.split('<')
                    policy['negative_condition'][k.strip()] = f"< {float(v.strip())}"

        # 添加诊断信息
        policy['failure_modes'] = policy.get('failure_modes', [])
        policy['failure_modes'].extend(rule.get('reason', '') for rule in diagnostic_rules)
        policy['diagnostic_note'] = reflection.get('key_insights', [''])[0]

        return policy