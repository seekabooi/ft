# experiments/autotune/branch_loader.py
"""
分支加载（Branch Loading）
对应 MMSkills 的 branch loading + view selection 机制
★ 2026-06-25 升级：所有分析方法真正调用 LLM，并新增适用性、退休、刷新分析
★ ★ 2026-06-25 增加 analyze_adaptation 方法（Hard Window Routing）
★ ★ 2026-06-25 增加 global_summary 参数支持
★ ★ ★ 2026-06-26 增加 judge_reward 方法（LLM 作为 Reward Judge）
"""

import json
import re
import numpy as np
from typing import Dict, List, Optional, Any


class BranchLoader:
    """
    分支加载器

    流程：
    1. 状态卡 → 技能分支选择器
    2. 选择 Top-K 技能分支
    3. 在临时分支中做深入推理
    4. 返回结构化决策摘要给主流程
    """

    def __init__(self, config: Dict, logger):
        self.config = config
        self.logger = logger
        self.model = config.get('llm', {}).get('model', 'glm-4')
        self.branch_config = config.get('branch_loading', {})
        self.enabled = self.branch_config.get('enabled', True)
        if not self.enabled:
            self.logger.log("ℹ️ BranchLoader 已禁用（配置关闭）")

    # -------------------- 预测阶段：策略适用性分析 --------------------
    def analyze_applicability(self, policy, state: Dict, history: Optional[np.ndarray] = None,
                               global_summary: Optional[str] = None) -> Dict:
        """分析策略在当前状态下的适用性，返回权重调整建议"""
        if not self.enabled:
            return {'applicable': True, 'confidence': 0.5, 'suggested_weight_adjustment': 0.0}

        self.logger.log("   🔍 分支加载：分析策略适用性...")

        prompt = self._build_applicability_prompt(policy, state, history, global_summary)
        try:
            result = self._call_llm(prompt)
            return {
                'applicable': result.get('applicable', True),
                'confidence': result.get('confidence', 0.5),
                'suggested_weight_adjustment': result.get('suggested_weight_adjustment', 0.0),
                'reason': result.get('reason', ''),
                'do_not_use': result.get('do_not_use', ''),
                'verification_cue': result.get('verification_cue', '')
            }
        except Exception as e:
            self.logger.log(f"      ⚠️ 适用性分析失败: {e}")
            return {'applicable': True, 'confidence': 0.5, 'suggested_weight_adjustment': 0.0}

    # -------------------- 合并分析（升级为 LLM） --------------------
    def analyze_merge(self, policy_a, policy_b, state: Dict,
                       global_summary: Optional[str] = None) -> Dict:
        """分析两条策略是否应该合并（LLM决策）"""
        if not self.enabled:
            return self._statistical_merge(policy_a, policy_b, state)

        self.logger.log("   🔀 分支加载：LLM分析策略合并...")
        prompt = self._build_merge_prompt(policy_a, policy_b, state, global_summary)
        try:
            result = self._call_llm(prompt)
            return {
                'should_merge': result.get('should_merge', False),
                'reason': result.get('reason', ''),
                'merge_score': result.get('merge_score', 0.5),
                'new_condition': result.get('new_condition', {}),
                'merged_strategy': result.get('merged_strategy', {})
            }
        except Exception as e:
            self.logger.log(f"      ⚠️ 合并分析失败，回退统计: {e}")
            return self._statistical_merge(policy_a, policy_b, state)

    # -------------------- 退休分析（新增） --------------------
    def analyze_retire(self, policy, performance_summary: Dict,
                        global_summary: Optional[str] = None) -> Dict:
        """分析策略是否应该被退休"""
        if not self.enabled:
            return {'should_retire': False, 'confidence': 0.0, 'target_status': 'ACTIVE'}

        self.logger.log("   🗑️ 分支加载：分析退休必要性...")
        prompt = self._build_retire_prompt(policy, performance_summary, global_summary)
        try:
            result = self._call_llm(prompt)
            return {
                'should_retire': result.get('should_retire', False),
                'confidence': result.get('confidence', 0.0),
                'target_status': result.get('target_status', 'ACTIVE'),
                'reason': result.get('reason', '')
            }
        except Exception as e:
            self.logger.log(f"      ⚠️ 退休分析失败: {e}")
            return {'should_retire': False, 'confidence': 0.0, 'target_status': 'ACTIVE'}

    # -------------------- 补丁分析（升级为 LLM） --------------------
    def analyze_patch(self, policy, window_data: List[Dict], state: Dict,
                       global_summary: Optional[str] = None) -> Dict:
        """分析困难窗口需要什么样的补丁"""
        if not self.enabled:
            return {'need_patch': False}

        self.logger.log("   📝 分支加载：分析补丁需求...")
        prompt = self._build_patch_prompt(policy, window_data, state, global_summary)
        try:
            result = self._call_llm(prompt)
            return {
                'need_patch': result.get('need_patch', False),
                'issue_type': result.get('issue_type', 'unknown'),
                'suggested_fix': result.get('suggested_fix', {}),
                'confidence': result.get('confidence', 0.5)
            }
        except Exception as e:
            self.logger.log(f"      ⚠️ 补丁分析失败: {e}")
            return {'need_patch': False}

    # -------------------- 刷新分析（新增） --------------------
    def analyze_refresh(self, policy, drift_info: Dict,
                         global_summary: Optional[str] = None) -> Dict:
        """分析策略是否需要刷新，以及刷新方向"""
        if not self.enabled:
            return {'should_refresh': False, 'confidence': 0.0}

        self.logger.log("   🔄 分支加载：分析刷新需求...")
        prompt = self._build_refresh_prompt(policy, drift_info, global_summary)
        try:
            result = self._call_llm(prompt)
            return {
                'should_refresh': result.get('should_refresh', False),
                'new_embedding_direction': result.get('new_embedding_direction', []),
                'new_condition_adjustments': result.get('new_condition_adjustments', {}),
                'confidence': result.get('confidence', 0.5),
                'reason': result.get('reason', '')
            }
        except Exception as e:
            self.logger.log(f"      ⚠️ 刷新分析失败: {e}")
            return {'should_refresh': False, 'confidence': 0.0}

    # -------------------- 适配分析（Hard Window Routing） --------------------
    def analyze_adaptation(self, policy, window_features: Dict,
                            window_id: Optional[int] = None) -> Dict:
        """分析策略是否可以通过微调适配困难窗口"""
        if not self.enabled:
            return {'can_adapt': False, 'confidence': 0.0}

        self.logger.log(f"   🔧 分支加载：分析策略适配... (window={window_id})")
        prompt = self._build_adaptation_prompt(policy, window_features, window_id)
        try:
            result = self._call_llm(prompt)
            return {
                'can_adapt': result.get('can_adapt', False),
                'confidence': result.get('confidence', 0.0),
                'suggested_changes': result.get('suggested_changes', {}),
                'reason': result.get('reason', '')
            }
        except Exception as e:
            self.logger.log(f"      ⚠️ 适配分析失败: {e}")
            return {'can_adapt': False, 'confidence': 0.0}

    # ★★★ 新增：Reward Judge（LLM 判断 reward 质量） ★★★
    def judge_reward(self, prediction: np.ndarray, actual: np.ndarray,
                     history: np.ndarray, window_id: Optional[int] = None) -> Dict:
        """
        LLM 判断 reward 质量

        Returns:
            {
                'reward_quality': 'high' | 'medium' | 'low',
                'confidence': 0.0~1.0,
                'reason': str
            }
        """
        if not self.enabled:
            return {'reward_quality': 'medium', 'confidence': 0.5, 'reason': 'LLM disabled'}

        self.logger.log(f"   📊 分支加载：LLM 判断 Reward 质量 (window={window_id})")

        prompt = self._build_reward_judge_prompt(prediction, actual, history, window_id)
        try:
            result = self._call_llm(prompt)
            return {
                'reward_quality': result.get('reward_quality', 'medium'),
                'confidence': result.get('confidence', 0.5),
                'reason': result.get('reason', '')
            }
        except Exception as e:
            self.logger.log(f"      ⚠️ Reward Judge 失败: {e}")
            return {'reward_quality': 'medium', 'confidence': 0.3, 'reason': f'LLM error: {e}'}

    # -------------------- 内部辅助方法 --------------------
    def _statistical_merge(self, policy_a, policy_b, state):
        """回退统计合并逻辑"""
        feature_sim = self._feature_similarity(policy_a, policy_b)
        behavior_sim = self._behavior_similarity(policy_a, policy_b)
        outcome_sim = self._outcome_similarity(policy_a, policy_b)
        total_sim = 0.5 * feature_sim + 0.3 * behavior_sim + 0.2 * outcome_sim
        if total_sim > 0.7:
            merged_condition = self._merge_conditions(policy_a.state_condition, policy_b.state_condition)
            merged_strategy = self._merge_strategies(policy_a.skill_strategy, policy_b.skill_strategy)
            return {
                'should_merge': True,
                'reason': f'总相似度 {total_sim:.2f} > 0.7',
                'merge_score': total_sim,
                'new_condition': merged_condition,
                'merged_strategy': merged_strategy
            }
        else:
            return {
                'should_merge': False,
                'reason': f'总相似度 {total_sim:.2f} < 0.7',
                'merge_score': total_sim,
                'new_condition': {},
                'merged_strategy': {}
            }

    def _build_applicability_prompt(self, policy, state, history, global_summary):
        features = state.get('numeric', {})
        lines = [
            f"你是一个策略评估专家。现有策略 {policy.name}，其语义描述为：{policy.semantic_description or '无'}。",
            f"状态条件：{policy.state_condition}，特征组：{policy.feature_groups}，平均MASE：{policy.avg_mase:.4f}。",
            f"当前状态特征：{features}。"
        ]
        if global_summary:
            lines.append(f"\n{global_summary}")
        lines.append("""请判断该策略是否适用于当前状态，并给出：
1. 适用性判断（true/false）
2. 置信度（0~1）
3. 权重调整建议（相对值，如 +0.05 或 -0.03，范围 -0.2~0.2）
4. 简要理由
5. 若不适用，应避免什么操作
6. 验证该策略是否有效的关键特征提示（不涉及具体数值）

输出JSON格式：
{
    "applicable": true/false,
    "confidence": 0.5,
    "suggested_weight_adjustment": 0.0,
    "reason": "简短理由",
    "do_not_use": "若不适用，说明原因",
    "verification_cue": "关键验证提示"
}
只输出JSON，不要解释。""")
        return '\n'.join(lines)

    def _build_merge_prompt(self, policy_a, policy_b, state, global_summary):
        lines = [
            "你是一个策略合并决策专家。你的目标是**尽量合并冗余策略**，以精简策略池，同时保持性能。",
            "只有在合并会明显损害性能或导致条件冲突时才拒绝合并。",
            "",
            f"策略A: {policy_a.name}（{policy_a.semantic_description or '无描述'}）",
            f"  - 条件: {policy_a.state_condition}",
            f"  - 特征组: {policy_a.feature_groups}",
            f"  - 平均MASE: {policy_a.avg_mase:.4f}",
            f"  - 效用: {policy_a.utility_ema:.3f}",
            f"  - 簇: {getattr(policy_a, 'cluster_id', '未分配')}",
            "",
            f"策略B: {policy_b.name}（{policy_b.semantic_description or '无描述'}）",
            f"  - 条件: {policy_b.state_condition}",
            f"  - 特征组: {policy_b.feature_groups}",
            f"  - 平均MASE: {policy_b.avg_mase:.4f}",
            f"  - 效用: {policy_b.utility_ema:.3f}",
            f"  - 簇: {getattr(policy_b, 'cluster_id', '未分配')}",
        ]
        if global_summary:
            lines.append(f"\n{global_summary}")
        lines.append("""请判断合并是否合理。**除非存在明确的性能下降风险或条件冲突，否则应倾向于合并。**
给出：
- should_merge: true/false（尽量为 true）
- merge_score: 0~1 的合并得分（反映合并的合理性）
- reason: 理由
- new_condition: 合并后的新条件（若合并，建议取两者的并集）
- merged_strategy: 合并后的策略组合（若合并，选择效用更高的策略结构）

输出JSON格式，只输出JSON。""")
        return '\n'.join(lines)

    def _build_retire_prompt(self, policy, perf_summary, global_summary):
        lines = [
            "你是一个策略退休决策专家。评估以下策略是否应该被淘汰。",
            "",
            f"策略: {policy.name}（{policy.semantic_description or '无描述'}）",
            f"- 状态: {policy.status}",
            f"- 平均MASE: {policy.avg_mase:.4f}",
            f"- 效用: {policy.utility_ema:.3f}",
            f"- 激活次数: {policy.activation_count}",
            f"- 边际价值: {policy.marginal_value:.4f}",
            f"- 簇: {getattr(policy, 'cluster_id', '未分配')}",
            f"\n性能摘要：{perf_summary}"
        ]
        if global_summary:
            lines.append(f"\n{global_summary}")
        lines.append("""请判断是否应该退休，并给出目标状态（DEPRECATED 或 ARCHIVE）及理由。
输出JSON:
{
    "should_retire": true/false,
    "confidence": 0.0~1.0,
    "target_status": "DEPRECATED" 或 "ARCHIVE",
    "reason": "理由"
}""")
        return '\n'.join(lines)

    def _build_patch_prompt(self, policy, window_data, state, global_summary):
        lines = [
            "你是一个补丁决策专家。分析困难窗口是否适合打补丁。",
            "",
            f"策略: {policy.name}",
            f"困难窗口数: {len(window_data)}",
            f"窗口特征示例: {window_data[0] if window_data else {}}",
            f"当前状态: {state.get('numeric', {})}"
        ]
        if global_summary:
            lines.append(f"\n{global_summary}")
        lines.append("""判断是否需要补丁，输出JSON:
{
    "need_patch": true/false,
    "issue_type": "overestimation|underestimation|volatility_misread|trend_misread|seasonal_misread",
    "suggested_fix": {"field": "suggested_value"},
    "confidence": 0.0~1.0
}""")
        return '\n'.join(lines)

    def _build_refresh_prompt(self, policy, drift_info, global_summary):
        lines = [
            "你是一个策略刷新决策专家。评估策略是否应该刷新（更新embedding和条件）。",
            "",
            f"策略: {policy.name}（{policy.semantic_description or '无描述'}）",
            f"漂移信息: {drift_info}"
        ]
        if global_summary:
            lines.append(f"\n{global_summary}")
        lines.append("""输出JSON:
{
    "should_refresh": true/false,
    "new_embedding_direction": [0.1, -0.05, ...],
    "new_condition_adjustments": {"feature": "new_operator value"},
    "confidence": 0.0~1.0,
    "reason": "理由"
}""")
        return '\n'.join(lines)

    def _build_adaptation_prompt(self, policy, window_features, window_id):
        return f"""你是一个策略适配专家。判断现有策略是否可以通过微调来适配困难窗口。

策略: {policy.name}（{policy.semantic_description or '无描述'}）
- 状态条件: {policy.state_condition}
- 特征组: {policy.feature_groups}
- 平均MASE: {policy.avg_mase:.4f}

困难窗口特征:
- {window_features}
- 窗口ID: {window_id}

请判断：
1. 该策略是否可以通过微调适配这个窗口（can_adapt）
2. 建议的调整方向（suggested_changes）
3. 置信度（confidence）
4. 理由（reason）

输出JSON:
{{
    "can_adapt": true/false,
    "confidence": 0.0~1.0,
    "suggested_changes": {{"feature": "adjustment"}},
    "reason": "理由"
}}
只输出JSON，不要解释。"""

    # ★★★ 新增：Reward Judge Prompt ★★★
    def _build_reward_judge_prompt(self, prediction: np.ndarray, actual: np.ndarray,
                                   history: np.ndarray, window_id: Optional[int] = None) -> str:
        pred_str = ', '.join([f'{x:.2f}' for x in prediction[:10]])
        actual_str = ', '.join([f'{x:.2f}' for x in actual[:10]])
        hist_str = ', '.join([f'{x:.2f}' for x in history[-20:]])

        return f"""你是一个预测质量评估专家。判断以下预测结果的 reward 是否可信。

窗口ID: {window_id}

历史数据（最近20点）:
[{hist_str}]

预测值（前10点）:
[{pred_str}]

真实值（前10点）:
[{actual_str}]

请判断：
1. 预测质量（reward_quality）：high / medium / low
2. 置信度（confidence）：0.0~1.0
3. 简短理由（reason）

输出JSON:
{{
    "reward_quality": "high" | "medium" | "low",
    "confidence": 0.5,
    "reason": "简短理由"
}}
只输出JSON，不要解释。"""

    def _call_llm(self, prompt: str) -> Dict:
        """调用 LLM 并解析 JSON"""
        try:
            from src.agents.llm_client import LLMClient
            import sys, io
            old_out = sys.stdout
            sys.stdout = io.StringIO()
            client = LLMClient(model=self.model, logger=self.logger)
            resp = client.call_with_retry(prompt, max_retries=2)
            sys.stdout = old_out
            content = resp.choices[0].message.content
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                return json.loads(match.group())
            return {}
        except Exception as e:
            self.logger.log(f"      ❌ LLM调用失败: {e}")
            return {}

    # -------------------- 原有统计方法（保留用于回退） --------------------
    def _feature_similarity(self, a, b) -> float:
        keys = set(a.state_condition.keys()) | set(b.state_condition.keys())
        if not keys:
            return 0.5
        matches = 0
        for key in keys:
            val_a = a.state_condition.get(key, '')
            val_b = b.state_condition.get(key, '')
            if val_a == val_b and val_a != '':
                matches += 1
        return matches / len(keys) if keys else 0.5

    def _behavior_similarity(self, a, b) -> float:
        stages_a = a.skill_strategy.get('stages', [])
        stages_b = b.skill_strategy.get('stages', [])
        if not stages_a or not stages_b:
            return 0.5
        skills_a = set()
        skills_b = set()
        for stage in stages_a:
            skills_a.update(stage.get('weights', {}).keys())
        for stage in stages_b:
            skills_b.update(stage.get('weights', {}).keys())
        if not skills_a or not skills_b:
            return 0.5
        intersection = len(skills_a & skills_b)
        union = len(skills_a | skills_b)
        stage_sim = 1 - abs(len(stages_a) - len(stages_b)) / max(len(stages_a), len(stages_b))
        skill_sim = intersection / union if union > 0 else 0
        return 0.5 * stage_sim + 0.5 * skill_sim

    def _outcome_similarity(self, a, b) -> float:
        mase_a = a.avg_mase
        mase_b = b.avg_mase
        if mase_a <= 0 or mase_b <= 0:
            return 0.5
        return min(mase_a, mase_b) / max(mase_a, mase_b)

    def _merge_conditions(self, cond_a: Dict, cond_b: Dict) -> Dict:
        merged = {}
        all_keys = set(cond_a.keys()) | set(cond_b.keys())
        for key in all_keys:
            val_a = cond_a.get(key, '')
            val_b = cond_b.get(key, '')
            if val_a and val_b and val_a != val_b:
                merged[key] = f"OR({val_a}, {val_b})"
            elif val_a:
                merged[key] = val_a
            elif val_b:
                merged[key] = val_b
        return merged

    def _merge_strategies(self, strat_a: Dict, strat_b: Dict) -> Dict:
        stages_a = strat_a.get('stages', [])
        stages_b = strat_b.get('stages', [])
        if not stages_a:
            return strat_b
        if not stages_b:
            return strat_a
        return strat_a if len(stages_a) >= len(stages_b) else strat_b