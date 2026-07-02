# experiments/autotune/merge_simulator.py
"""
Merge Simulator（P6 + P11）
两阶段筛选 + 模拟验证
★ 增量优化：添加缓存 + 预过滤（不改变任何原有逻辑）
★ ★ 2026-06-25 增加 cluster_id 过滤，支持簇内合并
★ ★ ★ 2026-06-27 增加 stability-aware 概率门控（Patch 1）
"""

import numpy as np
import random
from typing import Dict, List, Optional, Any, Tuple


class MergeSimulator:
    """
    合并模拟器（P6）

    两阶段：
        Stage 1: 廉价筛选（corr > 0.8 或 merge_score > 0.7）→ 2~3个候选
        Stage 2: Merge Simulation（模拟合并 → 验证 → score_after >= score_before）

    Merge Score（P6）：
        行为相似度 40% + 状态相似度 30% + 效用相似度 20% + 持续性 10%

    两级机制（P6）：
        score > 0.85 → 直接Merge
        0.70 < score < 0.85 → 观察3周期后Merge
    """

    def __init__(self, config: Dict, logger):
        self.config = config
        self.logger = logger

        merge_cfg = config.get('merge', {})
        cheap_cfg = merge_cfg.get('cheap_filter', {})
        sim_cfg = merge_cfg.get('simulation', {})
        cand_cfg = merge_cfg.get('candidate_threshold', {})

        self.corr_threshold = cheap_cfg.get('corr_threshold', 0.8)
        self.merge_score_threshold = cheap_cfg.get('merge_score_threshold', 0.7)

        self.validation_required = sim_cfg.get('validation_required', True)

        self.direct_merge_threshold = cand_cfg.get('direct_merge', 0.85)
        self.observe_merge_threshold = cand_cfg.get('observe_merge', 0.70)
        self.observe_periods = cand_cfg.get('observe_periods', 3)

        self._observation_list = {}
        self._similarity_cache = {}
        self._cache_max_size = 100

        # ★★★ Patch 1：Evolution Soft Gate 参数 ★★★
        rl_cfg = config.get('rl', {})
        self.evolution_effect_strength = rl_cfg.get('evolution_effect_strength', 0.30)
        self._merge_proposals = []  # 记录被跳过的合并提案

    def compute_merge_score(self, policy_a, policy_b, state_samples: List[Dict]) -> float:
        """计算合并分数（P6）"""
        pair_key = tuple(sorted([policy_a.policy_id, policy_b.policy_id]))
        if pair_key in self._similarity_cache:
            return self._similarity_cache[pair_key]

        behavior_score = self._compute_behavior_similarity(policy_a, policy_b)
        state_score = self._compute_state_similarity(policy_a, policy_b, state_samples)
        utility_score = 1.0 - abs(policy_a.utility_ema - policy_b.utility_ema)

        pair_key = tuple(sorted([policy_a.policy_id, policy_b.policy_id]))
        if pair_key in self._observation_list:
            periods = self._observation_list[pair_key]['periods']
            persistence = min(1.0, periods / self.observe_periods)
        else:
            persistence = 0.0

        merge_score = (
                0.40 * behavior_score +
                0.30 * state_score +
                0.20 * utility_score +
                0.10 * persistence
        )

        self._similarity_cache[pair_key] = merge_score
        if len(self._similarity_cache) > self._cache_max_size:
            keys = list(self._similarity_cache.keys())[:self._cache_max_size // 2]
            for k in keys:
                del self._similarity_cache[k]

        return merge_score

    def _compute_behavior_similarity(self, policy_a, policy_b) -> float:
        vec_a = np.array([policy_a.utility_ema, policy_a.coverage_rate, policy_a.win_rate])
        vec_b = np.array([policy_b.utility_ema, policy_b.coverage_rate, policy_b.win_rate])

        if np.linalg.norm(vec_a) == 0 or np.linalg.norm(vec_b) == 0:
            return 0.0

        cos_sim = np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
        return max(0.0, min(1.0, (cos_sim + 1.0) / 2.0))

    def _compute_state_similarity(self, policy_a, policy_b, state_samples: List[Dict]) -> float:
        if policy_a.embedding and policy_b.embedding:
            vec_a = np.array(policy_a.embedding)
            vec_b = np.array(policy_b.embedding)

            if np.linalg.norm(vec_a) > 0 and np.linalg.norm(vec_b) > 0:
                emb_sim = np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
                emb_sim = max(0.0, min(1.0, (emb_sim + 1.0) / 2.0))
            else:
                emb_sim = 0.5
        else:
            emb_sim = 0.5

        overlap = self._compute_activation_overlap(policy_a, policy_b)
        return 0.6 * emb_sim + 0.4 * overlap

    def _compute_activation_overlap(self, policy_a, policy_b) -> float:
        keys_a = set(policy_a.state_condition.keys())
        keys_b = set(policy_b.state_condition.keys())

        if not keys_a or not keys_b:
            return 0.5

        intersection = len(keys_a & keys_b)
        union = len(keys_a | keys_b)

        if union == 0:
            return 0.5

        return intersection / union

    def stage1_filter(self, policies: List, state_samples: List[Dict],
                      cluster_id: Optional[str] = None) -> List[Tuple]:
        """
        Stage 1: 廉价筛选（P6）

        Args:
            policies: 策略列表
            state_samples: 状态样本
            cluster_id: 可选，只在该簇内筛选；若为 None，全局筛选

        Returns:
            List of (policy_a, policy_b, merge_score)
        """
        if len(policies) < 2:
            return []

        candidates = []

        # ★ 如果指定了 cluster_id，只筛选该簇内的策略
        if cluster_id is not None:
            cluster_policies = [p for p in policies if getattr(p, 'cluster_id', None) == cluster_id]
            if len(cluster_policies) < 2:
                return []
            policies_to_check = cluster_policies
        else:
            policies_to_check = [p for p in policies if p.status in ['ACTIVE', 'TRIAL']]

        check_limit = min(8, len(policies_to_check))
        active_policies = policies_to_check[:check_limit]

        for i in range(len(active_policies)):
            for j in range(i + 1, len(active_policies)):
                policy_a = active_policies[i]
                policy_b = active_policies[j]

                groups_a = set(policy_a.feature_groups)
                groups_b = set(policy_b.feature_groups)
                if groups_a and groups_b and len(groups_a & groups_b) == 0:
                    continue

                if policy_a.in_cooldown or policy_b.in_cooldown:
                    continue

                merge_score = self.compute_merge_score(policy_a, policy_b, state_samples)

                if merge_score > self.merge_score_threshold:
                    candidates.append((policy_a, policy_b, merge_score))

        candidates.sort(key=lambda x: x[2], reverse=True)
        return candidates[:3]

    # ★★★ 修改：stage2_simulate 增加 stability-aware 概率门控 ★★★
    def stage2_simulate(self, candidate_pairs: List[Tuple], validation_func) -> List[Dict]:
        """Stage 2: 模拟验证（P6）+ stability-aware 概率门控"""
        results = []

        if not self.validation_required:
            for policy_a, policy_b, merge_score in candidate_pairs:
                # ★★★ Patch 1：计算 stability_score 和触发概率 ★★★
                stability_a = policy_a.get_stability_score() if hasattr(policy_a, 'get_stability_score') else 0.5
                stability_b = policy_b.get_stability_score() if hasattr(policy_b, 'get_stability_score') else 0.5
                avg_stability = (stability_a + stability_b) / 2.0

                # p_evo = strength * (1 - stability) → 不稳定策略更易被合并
                p_evo = self.evolution_effect_strength * (1.0 - avg_stability)
                p_evo = max(0.05, min(0.4, p_evo))  # 限幅 [5%, 40%]

                # 记录日志
                self.logger.log(f"   🔀 Merge 候选: A={policy_a.name[:8]}(st={stability_a:.3f}) + "
                                f"B={policy_b.name[:8]}(st={stability_b:.3f}) → "
                                f"score={merge_score:.3f}, p_evo={p_evo:.3f}")

                if merge_score >= self.direct_merge_threshold:
                    action = 'direct'
                    should_merge = True
                elif merge_score >= self.observe_merge_threshold:
                    pair_key = tuple(sorted([policy_a.policy_id, policy_b.policy_id]))
                    if pair_key not in self._observation_list:
                        self._observation_list[pair_key] = {'score': merge_score, 'periods': 0}
                    self._observation_list[pair_key]['periods'] += 1

                    if self._observation_list[pair_key]['periods'] >= self.observe_periods:
                        action = 'observe'
                        should_merge = True
                    else:
                        action = 'observing'
                        should_merge = False
                else:
                    action = 'reject'
                    should_merge = False

                # ★★★ Patch 1：概率门控（即使 should_merge=True，也按概率执行） ★★★
                if should_merge:
                    if random.random() < p_evo:
                        self.logger.log(f"      ✅ 已执行合并 (p={p_evo:.3f})")
                        # 保留原有执行逻辑
                        pass
                    else:
                        # 仅记录提案，不执行
                        self.logger.log(f"      ⏭️ 已跳过合并 (仅记录提案, p={p_evo:.3f})")
                        should_merge = False
                        action = 'proposed'

                results.append({
                    'policy_a': policy_a,
                    'policy_b': policy_b,
                    'merge_score': merge_score,
                    'validation_before': merge_score,
                    'validation_after': merge_score,
                    'should_merge': should_merge,
                    'action': action,
                    'p_evo': p_evo,
                    'avg_stability': avg_stability
                })
            return results

        for policy_a, policy_b, merge_score in candidate_pairs:
            # ★★★ Patch 1：计算 stability_score 和触发概率 ★★★
            stability_a = policy_a.get_stability_score() if hasattr(policy_a, 'get_stability_score') else 0.5
            stability_b = policy_b.get_stability_score() if hasattr(policy_b, 'get_stability_score') else 0.5
            avg_stability = (stability_a + stability_b) / 2.0

            p_evo = self.evolution_effect_strength * (1.0 - avg_stability)
            p_evo = max(0.05, min(0.4, p_evo))

            self.logger.log(f"   🔀 Merge 候选: A={policy_a.name[:8]}(st={stability_a:.3f}) + "
                            f"B={policy_b.name[:8]}(st={stability_b:.3f}) → "
                            f"score={merge_score:.3f}, p_evo={p_evo:.3f}")

            merged_policies = self._merge_policies(policy_a, policy_b)

            baseline = validation_func([policy_a, policy_b])
            merged_score = validation_func(merged_policies)

            if merge_score >= self.direct_merge_threshold and merged_score >= baseline:
                action = 'direct'
                should_merge = True
            elif merge_score >= self.observe_merge_threshold:
                pair_key = tuple(sorted([policy_a.policy_id, policy_b.policy_id]))
                if pair_key not in self._observation_list:
                    self._observation_list[pair_key] = {'score': merge_score, 'periods': 0}
                self._observation_list[pair_key]['periods'] += 1

                if self._observation_list[pair_key]['periods'] >= self.observe_periods:
                    action = 'observe'
                    should_merge = True
                else:
                    action = 'observing'
                    should_merge = False
            else:
                action = 'reject'
                should_merge = False

            # ★★★ Patch 1：概率门控 ★★★
            if should_merge:
                if random.random() < p_evo:
                    self.logger.log(f"      ✅ 已执行合并 (p={p_evo:.3f})")
                else:
                    self.logger.log(f"      ⏭️ 已跳过合并 (仅记录提案, p={p_evo:.3f})")
                    should_merge = False
                    action = 'proposed'

            results.append({
                'policy_a': policy_a,
                'policy_b': policy_b,
                'merge_score': merge_score,
                'validation_before': baseline,
                'validation_after': merged_score,
                'should_merge': should_merge,
                'action': action,
                'p_evo': p_evo,
                'avg_stability': avg_stability
            })

        return results

    def _merge_policies(self, policy_a, policy_b) -> List:
        """
        合并两条策略（保留A吸收B，不生成新规则）
        """
        if policy_a.utility_ema >= policy_b.utility_ema:
            main = policy_a
            absorbed = policy_b
        else:
            main = policy_b
            absorbed = policy_a

        main.feature_groups = list(set(main.feature_groups + absorbed.feature_groups))
        main.state_condition.update(absorbed.state_condition)

        total_utility = main.utility_ema * main.activation_count + absorbed.utility_ema * absorbed.activation_count
        total_activations = main.activation_count + absorbed.activation_count
        if total_activations > 0:
            main.utility_ema = total_utility / total_activations

        absorbed.status = 'ARCHIVE'

        return [main, absorbed]