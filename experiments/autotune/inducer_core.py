# experiments/autotune/inducer_core.py
"""
Skill Policy Induction Core - 稳定核心逻辑
包含：聚类、多样性检查、生命周期、检查点等
★ ★ ★ 2026-06-24 禁止克隆，改为变体生成（variant_）
★ ★ ★ 2026-06-24 Re-Induction 新策略带 reind_ 前缀
★ ★ ★ 2026-06-24 延迟导入 PolicySpacePartitioner 避免循环依赖
★ ★ ★ 2026-06-25 增加 semantic_description 生成
★ ★ ★ 2026-06-25 返回 PolicyGraph 对象（结构升级）
★ ★ ★ 2026-06-26 语义描述升级：适用场景+相对优势+已知弱点，30-50字
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import time
import hashlib
import traceback
import copy

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from experiments.autotune.utils import (
    ProgressLogger, load_window_data, compute_mase, extract_features,
    format_weight_for_display, compute_strategy_similarity
)
from experiments.autotune.skill_policy import SkillPolicy
from experiments.autotune.policy_graph import PolicyGraph, PolicyCluster


class SkillPolicyInductorCore:
    """核心归纳逻辑（不常改）"""

    def __init__(self, config: Dict, logger: ProgressLogger):
        self.config = config
        self.logger = logger
        self.cache = None

        policy_cfg = config.get('policy_pool', {})
        self.min_policies = policy_cfg.get('min_policies', 5)
        self.max_policies = policy_cfg.get('hard_max', 15)
        self.target_policies = policy_cfg.get('target_policies', 8)
        self.embedding_dim = config.get('policy', {}).get('embedding_dim', 8)

        evo_cfg = config.get('evolution', {})
        self.lifetime_budget = evo_cfg.get('lifetime_budget', {}).get('max_new_rules', 30)
        self._cumulative_new_rules = 0

        self.feature_pool = [
            ['trend_strength', 'seasonal_strength'],
            ['seasonal_strength', 'adf_pvalue'],
            ['cv', 'volatility_ratio'],
            ['local_slope_120', 'local_std_ratio_120'],
            ['trend_strength', 'cv'],
            ['seasonal_strength', 'volatility_ratio'],
            ['adf_pvalue', 'local_slope_120'],
            ['trend_strength', 'local_slope_120'],
            ['cv', 'local_slope_120'],
            ['volatility_ratio', 'local_std_ratio_120'],
        ]

        self.similarity_threshold = 0.7

    def induce(self, collected_data: pd.DataFrame, force_regenerate: bool = False) -> Dict:
        """归纳策略（主入口）

        Args:
            collected_data: 窗口数据
            force_regenerate: 是否强制重新生成（不读缓存），Re-Induction 时传入 True

        Returns:
            {
                'policies': List[SkillPolicy],
                'policy_graph': PolicyGraph   # ★ 新增
            }
        """
        try:
            from experiments.autotune.cluster import PolicySpacePartitioner
        except ImportError as e:
            self.logger.log(f"   ❌ PolicySpacePartitioner 导入失败: {e}")
            return {'policies': [], 'policy_graph': None}

        self.logger.log("\n" + "=" * 70)
        self.logger.log("🧠 Skill Policy Induction v5 (策略生命周期管理版)")
        self.logger.log("=" * 70)

        if collected_data.empty:
            return {'policies': [], 'policy_graph': None}

        self.logger.log(f"📊 数据量: {len(collected_data)} 个窗口")

        window_best = self._process_windows(collected_data, force_regenerate=force_regenerate)

        if not window_best:
            self.logger.log("⚠️ 没有有效的策略，返回空")
            return {'policies': [], 'policy_graph': None}

        self.logger.log("\n" + "=" * 70)
        self.logger.log("🔬 Policy Space 分割（动态聚类）")
        self.logger.log("=" * 70)

        n_strategies = len(window_best)
        n_clusters = min(4, max(3, n_strategies // 3))
        n_clusters = max(3, min(n_clusters, 6))

        partitioner = PolicySpacePartitioner(self.logger)
        partitions = partitioner.partition(window_best, n_clusters=n_clusters)
        partitions = self._ensure_min_partition_size(partitions, min_size=2)

        policies = self._generate_policies_with_distinct_conditions(partitions, is_reinduction=force_regenerate)

        policies = self._ensure_diversity(policies)
        policies = self._ensure_policy_count(policies)

        for policy in policies:
            if len(policy.feature_groups) < 2:
                self._supplement_feature_groups(policy)

        # ★ 构建 PolicyGraph
        policy_graph = PolicyGraph.from_policies(policies, self.config)

        # ★ 将策略的 cluster_id 回填
        for cluster in policy_graph.clusters:
            for pid in cluster.policies:
                for policy in policies:
                    if policy.policy_id == pid:
                        policy.cluster_id = cluster.id
                        break

        self.logger.log(f"\n✅ 归纳完成: {len(policies)} 条策略, {len(policy_graph.clusters)} 个簇")

        # ★ 打印簇信息
        self.logger.log("\n📋 簇分布:")
        for c in policy_graph.clusters:
            self.logger.log(f"   {c.id}: {c.scene_label} ({len(c.policies)} 条策略, avg_mase={c.avg_mase:.4f})")

        self.logger.log("\n" + "=" * 70)
        self.logger.log("📋 策略详情:")
        self.logger.log("=" * 70)

        for i, policy in enumerate(policies):
            self._print_policy_detail(i + 1, policy)

        if not force_regenerate:
            progress_file = os.path.join(self.config.get('llog_dir', 'llog'), 'induction_progress.json')
            if os.path.exists(progress_file):
                os.remove(progress_file)
                self.logger.log("   🧹 已清理进度缓存")

        return {
            'policies': [p.to_dict() for p in policies],
            'policy_graph': policy_graph.to_dict() if policy_graph else None
        }

    # ==================== ★ 核心修改：语义描述生成 ====================
    def _generate_semantic_description(self, feature_groups: List[str], skill_strategy: Dict,
                                       avg_mase: float = 1.0, error_std: float = 0.0,
                                       all_mases: Optional[List[float]] = None) -> str:
        """
        生成凝练的语义描述（30-50字）
        包含：适用场景 + 相对优势 + 已知弱点
        """
        parts = []

        # 1. 适用场景（从 feature_groups 推断）
        scene_parts = []
        if 'seasonal_strength' in feature_groups:
            scene_parts.append("季节敏感")
        if 'trend_strength' in feature_groups:
            scene_parts.append("趋势敏感")
        if 'cv' in feature_groups or 'volatility_ratio' in feature_groups:
            scene_parts.append("波动敏感")
        if 'local_slope_120' in feature_groups:
            scene_parts.append("局部结构敏感")
        if 'adf_pvalue' in feature_groups:
            scene_parts.append("平稳性敏感")

        if scene_parts:
            parts.append(f"适用: {','.join(scene_parts)}")

        # 2. 相对优势（基于 avg_mase 和全局比较）
        if all_mases and len(all_mases) > 1:
            min_mase = min(all_mases)
            max_mase = max(all_mases)
            if avg_mase <= min_mase * 1.05:
                parts.append("全池最优")
            elif avg_mase <= min_mase * 1.15:
                parts.append("表现优异")
            elif avg_mase >= max_mase * 0.95:
                parts.append("表现较差")
        parts.append(f"MASE={avg_mase:.3f}")

        # 3. 已知弱点（基于 error_std）
        if error_std > 0.3:
            parts.append("弱点:稳定性一般")
        elif error_std > 0.2:
            parts.append("弱点:局部波动敏感")

        # 4. 策略特点（从 stages 提取）
        stages = skill_strategy.get('stages', [])
        if stages:
            skill_names = set()
            for st in stages:
                skill_names.update(st.get('weights', {}).keys())
            if skill_names:
                main_skills = list(skill_names)[:3]
                parts.append(f"用{','.join(main_skills)}")

        desc = "；".join(parts)

        if len(desc) > 60:
            desc = desc[:55] + "..."

        if not desc or len(desc) < 5:
            desc = f"通用策略，MASE={avg_mase:.3f}"

        return desc

    # ==================== 以下方法保持不变 ====================
    def _ensure_diversity(self, policies: List[SkillPolicy]) -> List[SkillPolicy]:
        if len(policies) < 2:
            return policies

        diverse_policies = []
        used_feature_combos = set()
        used_state_conditions = []

        for policy in policies:
            if not policy.state_condition:
                self.logger.log(f"   ⚠️ 策略 {policy.name} 的 state_condition 为空，设置默认条件")
                policy.state_condition = {"trend_strength": "> 0.0", "seasonal_strength": "> 0.0"}
                if not policy.feature_groups:
                    policy.feature_groups = ['trend_strength', 'seasonal_strength']

            combo_key = tuple(sorted(policy.feature_groups))
            if combo_key in used_feature_combos:
                new_groups = self._find_alternative_feature_groups(used_feature_combos)
                if new_groups:
                    policy.feature_groups = new_groups
                    policy.state_condition = {}
                    for g in new_groups:
                        policy.state_condition[g] = "> 0.0"
                    combo_key = tuple(sorted(new_groups))

            used_feature_combos.add(combo_key)

            cond_str = self._condition_to_str(policy.state_condition)
            if cond_str in used_state_conditions:
                first_key = list(policy.state_condition.keys())[0]
                policy.state_condition[first_key] = "> 0.001"
                cond_str = self._condition_to_str(policy.state_condition)

            used_state_conditions.append(cond_str)

            policy_weights = self._extract_weights(policy)
            is_similar = False
            for existing in diverse_policies:
                existing_weights = self._extract_weights(existing)
                similarity = compute_strategy_similarity(policy_weights, existing_weights)
                if similarity > self.similarity_threshold:
                    is_similar = True
                    for stage in policy.skill_strategy.get('stages', []):
                        weights = stage.get('weights', {})
                        for k in weights:
                            weights[k] = max(0.01, min(0.99, weights[k] + np.random.uniform(-0.1, 0.1)))
                    total = sum(weights.values())
                    if total > 0:
                        for k in weights:
                            weights[k] = round(weights[k] / total, 6)
                    break

            diverse_policies.append(policy)

        return diverse_policies

    def _condition_to_str(self, condition: dict) -> str:
        if not condition:
            return "通用"
        items = sorted(condition.items())
        return " AND ".join([f"{k} {v}" for k, v in items])

    def _extract_weights(self, policy: SkillPolicy) -> dict:
        weights = {}
        for stage in policy.skill_strategy.get('stages', []):
            for skill, w in stage.get('weights', {}).items():
                weights[skill] = weights.get(skill, 0) + w
        return weights

    def _find_alternative_feature_groups(self, used_combos: set) -> List[str]:
        for combo in self.feature_pool:
            combo_key = tuple(sorted(combo))
            if combo_key not in used_combos:
                return combo
        return self.feature_pool[0]

    def _print_candidate_strategies(self, candidates: List[Dict], window_id: int):
        if not candidates:
            self.logger.log(f"   📋 窗口 {window_id}: 无候选策略")
            return

        self.logger.log(f"   📋 窗口 {window_id} 候选策略 ({len(candidates)} 个):")
        for i, strategy in enumerate(candidates):
            try:
                name = strategy.get('name', f'候选{i + 1}')
                stages = strategy.get('stages', [])
                if not stages:
                    self.logger.log(f"      策略 {i + 1}: {name} → (无阶段)")
                    continue
                stage_desc = []
                for j, stage in enumerate(stages):
                    steps = stage.get('steps', 0)
                    weights = stage.get('weights', {})
                    w_str = ', '.join([f"{k}:{format_weight_for_display(v, precision=6)}" for k, v in weights.items()])
                    stage_desc.append(f"{steps}步{{{w_str}}}")
                self.logger.log(f"      策略 {i + 1}: {name} → {' → '.join(stage_desc)}")
            except Exception as e:
                self.logger.log(f"      策略 {i + 1}: 打印异常 - {type(e).__name__}: {e}")
                self.logger.log(traceback.format_exc())

    def _print_policy_detail(self, index: int, policy: SkillPolicy):
        self.logger.log(f"\n📌 策略 {index}: {policy.name}")
        self.logger.log(f"   ID: {policy.policy_id}")
        self.logger.log(f"   状态: {policy.status}")
        self.logger.log(f"   Feature Groups: {policy.feature_groups}")
        self.logger.log(f"   📝 语义描述: {policy.semantic_description or '无'}")
        self.logger.log(f"   📍 簇: {policy.cluster_id or '未分配'}")

        stages = policy.skill_strategy.get('stages', [])
        if stages:
            self.logger.log(f"   📊 策略组合:")
            for j, stage in enumerate(stages):
                steps = stage.get('steps', 0)
                weights = stage.get('weights', {})
                w_str = ', '.join([f"{k}:{format_weight_for_display(v, precision=6)}" for k, v in weights.items()])
                self.logger.log(f"      阶段{j + 1}: {steps}步 → {{{w_str}}}")
        else:
            self.logger.log(f"   ⚠️ 无策略组合")

        if policy.state_condition:
            cond_str = ' AND '.join([f"{k} {v}" for k, v in policy.state_condition.items()])
            self.logger.log(f"   🎯 条件: {cond_str}")
        else:
            self.logger.log(f"   🎯 条件: 通用（⚠️ 应避免）")

        self.logger.log(f"   📈 性能: avg_mase={policy.avg_mase:.6f}, utility={policy.utility_ema:.6f}")

    def _process_windows(self, data: pd.DataFrame, force_regenerate: bool = False) -> List[Dict]:
        total_windows = len(data)
        progress_file = os.path.join(self.config.get('llog_dir', 'llog'), 'induction_progress.json')
        window_best = []
        processed_window_ids = set()

        if force_regenerate:
            self.logger.log("   🔄 强制重新生成模式（忽略缓存，基于当前窗口数据生成新策略）")
            if os.path.exists(progress_file):
                os.remove(progress_file)
                self.logger.log("   🧹 已清理进度缓存")
            window_best = []
            processed_window_ids = set()
        else:
            if os.path.exists(progress_file):
                try:
                    with open(progress_file, 'r', encoding='utf-8') as f:
                        progress = json.load(f)
                    window_best = progress.get('window_best', [])
                    processed_window_ids = set(progress.get('processed_ids', []))
                    self.logger.log(f"📂 加载进度缓存: 已处理 {len(processed_window_ids)} 个窗口")
                    if window_best:
                        self.logger.log(f"   已有 {len(window_best)} 个有效策略")

                    if window_best:
                        trouble_count = 0
                        mases = []
                        for item in window_best:
                            m = item.get('_mase', float('inf'))
                            if m != float('inf'):
                                mases.append(m)
                            if m > self.trouble_mase_threshold:
                                trouble_count += 1
                        if mases:
                            self.logger.log(f"   📊 历史窗口摘要:")
                            self.logger.log(f"      平均MASE: {np.mean(mases):.4f}")
                            self.logger.log(f"      最小MASE: {np.min(mases):.4f}")
                            self.logger.log(f"      最大MASE: {np.max(mases):.4f}")
                            self.logger.log(f"      困难窗口数: {trouble_count} (MASE > {self.trouble_mase_threshold})")

                    for item in window_best:
                        mase = item.get('_mase', float('inf'))
                        if mase > self.trouble_mase_threshold:
                            wid = item.get('_window_id')
                            origin = item.get('_origin', 0)
                            window_size = item.get('_window_size', 600)
                            window_data_path = None
                            if wid is not None:
                                row = data[data['window_id'] == wid]
                                if not row.empty:
                                    window_data_path = row.iloc[0].get('window_data_path')
                            if window_data_path is None:
                                continue
                            strategy_name = item.get('name', 'unknown')
                            self._collect_trouble_window(
                                wid, mase, window_data_path, origin, window_size, {'name': strategy_name}
                            )
                except Exception as e:
                    self.logger.log(f"   ⚠️ 加载进度缓存失败: {e}，从头开始")
                    window_best = []
                    processed_window_ids = set()

            if not force_regenerate:
                recovered_best, recovered_ids = self._load_window_results_from_files(data)

                newly_recovered = []
                for item in recovered_best:
                    wid = item.get('_window_id')
                    if wid is not None and wid not in processed_window_ids:
                        window_best.append(item)
                        processed_window_ids.add(wid)
                        newly_recovered.append(wid)

                if newly_recovered:
                    self.logger.log(f"   ✅ 从独立结果文件恢复 {len(newly_recovered)} 个窗口: {sorted(newly_recovered)}")
                    self.logger.log(f"   📊 总进度: 已处理 {len(processed_window_ids)}/{total_windows} 个窗口")
                    self._save_progress(progress_file, window_best, processed_window_ids)

        self.logger.log(f"   📊 全局困难池: {len(self.trouble_pool)} 个窗口")

        self.logger.log("\n" + "=" * 70)
        self.logger.log("📋 逐窗口策略生成与评估 (窗口级并行版)")
        self.logger.log("=" * 70)

        pending_windows = []
        for idx, row in data.iterrows():
            window_id = row.get('window_id', idx + 1)
            if window_id in processed_window_ids:
                continue
            window_data_path = row.get('window_data_path', '')
            if not window_data_path or not os.path.exists(window_data_path):
                self.logger.log(f"⚠️ 窗口 {window_id} 数据路径不存在，跳过")
                processed_window_ids.add(window_id)
                continue
            pending_windows.append({
                'window_id': window_id,
                'origin': row.get('origin', 0),
                'window_size': row.get('window_size', 600),
                'window_data_path': window_data_path,
                'features': self._extract_features(row),
                'trajectory': self._get_trajectory(row),
                'skill_filter': self.config.get('skill_filter', {}),
                'llm_config': self.config.get('llm', {})
            })

        if not pending_windows:
            self.logger.log("✅ 所有窗口已处理完成")
            return window_best

        pending_ids = sorted([w['window_id'] for w in pending_windows])
        self.logger.log(f"📊 待处理窗口: {pending_ids[0]}-{pending_ids[-1]} (共 {len(pending_windows)} 个)")
        self.logger.log(f"📊 待处理窗口数: {len(pending_windows)}")

        if self.window_parallel and len(pending_windows) > 1:
            results, failed_windows = self._process_windows_parallel(
                pending_windows, data, total_windows
            )
        else:
            results = self._process_windows_serial(pending_windows, data, total_windows)
            failed_windows = []

        for result in results:
            if result.get('error') is not None:
                self.logger.log(f"   ⚠️ 窗口 {result.get('window_id')} 处理失败: {result.get('error')}")
                if result.get('window_id') not in [w.get('window_id') for w in failed_windows]:
                    failed_windows.append(result)
                continue

            wid = result.get('window_id')
            best = result.get('best_strategy')
            best_mase = result.get('best_mase', float('inf'))
            origin = result.get('origin', 0)
            window_size = result.get('window_size', 600)
            window_data_path = result.get('window_data_path', '')
            features = result.get('features', {})
            is_trouble = result.get('is_trouble', False)

            if result.get('logs'):
                for log_msg in result['logs']:
                    pid = os.getpid()
                    self.logger.log(f"[PID:{pid}] {log_msg}")

            if best is None or best_mase == float('inf'):
                self.logger.log(f"   ⚠️ 窗口 {wid} 无有效策略，跳过")
                processed_window_ids.add(wid)
                self._save_progress(progress_file, window_best, processed_window_ids)
                continue

            if is_trouble:
                self._collect_trouble_window(
                    wid, best_mase, window_data_path,
                    origin, window_size, best
                )

            best['_window_id'] = wid
            best['_origin'] = origin
            best['_mase'] = best_mase
            best['_features'] = features
            window_best.append(best)
            processed_window_ids.add(wid)

            self._processed_count += 1

            self._save_progress(progress_file, window_best, processed_window_ids)

        if failed_windows:
            self.logger.log(f"\n   🔄 检测到 {len(failed_windows)} 个失败窗口，准备重试...")
            for retry_attempt in range(self.window_max_retries):
                if not failed_windows:
                    break
                self.logger.log(f"   🔄 重试第 {retry_attempt + 1}/{self.window_max_retries} 次...")
                time.sleep(self.window_retry_delay)

                remaining_windows = []
                for fw in failed_windows:
                    orig = next((w for w in pending_windows if w.get('window_id') == fw.get('window_id')), None)
                    if orig:
                        remaining_windows.append(orig)

                if not remaining_windows:
                    break

                retry_results = self._process_windows_serial(remaining_windows, data, total_windows)

                still_failed = []
                for result in retry_results:
                    if result.get('error') is not None:
                        self.logger.log(f"      ⚠️ 重试窗口 {result.get('window_id')} 仍失败: {result.get('error')}")
                        still_failed.append(result)
                        continue

                    wid = result.get('window_id')
                    best = result.get('best_strategy')
                    best_mase = result.get('best_mase', float('inf'))
                    origin = result.get('origin', 0)
                    window_size = result.get('window_size', 600)
                    window_data_path = result.get('window_data_path', '')
                    features = result.get('features', {})
                    is_trouble = result.get('is_trouble', False)

                    if result.get('logs'):
                        for log_msg in result['logs']:
                            pid = os.getpid()
                            self.logger.log(f"[PID:{pid}] {log_msg}")

                    if best is None or best_mase == float('inf'):
                        self.logger.log(f"      ⚠️ 窗口 {wid} 仍无有效策略，跳过")
                        processed_window_ids.add(wid)
                        self._save_progress(progress_file, window_best, processed_window_ids)
                        continue

                    self.logger.log(f"      ✅ 窗口 {wid} 重试成功! MASE={best_mase:.6f}")
                    if is_trouble:
                        self._collect_trouble_window(
                            wid, best_mase, window_data_path,
                            origin, window_size, best
                        )
                    best['_window_id'] = wid
                    best['_origin'] = origin
                    best['_mase'] = best_mase
                    best['_features'] = features
                    window_best.append(best)
                    processed_window_ids.add(wid)
                    self._processed_count += 1
                    self._save_progress(progress_file, window_best, processed_window_ids)

                failed_windows = still_failed

            if failed_windows:
                self.logger.log(f"   ⚠️ {len(failed_windows)} 个窗口在 {self.window_max_retries} 次重试后仍失败，跳过")
                for fw in failed_windows:
                    processed_window_ids.add(fw.get('window_id'))
                    self._save_progress(progress_file, window_best, processed_window_ids)

        self.logger.log(f"\n📊 第1轮归纳完成统计:")
        self.logger.log(f"   已处理窗口: {len(processed_window_ids)}/{total_windows}")
        self.logger.log(f"   有效策略数: {len(window_best)}")
        self.logger.log(f"   困难窗口数: {len(self.trouble_pool)} (MASE > {self.trouble_mase_threshold})")

        self.logger.log(f"\n📊 共收集 {len(window_best)} 个窗口的best策略")
        return window_best

    def _load_window_results_from_files(self, data: pd.DataFrame):
        recovered_best = []
        recovered_ids = set()

        if not hasattr(self, 'window_results_dir'):
            return recovered_best, recovered_ids

        window_results_dir = os.path.join(self.config.get('llog_dir', 'llog'), 'window_results')
        if not os.path.exists(window_results_dir):
            return recovered_best, recovered_ids

        try:
            for filename in os.listdir(window_results_dir):
                if not filename.startswith('window_') or not filename.endswith('.json'):
                    continue

                filepath = os.path.join(window_results_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        result_data = json.load(f)

                    window_id = result_data.get('window_id')
                    if window_id is None:
                        continue

                    best_strategy = result_data.get('best_strategy')
                    best_mase = result_data.get('best_mase', float('inf'))
                    origin = result_data.get('origin', 0)
                    window_size = result_data.get('window_size', 600)
                    features = result_data.get('features', {})
                    is_trouble = result_data.get('is_trouble', False)

                    if best_strategy is None or best_mase == float('inf'):
                        continue

                    best_strategy['_window_id'] = window_id
                    best_strategy['_origin'] = origin
                    best_strategy['_mase'] = best_mase
                    best_strategy['_features'] = features

                    recovered_best.append(best_strategy)
                    recovered_ids.add(window_id)

                    if is_trouble:
                        row = data[data['window_id'] == window_id]
                        if not row.empty:
                            window_data_path = row.iloc[0].get('window_data_path', '')
                            if window_data_path:
                                self.logger.log(f"   📌 从独立文件恢复困难窗口 {window_id} (MASE={best_mase:.4f})")
                                self._collect_trouble_window(
                                    window_id, best_mase, window_data_path,
                                    origin, window_size, best_strategy
                                )

                except Exception as e:
                    self.logger.log(f"   ⚠️ 读取窗口结果文件失败 {filename}: {e}")
                    continue

        except Exception as e:
            self.logger.log(f"   ⚠️ 扫描窗口结果目录失败: {e}")

        return recovered_best, recovered_ids

    def _save_progress(self, progress_file: str, window_best: List, processed_ids: set):
        try:
            def convert_to_serializable(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_serializable(item) for item in obj]
                else:
                    return obj

            serializable_best = convert_to_serializable(window_best)
            progress = {
                'window_best': serializable_best,
                'processed_ids': list(processed_ids),
                'timestamp': time.time()
            }
            with open(progress_file, 'w', encoding='utf-8') as f:
                json.dump(progress, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.log(f"   ⚠️ 保存进度失败: {e}")

    def _generate_policies_with_distinct_conditions(self, partitions: List[Dict], is_reinduction: bool = False) -> List[SkillPolicy]:
        policies = []
        if not partitions:
            return policies

        partition_features = []
        for p in partitions:
            strategies = p.get('strategies', [])
            if not strategies:
                continue

            features_list = [s.get('_features', {}) for s in strategies if s.get('_features')]
            if not features_list:
                continue

            avg_features = {}
            for key in ['trend_strength', 'seasonal_strength', 'adf_pvalue', 'period', 'cv']:
                values = [f.get(key, 0) for f in features_list if key in f]
                avg_features[key] = np.mean(values) if values else 0

            all_mases = [s.get('_mase', 1.0) for s in strategies if s.get('_mase') is not None]

            partition_features.append({
                'avg': avg_features,
                'strategies': strategies,
                'avg_mase': p.get('avg_mase', 0),
                'size': len(strategies),
                'all_mases': all_mases
            })

        partition_features.sort(key=lambda x: x['avg_mase'])

        used_combinations = set()
        for i, pf in enumerate(partition_features):
            available = []
            for combo in self.feature_pool:
                combo_key = tuple(sorted(combo))
                if combo_key not in used_combinations:
                    available.append(combo)

            if available:
                selected = available[0]
            else:
                selected = self.feature_pool[i % len(self.feature_pool)]

            used_combinations.add(tuple(sorted(selected)))

            condition_parts = []
            for feat in selected:
                val = pf['avg'].get(feat, 0)
                if len(partition_features) == 1:
                    continue
                elif i == 0:
                    condition_parts.append(f"{feat} <= {val:.3f}")
                elif i == len(partition_features) - 1:
                    prev_val = partition_features[i - 1]['avg'].get(feat, 0)
                    condition_parts.append(f"{feat} > {prev_val:.3f}")
                else:
                    prev_val = partition_features[i - 1]['avg'].get(feat, 0)
                    condition_parts.append(f"{feat} > {prev_val:.3f} and {feat} <= {val:.3f}")

            if not condition_parts:
                condition_parts = ["trend_strength > 0.0", "seasonal_strength > 0.0"]

            condition = ' and '.join(condition_parts)

            policy = self._create_policy_from_strategy(
                pf['strategies'][0] if pf['strategies'] else {},
                pf['avg_mase'],
                condition,
                selected,
                f"policy_{i}",
                is_reinduction=is_reinduction,
                all_mases=pf.get('all_mases', [])
            )
            policies.append(policy)

        if policies:
            best_policy = min(policies, key=lambda p: p.avg_mase)
            fallback = best_policy.to_dict()
            fallback_policy = SkillPolicy.from_dict(fallback)
            fallback_policy.policy_id = hashlib.md5(f"fallback_{time.time()}".encode()).hexdigest()[:8]
            if is_reinduction:
                fallback_policy.name = f"reind_fallback"
            else:
                fallback_policy.name = "fallback_policy"
            fallback_policy.state_condition = {}
            fallback_policy.feature_groups = []
            fallback_policy.utility_ema = 0.3
            fallback_policy.reward_ema = 0.3
            fallback_policy.semantic_description = "通用兜底策略，适用于所有场景"
            policies.append(fallback_policy)

        return policies

    def _create_policy_from_strategy(self, strategy: Dict, avg_mase: float,
                                     condition: str, feature_groups: List[str],
                                     name_prefix: str, is_reinduction: bool = False,
                                     all_mases: List[float] = None) -> SkillPolicy:
        policy_id = hashlib.md5(f"{condition}_{time.time()}".encode()).hexdigest()[:8]
        state_condition = self._condition_to_dict(condition)
        embedding = list(np.random.randn(self.embedding_dim) * 0.1)
        utility = 1.0 / (avg_mase + 0.01)

        if is_reinduction:
            name = f"reind_{policy_id[:4]}"
        else:
            name = f"{name_prefix}_{policy_id[:4]}"

        error_std = 0.0
        if 'error_std' in strategy:
            error_std = strategy.get('error_std', 0.0)

        semantic_description = self._generate_semantic_description(
            feature_groups, strategy, avg_mase, error_std, all_mases
        )

        return SkillPolicy(
            policy_id=policy_id,
            name=name,
            embedding=embedding,
            state_condition=state_condition,
            feature_groups=feature_groups,
            skill_strategy=strategy,
            avg_mase=avg_mase,
            error_mean=avg_mase,
            confidence=0.5,
            reward_ema=utility,
            utility_ema=utility,
            created_at=time.strftime('%Y-%m-%d %H:%M:%S'),
            semantic_description=semantic_description,
            cluster_id=None
        )

    def _condition_to_dict(self, condition: str) -> Dict:
        if condition == 'True':
            return {}

        result = {}
        parts = condition.split(' and ')
        for part in parts:
            part = part.strip()
            matched = False

            for op in ['>=', '<=', '==', '!=', '>', '<']:
                if op in part:
                    k, v = part.split(op, 1)
                    k = k.strip()
                    try:
                        v = float(v.strip())
                    except:
                        pass
                    result[k] = f"{op} {v}"
                    matched = True
                    break

            if not matched:
                if '==' in part:
                    k, v = part.split('==', 1)
                else:
                    k, v = part, part
                try:
                    result[k.strip()] = float(v.strip())
                except:
                    result[k.strip()] = v.strip()

        return result

    def _ensure_policy_count(self, policies: List[SkillPolicy]) -> List[SkillPolicy]:
        if len(policies) >= self.min_policies and len(policies) <= self.max_policies:
            return policies

        if len(policies) < self.min_policies:
            additional = []
            needed = self.min_policies - len(policies)
            self.logger.log(f"   🔄 策略数 {len(policies)} < {self.min_policies}，生成 {needed} 条变体策略（非克隆）")

            for i in range(needed):
                base = policies[i % len(policies)]
                new_policy = copy.deepcopy(base)
                new_policy.policy_id = hashlib.md5(f"variant_{time.time()}_{i}".encode()).hexdigest()[:8]
                new_policy.name = f"variant_{base.name[:8]}_{i}"

                if new_policy.embedding:
                    noise = np.random.randn(len(new_policy.embedding)) * 0.3
                    new_policy.embedding = list(np.array(new_policy.embedding) + noise)
                    norm = np.linalg.norm(new_policy.embedding)
                    if norm > 0:
                        new_policy.embedding = (np.array(new_policy.embedding) / norm).tolist()

                for stage in new_policy.skill_strategy.get('stages', []):
                    weights = stage.get('weights', {})
                    if weights:
                        for k in weights:
                            weights[k] = max(0.01, min(1.0, weights[k] + np.random.uniform(-0.2, 0.2)))
                        total = sum(weights.values())
                        if total > 0:
                            for k in weights:
                                weights[k] = round(weights[k] / total, 6)

                if new_policy.state_condition:
                    for key in list(new_policy.state_condition.keys()):
                        try:
                            val = float(new_policy.state_condition[key].split()[1])
                            new_val = max(0.0, min(1.0, val + np.random.uniform(-0.15, 0.15)))
                            new_policy.state_condition[key] = f"> {new_val:.3f}"
                        except:
                            pass

                new_policy.status = 'TRIAL'
                new_policy.utility_ema = base.utility_ema * 0.9
                if base.semantic_description:
                    new_policy.semantic_description = base.semantic_description + "（变体）"
                else:
                    new_policy.semantic_description = f"变体策略，MASE={base.avg_mase:.3f}"
                additional.append(new_policy)
                self.logger.log(f"      ✅ 生成变体 {i + 1}/{needed}: {new_policy.name}")

            policies.extend(additional)
            self.logger.log(f"   ✅ 策略数从 {len(policies) - needed} → {len(policies)}（其中 {needed} 条为变体）")

        if len(policies) > self.max_policies:
            policies.sort(key=lambda p: p.utility_ema, reverse=True)
            policies = policies[:self.max_policies]
            self.logger.log(f"   ✅ 截断到 {len(policies)} 条策略")

        return policies

    def _ensure_min_partition_size(self, partitions: List[Dict], min_size: int = 2) -> List[Dict]:
        result = []
        small_partitions = []

        for p in partitions:
            if p.get('size', 0) >= min_size:
                result.append(p)
            else:
                small_partitions.append(p)

        if small_partitions and result:
            for small in small_partitions:
                closest = min(result, key=lambda x: abs(x['avg_mase'] - small['avg_mase']))
                closest['strategies'].extend(small['strategies'])
                closest['size'] = len(closest['strategies'])
                if closest['strategies']:
                    closest['avg_mase'] = np.mean([s.get('_mase', 0) for s in closest['strategies']])

        if not result:
            all_strategies = []
            for p in partitions:
                all_strategies.extend(p.get('strategies', []))
            if all_strategies:
                result = [{
                    'strategies': all_strategies,
                    'avg_mase': np.mean([s.get('_mase', 0) for s in all_strategies]),
                    'size': len(all_strategies)
                }]

        return result

    def _extract_features(self, row: pd.Series) -> Dict:
        features = {}
        for col in row.index:
            if col not in ['dataset', 'window_id', 'origin', 'train_size', 'test_size', 'period', 'mase_scale',
                           'best_config_name', 'best_mase', 'best_trajectory', 'window_data_path', 'horizon',
                           'window_size']:
                try:
                    features[col] = float(row[col])
                except:
                    pass
        return features

    def _get_trajectory(self, row: pd.Series) -> List:
        try:
            traj_str = row.get('best_trajectory', '[]')
            return json.loads(traj_str) if traj_str else []
        except:
            return []

    def _supplement_feature_groups(self, policy: SkillPolicy):
        all_features = ['trend_strength', 'seasonal_strength', 'adf_pvalue', 'cv', 'local_slope_120',
                        'volatility_ratio']
        current = set(policy.feature_groups)
        needed = 2 - len(current)

        for feat in all_features:
            if feat not in current and needed > 0:
                policy.feature_groups.append(feat)
                if feat not in policy.state_condition:
                    policy.state_condition[feat] = "> 0.0"
                needed -= 1

    def _collect_trouble_window(self, *args, **kwargs):
        pass

    def _process_windows_parallel(self, pending_windows, data, total_windows):
        pass

    def _process_windows_serial(self, pending_windows, data, total_windows):
        pass

    def _load_trouble_pool(self):
        return []