# experiments/autotune/tuner_core.py
"""
SPLS AutoTuner - 核心类（SPLSAutoTuner）v6 强化学习版
★ 增加自动复活退休策略功能
★ ★ 2026-07-02 加载检查点时自动复活 DEPRECATED/ARCHIVE/DELETE 策略
★ ★ ★ 2026-08-XX 复活策略自动分配簇
★ ★ ★ ★ 2026-09-XX 若检查点无 PolicyGraph，则从当前策略重建并分配簇
"""

import os
import sys
import json
import time
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from experiments.autotune.utils import (
    ProgressLogger, MemoryCache, load_config, create_run_folder,
    compute_all_metrics, generate_comparison_report,
    load_window_data, compute_mase, extract_features
)
from experiments.autotune.spls_loop import SPLSLoop
from experiments.autotune.skill_policy import SkillPolicy
from experiments.autotune.collector import StateWindowGenerator
from experiments.autotune.inducer import SkillPolicyInductor
from experiments.autotune.validator import PolicyEvaluationOracle
from experiments.autotune.visualizer import ResultVisualizer
from experiments.autotune.online_evolver import PolicyFeedbackTrigger
from experiments.autotune.iterative_refiner import PolicyEvolutionEngine
from experiments.autotune.checkpoint_manager import CheckpointManager
from experiments.autotune.branch_loader import BranchLoader
from experiments.autotune.retirement_mechanism import RetirementMechanism
from experiments.autotune.policy_graph import PolicyGraph

from experiments.autotune.tuner_utils import (
    log_config_summary, print_file_manifest, save_round_policies,
    get_policies_hash
)
from experiments.autotune.tuner_patch import patch_trouble_windows
from experiments.autotune.tuner_train import run_training_loop
from experiments.autotune.tuner_eval import (
    run_ablation_parallel, print_ablation_summary
)

from experiments.autotune.policy_distribution import PolicyDistributionModel
from experiments.autotune.rl_components import BaselineTracker
from experiments.autotune.replay_memory import ReplayMemory

from src.agents.llm_client import LLMClient


class SPLSAutoTuner:
    def __init__(self, config_path: str = None, verbose: bool = False, resume_dir: str = None):
        self.config = load_config(config_path)
        self.verbose = verbose

        if resume_dir:
            if os.path.exists(resume_dir):
                self.run_dir = resume_dir
            else:
                test_path = os.path.join("llog", resume_dir)
                if os.path.exists(test_path):
                    self.run_dir = test_path
                else:
                    run_path = os.path.join("llog", f"run_{resume_dir}" if not resume_dir.startswith("run_") else resume_dir)
                    if os.path.exists(run_path):
                        self.run_dir = run_path
                    else:
                        raise FileNotFoundError(f"❌ 找不到指定的运行目录: {resume_dir}\n"
                                               f"   请确保目录存在于 llog/ 下，例如: run_20260624_043426")
            self.llog_dir = self.run_dir
            self.logger = ProgressLogger(log_dir=self.llog_dir, verbose=verbose, run_folder=False)
            self.logger.start_log("spls_autotune")
            self.logger.log(f"📂 从指定目录恢复训练: {self.run_dir}")
            self.config['llog_dir'] = self.llog_dir
            self.logger.log(f"   🔧 [修复] config.llog_dir 已同步为: {self.config['llog_dir']}")
        else:
            self.run_dir = create_run_folder("llog")
            self.llog_dir = self.run_dir
            os.makedirs(self.llog_dir, exist_ok=True)
            self.logger = ProgressLogger(log_dir=self.llog_dir, verbose=verbose, run_folder=False)
            self.logger.start_log("spls_autotune")
            self.config['llog_dir'] = self.llog_dir

        self.output_dir = self.config.get('output_dir', 'storage/autotune_results')
        os.makedirs(self.output_dir, exist_ok=True)

        train_cfg = self.config.get('training', {})
        self.num_rounds = train_cfg.get('rounds', 4)
        self.auto_optimize = train_cfg.get('auto_optimize', True)
        self.save_all_rounds = train_cfg.get('save_all_rounds', True)

        # RL 组件
        rl_cfg = self.config.get('rl', {})
        self.rl_enabled = rl_cfg.get('enabled', True)
        self.evolution_effect_strength = rl_cfg.get('evolution_effect_strength', 0.30)
        self.theta_decay = rl_cfg.get('theta_decay', 0.01)
        self.theta_bias_alpha = rl_cfg.get('theta_bias_alpha', 0.20)

        self.distribution_model = PolicyDistributionModel(
            learning_rate=rl_cfg.get('learning_rate', 0.01),
            temperature=rl_cfg.get('temperature', 1.0),
            theta_decay=self.theta_decay
        )
        self.baseline_tracker = BaselineTracker(
            initial=0.0,
            decay=rl_cfg.get('baseline_decay', 0.9)
        )
        self.replay_memory = ReplayMemory(
            max_size=rl_cfg.get('replay_memory', {}).get('max_size', 1000)
        )

        self.branch_loader = BranchLoader(self.config, self.logger)

        self.loop = SPLSLoop(self.config, self.logger, branch_loader=self.branch_loader)
        self.loop.rl_enabled = self.rl_enabled
        self.loop.distribution_model = self.distribution_model
        self.loop.baseline_tracker = self.baseline_tracker
        self.loop.replay_memory = self.replay_memory
        self.loop.theta_bias_alpha = self.theta_bias_alpha

        self.collector = StateWindowGenerator(self.config, self.logger, MemoryCache())

        inducer_config = self.config.copy()
        inducer_config['llog_dir'] = self.run_dir
        self.inducer = SkillPolicyInductor(inducer_config, self.logger)

        self.validator = PolicyEvaluationOracle(self.config, self.logger)
        self.visualizer = ResultVisualizer(self.config, self.logger)
        self.trigger = PolicyFeedbackTrigger(self.config, self.logger)

        # ★★★ 实例化退休机制（已禁用，但保留复活功能） ★★★
        self.retirement_mechanism = RetirementMechanism(self.config, self.logger)

        self.evolver = PolicyEvolutionEngine(self.config, self.logger, self.inducer, branch_loader=self.branch_loader)
        self.evolver.set_distribution_model(self.distribution_model)
        if 'rl' not in self.evolver.config:
            self.evolver.config['rl'] = {}
        self.evolver.config['rl']['evolution_effect_strength'] = self.evolution_effect_strength

        self.checkpoint_manager = CheckpointManager(self.run_dir, self.logger)

        self.collector.set_verbose(verbose)
        self.collector.set_trigger(self.trigger)

        self.round_results = {}
        self.policy_snapshots = {}
        self.b_subsets = []

        parallel_cfg = self.config.get('parallel', {})
        self.test_parallel = parallel_cfg.get('test_parallel', True)
        self.test_workers = parallel_cfg.get('test_workers', 10)

        self._last_b_eval_result = None
        self._last_policies_hash = None

        if self.rl_enabled:
            self.logger.log(f"   🧠 RL 模式已启用")
            self.logger.log(f"      learning_rate: {self.distribution_model.learning_rate}")
            self.logger.log(f"      temperature: {self.distribution_model.temperature}")
            self.logger.log(f"      baseline_decay: {self.baseline_tracker.decay}")
            self.logger.log(f"      replay_memory_size: {self.replay_memory.max_size}")
            self.logger.log(f"\n   📋 RL 稳定化参数 (三个手术刀级 Patch):")
            self.logger.log(f"      evolution_effect_strength = {self.evolution_effect_strength} (Patch 1)")
            self.logger.log(f"      theta_decay = {self.theta_decay} (Patch 2)")
            self.logger.log(f"      theta_bias_alpha = {self.theta_bias_alpha} (Patch 3)")

    def _get_policies_hash(self, policies):
        return get_policies_hash(policies)

    def _save_round_policies(self, round_num: int, policies: List, version: str):
        return save_round_policies(self.llog_dir, round_num, policies, version)

    def _log_config_summary(self):
        return log_config_summary(self.logger, self.config, self.num_rounds)

    def _print_file_manifest(self):
        return print_file_manifest(self.logger, self.llog_dir)

    def _patch_trouble_windows(self, policies: List[SkillPolicy], dataset_name: str, horizon: int):
        return patch_trouble_windows(self.logger, self.inducer, self.config, policies, dataset_name, horizon)

    def _evaluate_policy_on_df(self, policy: SkillPolicy, df: pd.DataFrame):
        from experiments.autotune.tuner_patch import evaluate_policy_on_df
        return evaluate_policy_on_df(policy, df)

    def _run_multi_round_ablation_parallel(self, dataset_name: str, window_size: int,
                                            horizon: int, test_df: pd.DataFrame) -> Dict:
        return run_ablation_parallel(
            self.logger, self.loop, self.policy_snapshots, self.num_rounds,
            dataset_name, window_size, horizon, test_df,
            self.test_parallel, self.test_workers
        )

    def _print_ablation_summary(self, ablation_results: Dict):
        return print_ablation_summary(self.logger, ablation_results)

    def _assign_policies_to_clusters(self, policies: List[SkillPolicy], policy_graph: PolicyGraph,
                                      current_round: int, policy_dict: Dict[str, SkillPolicy],
                                      global_avg_mase: float):
        """
        将策略分配到簇中，并打印醒目簇创建信息
        """
        if not policy_graph:
            return

        # 计算最差簇的平均 MASE
        worst_cluster_avg = float('inf')
        if policy_graph.clusters:
            worst_cluster_avg = max(c.avg_mase for c in policy_graph.clusters if c.is_active)

        context = {
            'current_round': current_round,
            'policy_dict': policy_dict,
            'global_avg_mase': global_avg_mase,
            'worst_cluster_avg_mase': worst_cluster_avg,
            'logger': self.logger
        }

        # 记录分配前的簇数
        before_clusters = len(policy_graph.clusters)

        for policy in policies:
            if policy.cluster_id is None:
                policy_graph.add_policy(policy, context=context)

        # 分配后统计
        after_clusters = len(policy_graph.clusters)
        new_clusters = after_clusters - before_clusters

        # ★★★ 醒目打印簇创建总结 ★★★
        self.logger.log("")
        self.logger.log("█" * 80)
        self.logger.log(f"█  📊 策略簇分配完成")
        self.logger.log(f"█  ────────────────────────────────────────────────────")
        self.logger.log(f"█  总策略数: {len(policies)}")
        self.logger.log(f"█  总簇数:   {after_clusters}")
        if new_clusters > 0:
            self.logger.log(f"█  🆕 本次新增簇: {new_clusters} 个")
            # 列出新增簇的信息（从 clusters 列表末尾取 new_clusters 个）
            for i, cluster in enumerate(policy_graph.clusters[-new_clusters:]):
                policy_names = []
                for pid in cluster.policies:
                    p = policy_dict.get(pid)
                    if p:
                        policy_names.append(p.name)
                self.logger.log(f"█     - 簇 {cluster.id}: 场景 '{cluster.scene_label}', "
                               f"策略 {len(cluster.policies)} 条: {', '.join(policy_names[:5])}{'...' if len(policy_names) > 5 else ''}")
        self.logger.log(f"█  未分配策略: {len(policy_graph.unassigned)} 条")
        self.logger.log("█" * 80)
        self.logger.log("")

    def run(self, dataset_name: str = None, min_train: int = None,
            horizon: int = None, compare: bool = False):
        start_time = time.time()

        self.logger.log("=" * 70)
        self.logger.log("🚀 SPLS v6 强化学习版（Policy Gradient + Online Learning）")
        self.logger.log(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.log(f"📁 运行目录: {self.run_dir}")
        self.logger.log(f"🔄 训练轮数: {self.num_rounds}")
        self.logger.log(f"📌 主进程 PID: {os.getpid()}")
        if self.rl_enabled:
            self.logger.log(f"🧠 RL 模式: 启用")
            self.logger.log(f"   Patch 1 (Evolution Soft Gate): strength={self.evolution_effect_strength}")
            self.logger.log(f"   Patch 2 (Adaptive Decay): base_decay={self.theta_decay}")
            self.logger.log(f"   Patch 3 (θ Bias Injection): alpha={self.theta_bias_alpha}")
        else:
            self.logger.log(f"🧠 RL 模式: 禁用（使用后备 Soft Mixture）")
        self.logger.log("=" * 70)

        self._log_config_summary()

        datasets = self.config.get('datasets', [])
        if dataset_name:
            datasets = [d for d in datasets if d.get('name') == dataset_name]
            if not datasets:
                self.logger.log(f"❌ 数据集 {dataset_name} 不在配置中")
                return

        if not datasets:
            self.logger.log("⚠️ 没有配置数据集")
            return

        all_results = {}

        for ds_config in datasets:
            name = ds_config.get('name')
            if dataset_name and name != dataset_name:
                continue

            self.logger.log(f"\n{'=' * 70}")
            self.logger.log(f"📊 处理数据集: {name}")
            self.logger.log(f"{'=' * 70}")

            window_sizes = ds_config.get('window_sizes', [600])
            step = ds_config.get('step_size', 150)
            horizon_val = horizon or ds_config.get('horizon', 12)
            max_train = ds_config.get('max_train_size')

            collected = self.collector.generate(
                dataset_name=name,
                window_sizes=window_sizes,
                horizon=horizon_val,
                step=step,
                max_train_size=max_train
            )

            if not collected:
                self.logger.log(f"⚠️ 采集失败: {name}")
                continue

            collection_file = self.collector.save_results(self.output_dir)
            collected_df = pd.DataFrame(collected)

            # 数据划分
            split_cfg = self.config.get('data_split', {})
            first_round_ratio = split_cfg.get('first_round_ratio', 0.50)
            B_ratio = split_cfg.get('B_ratio', 0.50)
            b_subset_count = self.config.get('evolution', {}).get('b_subset_count', 4)
            b_subset_indices = self.config.get('evolution', {}).get('b_subset_indices', [2, 3, 0])

            self.logger.log(f"📋 数据划分配置:")
            self.logger.log(f"   第一轮归纳比例 (A): {first_round_ratio:.2f}")
            self.logger.log(f"   B部分比例: {B_ratio:.2f}")
            self.logger.log(f"   B子集数: {b_subset_count}")
            self.logger.log(f"   演化使用B子集索引: {b_subset_indices} (0-based)")

            n_windows = len(collected_df)

            a_end = int(n_windows * first_round_ratio)
            df_a = collected_df.iloc[:a_end].copy()
            self.logger.log(f"📊 A部分 (归纳): {len(df_a)} 个窗口 (索引 0~{a_end-1})")

            b_start = a_end
            b_end = int(n_windows * (first_round_ratio + B_ratio))
            if b_end > n_windows:
                b_end = n_windows
            df_b = collected_df.iloc[b_start:b_end].copy()
            self.logger.log(f"📊 B部分 (演化): {len(df_b)} 个窗口 (索引 {b_start}~{b_end-1})")

            all_b_subsets = []
            b_subset_size = len(df_b) // b_subset_count if b_subset_count > 0 else len(df_b)
            for i in range(b_subset_count):
                start_idx = i * b_subset_size
                end_idx = (i + 1) * b_subset_size if i < b_subset_count - 1 else len(df_b)
                subset = df_b.iloc[start_idx:end_idx].copy()
                all_b_subsets.append(subset)
                self.logger.log(f"   B{i+1}: {len(subset)} 个窗口 (索引 {b_start + start_idx}~{b_start + end_idx - 1})")

            test_dfs = []
            for i, subset in enumerate(all_b_subsets):
                n_sub = len(subset)
                test_size = n_sub // 3
                if test_size > 0:
                    test_part = subset.iloc[:test_size].copy()
                    test_dfs.append(test_part)
                    all_b_subsets[i] = subset.iloc[test_size:].copy()
                    self.logger.log(f"   从B{i+1}抽取 {test_size} 个窗口作为测试，剩余 {len(all_b_subsets[i])} 个")
                else:
                    self.logger.log(f"   B{i+1} 窗口数不足3，无法抽取测试集，全部保留用于演化")

            if test_dfs:
                test_df = pd.concat(test_dfs).reset_index(drop=True)
                self.logger.log(f"📊 测试集: {len(test_df)} 个窗口 (由各B子集前1/3组成)")
            else:
                test_df = pd.DataFrame()
                self.logger.log("📊 测试集: 无")

            self.b_subsets = []
            for idx in b_subset_indices:
                if 0 <= idx < len(all_b_subsets):
                    self.b_subsets.append(all_b_subsets[idx])
                    self.logger.log(f"   演化使用 B{idx+1} (剩余 {len(all_b_subsets[idx])} 个窗口)")
                else:
                    self.logger.log(f"⚠️ 索引 {idx} 超出范围，跳过")

            collected_df['split'] = 'unknown'
            if len(df_a) > 0:
                collected_df.loc[df_a.index, 'split'] = 'A'
            if len(df_b) > 0:
                collected_df.loc[df_b.index, 'split'] = 'B'
            if len(test_df) > 0:
                collected_df.loc[test_df.index, 'split'] = 'test'
            collected_df.to_csv(collection_file, index=False)

            # ★★★ 断点续跑：加载检查点 ★★★
            checkpoint = self.checkpoint_manager.load()
            completed_rounds = checkpoint.get('completed_rounds', 0)
            current_b_subset_idx = checkpoint.get('current_b_subset_idx', b_subset_indices[0] if self.b_subsets else 0)

            file_completed = self.checkpoint_manager.detect_completed_rounds()
            if file_completed > completed_rounds:
                completed_rounds = file_completed
                self.logger.log(f"📂 从文件系统检测到已完成 {completed_rounds} 轮")

            policies = []
            policy_graph = None
            next_round = completed_rounds + 1

            if completed_rounds > 0:
                policies = checkpoint.get('current_policies', [])
                if policies:
                    self.logger.log(f"📋 从检查点加载策略: {len(policies)} 条")
                    # ★★★ 复活历史退休策略 ★★★
                    revived_count = self.retirement_mechanism.revive_retired_policies(
                        policies, current_round=next_round, initial_theta=-0.5
                    )
                    if revived_count > 0:
                        self.logger.log(f"✅ 成功复活 {revived_count} 条策略，重新加载到系统")
                        self.logger.log("   🔄 复活策略将立即参与演化（无冻结期）")

                    # 加载到 loop
                    self.loop.load_policies(policies)
                    self.collector.set_policies(policies)

                    # ★★★ 加载或重建 PolicyGraph ★★★
                    graph_data = checkpoint.get('policy_graph')
                    if graph_data:
                        policy_graph = PolicyGraph.from_dict(graph_data)
                        self.logger.log(f"📊 加载 PolicyGraph: {len(policy_graph.clusters)} 个簇")
                    else:
                        # 如果检查点中没有 PolicyGraph，则从当前策略重建
                        self.logger.log("⚠️ 检查点中没有 PolicyGraph，将根据当前策略重建")
                        policy_graph = PolicyGraph.from_policies(policies, self.config)
                        self.logger.log(f"📊 重建 PolicyGraph: {len(policy_graph.clusters)} 个簇")

                    # ★★★ 为所有策略分配簇（包括复活策略） ★★★
                    if policy_graph:
                        policy_dict = {p.policy_id: p for p in policies}
                        mases = [p.avg_mase for p in policies if p.avg_mase != float('inf')]
                        global_avg = np.mean(mases) if mases else 1.0

                        self._assign_policies_to_clusters(
                            policies, policy_graph, next_round, policy_dict, global_avg
                        )

                        # 保存更新后的 policy_graph
                        self.checkpoint_manager.save(
                            completed_rounds=completed_rounds,
                            current_policies=policies.copy(),
                            dataset=name,
                            horizon=horizon_val,
                            round_results=self.round_results,
                            current_b_subset_idx=current_b_subset_idx,
                            policy_graph=policy_graph.to_dict(),
                            a_eval_completed=checkpoint.get('a_eval_completed', False),
                            pending_round_state=checkpoint.get('pending_round_state')
                        )
                        self.logger.log("💾 检查点已更新（簇分配已持久化）")
                else:
                    latest_round = completed_rounds
                    policies = self.checkpoint_manager.get_round_policies(latest_round, "optimized")
                    if policies:
                        self.logger.log(f"📋 从第 {latest_round} 轮加载策略: {len(policies)} 条")
                        # 复活
                        revived_count = self.retirement_mechanism.revive_retired_policies(
                            policies, current_round=next_round, initial_theta=-0.5
                        )
                        if revived_count > 0:
                            self.logger.log(f"✅ 成功复活 {revived_count} 条策略")
                        self.loop.load_policies(policies)
                        self.collector.set_policies(policies)
                    else:
                        policies = []
                        completed_rounds = 0

            if not policies and completed_rounds == 0:
                self.logger.log("📋 无检查点，从头开始训练")

            # 将 policy_graph 存入 config 供后续使用
            self.policy_graph = policy_graph

            # 训练循环
            policies = run_training_loop(
                self.logger, self.loop, self.collector, self.inducer, self.evolver,
                self.checkpoint_manager, self.config, policies,
                df_a, df_b, all_b_subsets, b_subset_indices,
                name, horizon_val,
                completed_rounds, current_b_subset_idx,
                self.num_rounds, self.llog_dir,
                self.auto_optimize, self.save_all_rounds,
                self.round_results, self.policy_snapshots,
                self.b_subsets,
                self.policy_graph  # 传递 policy_graph
            )

            # 第1轮评估
            if self.round_results.get('round_1'):
                if not self.checkpoint_manager.is_a_eval_completed():
                    self.logger.log("🔍 首次评估 split='A'（共 48 个窗口）...")
                    eval_result = self.validator.evaluate(
                        self.round_results['round_1'].get('policies', []),
                        name, split='A'
                    )
                    self.round_results['round_1']['avg_mase'] = eval_result.get('avg_mase', float('inf'))
                    self.round_results['round_1']['improvement'] = eval_result.get('improvement_score', 0)

                    self.checkpoint_manager.save(
                        completed_rounds=self.checkpoint_manager.get_completed_rounds(),
                        current_policies=policies,
                        dataset=name,
                        horizon=horizon_val,
                        round_results=self.round_results,
                        current_b_subset_idx=checkpoint.get('current_b_subset_idx', 0),
                        policy_graph=self.policy_graph.to_dict() if self.policy_graph else None,
                        a_eval_completed=True
                    )
                    self.logger.log("💾 检查点已保存 (A部分评估完成)")
                else:
                    self.logger.log("⏭️ 跳过 split='A' 评估（已在之前完成）")

            # 事后补丁
            self._patch_trouble_windows(policies, name, horizon_val)

            self.logger.log("⏭️ 跳过最终测试集评估（将由 Ablation 覆盖）")

            # Ablation测试
            if len(test_df) > 0:
                self.logger.log(f"\n{'=' * 70}")
                self.logger.log(f"📊 多版本公平 Ablation (测试集, {len(test_df)} 个窗口)")
                self.logger.log(f"   对比模式: no_rule + 各轮次策略")
                self.logger.log(f"   并行模式: {'启用' if self.test_parallel else '禁用'} (workers={self.test_workers})")
                self.logger.log(f"{'=' * 70}")

                ablation_results = self._run_multi_round_ablation_parallel(
                    name, window_sizes[0], horizon_val, test_df
                )

                comparison_report = generate_comparison_report(ablation_results, self.llog_dir)
                self.logger.log(f"📁 对比报告已保存: {comparison_report}")
                self._print_ablation_summary(ablation_results)
            else:
                self.logger.log("⚠️ 无测试集，跳过 Ablation")

            all_results[name] = {
                'collected': len(collected),
                'rounds': self.num_rounds,
                'final_policies': len(policies),
                'completed_rounds': self.checkpoint_manager.get_completed_rounds()
            }

        stats_str = LLMClient.print_token_stats("SPLS Token 统计")
        self.logger.log("\n" + stats_str)

        stats_file = os.path.join(self.llog_dir, f"token_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write(stats_str)

        elapsed = time.time() - start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        self.logger.log(f"\n⏱️  总耗时: {minutes}分{seconds}秒")

        self._print_file_manifest()

        self.logger.log("\n" + "=" * 70)
        self.logger.log("✅ SPLS v6 强化学习版训练完成!")
        self.logger.log(f"📁 所有文件保存在: {self.llog_dir}")
        self.logger.log("=" * 70)

        return all_results