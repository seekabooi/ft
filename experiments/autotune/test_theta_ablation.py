#!/usr/bin/env python
"""
θ 分位数 Ablation 测试工具 - 簇内 Top‑θ 版
支持 direct 和 llm 两种执行模式，并可自动合并两种模式的结果进行对比。

用法：
# Direct 执行模式（直接执行策略，不调用 LLM）
python -m experiments.autotune.test_theta_ablation --resume llog/cs2 --round 34 --workers 10 --exec-mode direct

# LLM 参考模式（策略作为参考注入 LLM，由 LLM 生成预测）
python -m experiments.autotune.test_theta_ablation --resume llog/cs2 --round 34 --workers 10 --exec-mode llm

# 运行时如果存在其他模式的结果缓存，会自动加载并合并对比。
"""

import os
import sys
import json
import time
import copy
import math
import threading
import concurrent.futures
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from experiments.autotune.utils import load_config, load_window_data, compute_all_metrics
from experiments.autotune.skill_policy import SkillPolicy
from experiments.autotune.state_encoder import StateEncoder
from experiments.autotune.policy_graph import PolicyGraph
from src.agents.llm_planner import LLMPlannerAgent
from src.tasks.instance import TaskInstance
from run_benchmark import build_full_registry

WINDOW_TIMEOUT = 120


class ThetaAblationTester:
    """θ 分位数 Ablation 测试器 - 簇内 Top‑θ 版，支持 direct/llm 对比"""

    def __init__(self, run_dir: str, round_num: int, config_path: str = None,
                 test_ratio: float = 0.5, exec_mode: str = 'llm'):
        self.run_dir = run_dir
        self.round_num = round_num
        self.config = load_config(config_path)
        self.output_dir = self.config.get('output_dir', 'storage/autotune_results')
        self.test_workers = 10
        self.test_ratio = test_ratio
        self.exec_mode = exec_mode  # 'llm' 或 'direct'
        self._timeout_counter = 0

        print("   🔧 构建技能注册表...")
        self.full_registry, _ = build_full_registry()
        print(f"   ✅ 注册了 {len(self.full_registry._skills)} 个技能")

        self.state_encoder = StateEncoder(self.config)
        self.model = self._detect_model()
        self.test_df = self._load_test_df()
        self.policies, self.policy_graph = self._load_round_policies_and_graph(round_num)
        self._agent_cache = {}
        self._lock = threading.Lock()

        self.theta_percentiles = [0.10, 0.30, 0.50, 1.00]
        self.theta_labels = ['Top 10%', 'Top 30%', 'Top 50%', 'All (100%)']

        # 存储所有模式的结果，按 (mode, exec_mode) 区分
        self.all_results = {}

    def _load_test_df(self) -> pd.DataFrame:
        csv_path = os.path.join(self.output_dir, "collected_windows.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"❌ 未找到采集数据: {csv_path}")

        df = pd.read_csv(csv_path)

        if 'split' in df.columns:
            test_df = df[df['split'] == 'test'].copy()
            if len(test_df) > 0:
                print(f"📊 使用已有测试集标签: {len(test_df)} 个窗口")
                return test_df

        b_mask = df['split'].str.startswith('B') if 'split' in df.columns else pd.Series([True] * len(df))
        if 'split' not in df.columns:
            n = len(df)
            a_end = int(n * 0.5)
            b_mask = pd.Series([False] * n)
            b_mask.iloc[a_end:] = True
            df['split'] = ['A'] * a_end + ['B'] * (n - a_end)

        b_df = df[b_mask].copy().sort_values('window_id').reset_index(drop=True)
        n_b = len(b_df)
        test_size = int(n_b * self.test_ratio)
        test_df = b_df.iloc[:test_size].copy()
        test_df['split'] = 'test'

        print(f"📊 测试集: {len(test_df)} 个窗口 (从 {n_b} 个 B 窗口抽取 {self.test_ratio:.0%})")
        return test_df

    def _load_round_policies_and_graph(self, round_num: int) -> Tuple[List[SkillPolicy], Optional[PolicyGraph]]:
        path = os.path.join(self.run_dir, f"round_{round_num}", "refined_policies_optimized.json")
        if not os.path.exists(path):
            path = os.path.join(self.run_dir, f"round_{round_num}", "refined_policies_raw.json")
        if not os.path.exists(path):
            print(f"❌ 未找到第 {round_num} 轮策略文件")
            return [], None

        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            policies = [SkillPolicy.from_dict(p) for p in data.get('policies', [])]
            print(f"📋 加载第 {round_num} 轮策略: {len(policies)} 条")
        except Exception as e:
            print(f"⚠️ 加载失败: {e}")
            return [], None

        graph = None
        checkpoint_path = os.path.join(self.run_dir, "checkpoint.json")
        if os.path.exists(checkpoint_path):
            try:
                with open(checkpoint_path, 'r', encoding='utf-8') as f:
                    ckpt = json.load(f)
                graph_data = ckpt.get('policy_graph')
                if graph_data:
                    graph = PolicyGraph.from_dict(graph_data)
                    print(f"📊 从 checkpoint 加载 PolicyGraph: {len(graph.clusters)} 个簇")
            except Exception as e:
                print(f"⚠️ 加载 checkpoint 中的 PolicyGraph 失败: {e}")

        if graph is None:
            print("⚠️ 未找到 PolicyGraph，将按 feature_groups 分组构建临时簇")
            graph = PolicyGraph.from_policies(policies, self.config)
            print(f"📊 构建临时 PolicyGraph: {len(graph.clusters)} 个簇")

        return policies, graph

    def _detect_model(self) -> Optional[str]:
        from src.agents.llm_client import LLMClient
        models = ["glm-4", "glm-4.5-air"]
        for model in models:
            try:
                client = LLMClient(model=model, verbose=False)
                resp = client.call_with_retry("请回复'OK'", max_retries=1)
                if resp and resp.choices and resp.choices[0].message.content:
                    print(f"   ✅ 使用模型: {model}")
                    return model
            except:
                continue
        print("   ⚠️ 无可用模型，将使用均值回退")
        return None

    def _create_agent(self) -> LLMPlannerAgent:
        return LLMPlannerAgent(
            model=self.model if self.model else "glm-4",
            skill_registry=self.full_registry,
            use_skills=True,
            verbose=False,
            llm_call_interval=3
        )

    def _get_agent_for_thread(self, thread_id: int) -> LLMPlannerAgent:
        if thread_id not in self._agent_cache:
            self._agent_cache[thread_id] = self._create_agent()
        return self._agent_cache[thread_id]

    def _predict_no_rule(self, task: TaskInstance, thread_id: int = 0) -> Optional[np.ndarray]:
        if self.model is None:
            return None
        try:
            agent = self._get_agent_for_thread(thread_id)
            agent.rule_engine = None
            agent._current_rule_strategy = None
            pred = agent.predict(task)
            return np.array(pred) if pred else None
        except Exception:
            return None

    def _predict_with_policies(self, task: TaskInstance, policies: List[SkillPolicy],
                               train: np.ndarray, horizon: int, period: int,
                               thread_id: int = 0) -> Optional[np.ndarray]:
        if not policies:
            return None

        try:
            state = self.state_encoder.encode(train)
            numeric_state = state.get('numeric', {})

            scored = []
            for policy in policies:
                try:
                    score = policy.compute_applicability_score(numeric_state)
                    scored.append((policy, score))
                except Exception:
                    continue

            if not scored:
                return None

            best_policy, best_score = max(scored, key=lambda x: x[1])

            if self.exec_mode == 'direct':
                pred = best_policy.execute(train, horizon, period)
                return np.array(pred) if pred is not None else None
            else:
                # llm 模式：将策略作为参考注入 LLM
                agent = self._get_agent_for_thread(thread_id)
                agent.rule_engine = None
                agent._current_rule_strategy = best_policy.skill_strategy
                pred = agent.predict(task)
                return np.array(pred) if pred is not None else None

        except Exception:
            return None

    def evaluate_single_window(self, idx: int, row: pd.Series,
                               mode: str, policies: List[SkillPolicy],
                               thread_id: int = 0) -> Dict:
        window_id = row.get('window_id', 'unknown')
        window_data_path = row.get('window_data_path')

        if not window_data_path or not os.path.exists(window_data_path):
            return {'window_id': window_id, 'success': False, 'error': '路径不存在'}

        try:
            wdata = load_window_data(window_data_path)
            train = wdata['train']
            test = wdata['test']
            period = wdata.get('period', 365)
            mase_scale = wdata.get('mase_scale', 1.0)
            horizon = wdata.get('horizon', 7)

            task = TaskInstance(
                id=f"test_{window_id}",
                dataset_id="melbourne_temp",
                template_id="fixed_origin",
                question="",
                question_type="numerical",
                history=train.tolist(),
                horizon=horizon,
                frequency="daily",
                prediction_target={},
                resolution_date=datetime.now(),
                difficulty_level=1,
                ground_truth_extractor="",
                dates=None,
                target_date=""
            )

            if mode == 'no_rule':
                pred = self._predict_no_rule(task, thread_id)
            else:
                pred = self._predict_with_policies(task, policies, train, horizon, period, thread_id)

            if pred is None or len(pred) != len(test):
                pred = np.full(len(test), np.mean(train))

            metrics = compute_all_metrics(pred, test, mase_scale)

            return {
                'window_id': window_id,
                'success': True,
                'mase': metrics.get('mase', float('inf')),
                'mae': metrics.get('mae', float('inf')),
                'rmse': metrics.get('rmse', float('inf')),
                'smape': metrics.get('smape', float('inf')),
                'owa': metrics.get('owa', float('inf')),
            }

        except Exception as e:
            return {'window_id': window_id, 'success': False, 'error': str(e)}

    def evaluate_mode(self, mode: str, policies: List[SkillPolicy],
                      thread_id: int = 0) -> Dict:
        tasks = [(idx, row) for idx, row in self.test_df.iterrows()]
        total = len(tasks)

        pbar_desc = f"   {mode} ({len(policies)} policies, {self.exec_mode})"

        results = []
        mases = []
        maes = []
        rmses = []
        smapes = []
        owas = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.test_workers) as executor:
            futures = {}
            for idx, row in tasks:
                thread_id = idx % self.test_workers
                future = executor.submit(
                    self.evaluate_single_window,
                    idx, row, mode, policies, thread_id
                )
                futures[future] = idx

            pbar = tqdm(
                total=total,
                desc=pbar_desc,
                unit="窗口",
                ncols=100,
                position=0,
                leave=True,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}"
            )

            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result(timeout=WINDOW_TIMEOUT)
                    if result.get('success', False):
                        results.append(result)
                        mases.append(result.get('mase', float('inf')))
                        maes.append(result.get('mae', float('inf')))
                        rmses.append(result.get('rmse', float('inf')))
                        smapes.append(result.get('smape', float('inf')))
                        owas.append(result.get('owa', float('inf')))

                        valid_mases = [m for m in mases if m != float('inf') and not np.isnan(m)]
                        avg_mase = np.mean(valid_mases) if valid_mases else float('inf')
                        pbar.set_postfix({'MASE': f"{avg_mase:.4f}" if avg_mase != float('inf') else '...'})
                    else:
                        pbar.set_postfix({'状态': f"❌ {result.get('error', '未知')[:20]}"})
                except concurrent.futures.TimeoutError:
                    with self._lock:
                        self._timeout_counter += 1
                    pbar.set_postfix({'状态': f"⏱️ 超时 ({self._timeout_counter})"})
                except Exception as e:
                    pbar.set_postfix({'状态': f"⚠️ {str(e)[:20]}"})
                pbar.update(1)

            pbar.close()

        valid_count = len([m for m in mases if m != float('inf') and not np.isnan(m)])

        if valid_count == 0:
            return {
                'mode': mode,
                'success': False,
                'window_count': 0,
                'avg_mase': float('inf'),
                'avg_mae': float('inf'),
                'avg_rmse': float('inf'),
                'avg_smape': float('inf'),
                'avg_owa': float('inf'),
                'results': [],
                'policy_count': len(policies)
            }

        valid_mases = [m for m in mases if m != float('inf') and not np.isnan(m)]
        valid_maes = [m for m in maes if m != float('inf') and not np.isnan(m)]
        valid_rmses = [m for m in rmses if m != float('inf') and not np.isnan(m)]
        valid_smapes = [m for m in smapes if m != float('inf') and not np.isnan(m)]
        valid_owas = [m for m in owas if m != float('inf') and not np.isnan(m)]

        return {
            'mode': mode,
            'success': True,
            'mases': valid_mases,
            'maes': valid_maes,
            'rmses': valid_rmses,
            'smapes': valid_smapes,
            'owas': valid_owas,
            'window_count': valid_count,
            'avg_mase': np.mean(valid_mases) if valid_mases else float('inf'),
            'avg_mae': np.mean(valid_maes) if valid_maes else float('inf'),
            'avg_rmse': np.mean(valid_rmses) if valid_rmses else float('inf'),
            'avg_smape': np.mean(valid_smapes) if valid_smapes else float('inf'),
            'avg_owa': np.mean(valid_owas) if valid_owas else float('inf'),
            'results': results,
            'policy_count': len(policies)
        }

    def run(self):
        print("\n" + "=" * 80)
        print("🧪 θ 分位数 Ablation 测试 (簇内 Top‑θ)")
        print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📁 运行目录: {self.run_dir}")
        print(f"🔢 指定轮次: round_{self.round_num}")
        print(f"📋 策略总数: {len(self.policies)}")
        if self.policy_graph:
            print(f"📊 簇数量: {len(self.policy_graph.clusters)}")
        print(f"⚡ 并行线程: {self.test_workers}")
        print(f"🔧 执行模式: {self.exec_mode} ({'直接执行策略' if self.exec_mode=='direct' else '注入LLM参考'})")
        print("=" * 80)

        if not self.policies:
            print("❌ 没有策略，退出")
            return

        # 按簇分组
        if self.policy_graph and self.policy_graph.clusters:
            cluster_groups = []
            for cluster in self.policy_graph.clusters:
                if not cluster.is_active:
                    continue
                cluster_policies = [p for p in self.policies if p.policy_id in cluster.policies]
                if cluster_policies:
                    cluster_groups.append(cluster_policies)
            unassigned = [p for p in self.policies if not any(p.policy_id in c.policies for c in self.policy_graph.clusters)]
            if unassigned:
                cluster_groups.append(unassigned)
                print(f"📌 未分配策略 {len(unassigned)} 条，作为一个独立组")
        else:
            cluster_groups = [self.policies]
            print("⚠️ 无 PolicyGraph，将所有策略作为一组")

        print(f"\n📊 共 {len(cluster_groups)} 个策略组（簇）")

        # 构建各分位数子集
        subsets = {}
        for pct, label in zip(self.theta_percentiles, self.theta_labels):
            selected_policies = []
            for group in cluster_groups:
                sorted_group = sorted(group, key=lambda p: p.logit_weight, reverse=True)
                k = max(1, int(len(sorted_group) * pct))
                selected_policies.extend(sorted_group[:k])

            selected_policies = list({p.policy_id: p for p in selected_policies}.values())
            subsets[label] = selected_policies
            print(f"\n   {label}: {len(selected_policies)} 条策略 (来自 {len(cluster_groups)} 个簇)")
            if selected_policies:
                theta_vals = [p.logit_weight for p in selected_policies]
                print(f"      θ 范围: [{min(theta_vals):.4f}, {max(theta_vals):.4f}]")

        # 所有模式列表（包括 no_rule）
        modes = ['no_rule'] + self.theta_labels

        # 尝试加载已有其他模式的结果（用于对比）
        existing_results = self._load_cached_results()
        if existing_results:
            print("\n📦 发现已有其他执行模式的结果，将合并对比")
            # 过滤掉与当前模式相同的结果（避免重复），但保留其他模式
            for key, val in existing_results.items():
                if not key.endswith(f"_{self.exec_mode}"):
                    self.all_results[key] = val

        # 运行当前模式
        for mode in modes:
            # 构建当前模式的键
            key = f"{mode}_{self.exec_mode}"
            # 如果已存在结果（缓存中），跳过
            if key in self.all_results and self.all_results[key].get('window_count', 0) > 0:
                print(f"\n   📦 使用缓存: {mode} ({self.exec_mode})")
                continue

            if mode == 'no_rule':
                policies = []
            else:
                policies = subsets.get(mode, [])

            start = time.time()
            result = self.evaluate_mode(mode, policies)
            result['elapsed'] = time.time() - start
            result['exec_mode'] = self.exec_mode
            self.all_results[key] = result
            self._save_intermediate_results(self.all_results)

        total_elapsed = time.time() - self._start_time if hasattr(self, '_start_time') else 0
        self._start_time = getattr(self, '_start_time', time.time())

        # 生成合并对比报告
        self._print_comparison_report(self.all_results, total_elapsed)
        self._generate_comparison_plots(self.all_results)
        self._save_final_results(self.all_results)

    def _load_cached_results(self) -> Dict:
        """加载缓存结果，返回所有结果字典，键为 '{mode}_{exec_mode}'"""
        cache_path = os.path.join(self.run_dir, "theta_ablation_results", "results.json")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                print(f"📂 加载缓存结果: {len(data)} 项")
                return data
            except Exception as e:
                print(f"⚠️ 加载缓存失败: {e}")
        return {}

    def _save_intermediate_results(self, all_results: Dict):
        output_dir = os.path.join(self.run_dir, "theta_ablation_results")
        os.makedirs(output_dir, exist_ok=True)

        summary = {}
        for key, r in all_results.items():
            if r.get('success', False):
                # 提取模式名和执行模式
                parts = key.split('_')
                mode = '_'.join(parts[:-1]) if len(parts) > 1 else key
                exec_mode = parts[-1] if len(parts) > 1 else 'unknown'
                summary[key] = {
                    'mode': mode,
                    'exec_mode': exec_mode,
                    'avg_mase': r.get('avg_mase', float('inf')),
                    'avg_mae': r.get('avg_mae', float('inf')),
                    'avg_rmse': r.get('avg_rmse', float('inf')),
                    'avg_smape': r.get('avg_smape', float('inf')),
                    'avg_owa': r.get('avg_owa', float('inf')),
                    'window_count': r.get('window_count', 0),
                    'policy_count': r.get('policy_count', 0)
                }

        if summary:
            json_path = os.path.join(output_dir, "results.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)

    def _print_comparison_report(self, all_results: Dict, total_elapsed: float):
        print("\n" + "=" * 120)
        print("📊 θ 分位数 Ablation 对比报告 (簇内 Top‑θ)")
        print(f"   执行模式对比: {', '.join(set(r.get('exec_mode', 'unknown') for r in all_results.values() if r.get('success'))) if all_results else '无'}")
        print("=" * 120)

        # 收集 no_rule 基线（任一模式均可，优先使用 direct 或 llm）
        no_rule_key = None
        for key in all_results:
            if key.startswith('no_rule_'):
                no_rule_key = key
                break
        no_rule_data = all_results.get(no_rule_key, {})
        no_rule_mase = no_rule_data.get('avg_mase', float('inf'))
        no_rule_mode = no_rule_data.get('exec_mode', 'unknown')

        # 打印表头
        print(f"\n{'模式':<15} | {'执行模式':<10} | {'策略数':<8} | {'MASE':<12} | {'MAE':<12} | {'RMSE':<12} | {'SMAPE':<12} | {'OWA':<12} | {'vs no_rule':<12}")
        print("-" * 150)

        # 排序：先 no_rule，再按 theta 分位数升序
        sorted_keys = sorted(all_results.keys(), key=lambda x: (0 if x.startswith('no_rule') else 1, x))

        for key in sorted_keys:
            data = all_results.get(key, {})
            if not data.get('success', False):
                continue

            mase = data.get('avg_mase', float('inf'))
            if mase == float('inf') or math.isnan(mase):
                continue

            parts = key.split('_')
            mode = '_'.join(parts[:-1]) if len(parts) > 1 else key
            exec_mode = parts[-1] if len(parts) > 1 else 'unknown'

            policy_count = data.get('policy_count', 0)
            mae = data.get('avg_mae', float('inf'))
            rmse = data.get('avg_rmse', float('inf'))
            smape = data.get('avg_smape', float('inf'))
            owa = data.get('avg_owa', float('inf'))

            imp = ""
            if not key.startswith('no_rule_') and no_rule_mase > 0 and no_rule_mase != float('inf'):
                imp = f"{(no_rule_mase - mase) / no_rule_mase * 100:+.2f}%"

            print(f"{mode:<15} | {exec_mode:<10} | {policy_count:<8} | {mase:<12.6f} | {mae:<12.6f} | {rmse:<12.6f} | {smape:<12.6f} | {owa:<12.6f} | {imp:<12}")

        print("-" * 150)
        print(f"\n📊 汇总统计:")
        print(f"   总耗时: {total_elapsed:.1f}s ({total_elapsed/60:.1f}分钟)")
        print(f"   no_rule 基线 (模式: {no_rule_mode}) MASE = {no_rule_mase:.6f}")
        print("   📌 策略按簇分组，每个簇内取前 xx% θ")

    def _generate_comparison_plots(self, all_results: Dict):
        """生成对比折线图，区分执行模式"""
        output_dir = os.path.join(self.run_dir, "theta_ablation_results")
        os.makedirs(output_dir, exist_ok=True)

        # 按执行模式分组
        modes_by_exec = {}
        for key, data in all_results.items():
            if not data.get('success', False):
                continue
            parts = key.split('_')
            if len(parts) < 2:
                continue
            mode = '_'.join(parts[:-1])
            exec_mode = parts[-1]
            if exec_mode not in modes_by_exec:
                modes_by_exec[exec_mode] = {'modes': [], 'mases': [], 'policy_counts': []}
            mase = data.get('avg_mase', float('inf'))
            if mase != float('inf') and not math.isnan(mase):
                modes_by_exec[exec_mode]['modes'].append(mode)
                modes_by_exec[exec_mode]['mases'].append(mase)
                modes_by_exec[exec_mode]['policy_counts'].append(data.get('policy_count', 0))

        if not modes_by_exec:
            print("⚠️ 无足够数据生成图像")
            return

        # 绘制折线图
        fig, ax = plt.subplots(figsize=(12, 7))
        colors = {'direct': '#2E86AB', 'llm': '#F5A623'}

        for exec_mode, data in modes_by_exec.items():
            if not data['modes']:
                continue
            # 确保顺序为 Top 10%, Top 30%, Top 50%, All (100%)
            order_map = {'Top 10%': 1, 'Top 30%': 2, 'Top 50%': 3, 'All (100%)': 4}
            sorted_indices = sorted(range(len(data['modes'])), key=lambda i: order_map.get(data['modes'][i], 5))
            sorted_modes = [data['modes'][i] for i in sorted_indices]
            sorted_mases = [data['mases'][i] for i in sorted_indices]
            sorted_counts = [data['policy_counts'][i] for i in sorted_indices]

            ax.plot(sorted_modes, sorted_mases, marker='o', color=colors.get(exec_mode, '#888'),
                    linewidth=2, markersize=8, label=f'{exec_mode} 模式')

            # 标注数值
            for i, (mode, mase, count) in enumerate(zip(sorted_modes, sorted_mases, sorted_counts)):
                ax.annotate(f'{mase:.4f}', (i, mase), textcoords="offset points",
                           xytext=(0, 10), ha='center', fontsize=8)

        ax.set_xlabel('分位数子集', fontsize=12)
        ax.set_ylabel('MASE', fontsize=12)
        ax.set_title(f'簇内 θ 分位数 Ablation: direct vs llm 模式对比 (round_{self.round_num})', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 添加 no_rule 基线水平线（使用任一模式的 no_rule）
        no_rule_keys = [k for k in all_results if k.startswith('no_rule_')]
        for key in no_rule_keys:
            data = all_results.get(key, {})
            if data.get('success', False):
                mase = data.get('avg_mase', float('inf'))
                if mase != float('inf') and not math.isnan(mase):
                    exec_mode = key.split('_')[-1]
                    ax.axhline(y=mase, color=colors.get(exec_mode, '#888'), linestyle='--', alpha=0.5,
                               label=f'no_rule ({exec_mode}) = {mase:.4f}')

        plt.tight_layout()
        plot_path = os.path.join(output_dir, "theta_comparison.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   📊 对比折线图已保存: {plot_path}")

        # 生成改善百分比图（直接 vs llm）
        self._generate_improvement_plot(all_results, output_dir)

    def _generate_improvement_plot(self, all_results: Dict, output_dir: str):
        """生成改善百分比柱状图，对比 direct 和 llm"""
        # 提取 no_rule 基线（优先 direct）
        no_rule_mase = float('inf')
        no_rule_mode = 'direct'
        for key in all_results:
            if key.startswith('no_rule_'):
                data = all_results.get(key, {})
                if data.get('success', False):
                    mase = data.get('avg_mase', float('inf'))
                    if mase != float('inf') and not math.isnan(mase):
                        no_rule_mase = mase
                        no_rule_mode = key.split('_')[-1]
                        break

        if no_rule_mase == float('inf') or math.isnan(no_rule_mase):
            return

        # 收集各分位数子集的改善
        improvements = {'direct': [], 'llm': []}
        labels = []
        for key, data in all_results.items():
            if key.startswith('no_rule_'):
                continue
            if not data.get('success', False):
                continue
            parts = key.split('_')
            if len(parts) < 2:
                continue
            mode = '_'.join(parts[:-1])
            exec_mode = parts[-1]
            mase = data.get('avg_mase', float('inf'))
            if mase == float('inf') or math.isnan(mase):
                continue
            imp = (no_rule_mase - mase) / no_rule_mase * 100
            improvements[exec_mode].append((mode, imp))

        # 按分位数排序
        order_map = {'Top 10%': 1, 'Top 30%': 2, 'Top 50%': 3, 'All (100%)': 4}
        for exec_mode in improvements:
            improvements[exec_mode].sort(key=lambda x: order_map.get(x[0], 5))

        if not improvements['direct'] and not improvements['llm']:
            return

        fig, ax = plt.subplots(figsize=(12, 6))
        width = 0.35
        x = np.arange(len(improvements.get('direct', [])) if improvements.get('direct') else len(improvements.get('llm', [])))

        # 绘制 direct 和 llm 并排
        if improvements['direct']:
            direct_modes = [item[0] for item in improvements['direct']]
            direct_imps = [item[1] for item in improvements['direct']]
            ax.bar(x - width/2, direct_imps, width, label='direct', color='#2E86AB', alpha=0.7)
            for i, v in enumerate(direct_imps):
                ax.text(i - width/2, v + 0.5, f'{v:.1f}%', ha='center', fontsize=8)

        if improvements['llm']:
            llm_modes = [item[0] for item in improvements['llm']]
            llm_imps = [item[1] for item in improvements['llm']]
            ax.bar(x + width/2, llm_imps, width, label='llm', color='#F5A623', alpha=0.7)
            for i, v in enumerate(llm_imps):
                ax.text(i + width/2, v + 0.5, f'{v:.1f}%', ha='center', fontsize=8)

        ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='基线 (0%)')
        ax.set_xticks(x)
        ax.set_xticklabels(direct_modes if improvements['direct'] else llm_modes, rotation=15)
        ax.set_xlabel('分位数子集', fontsize=12)
        ax.set_ylabel('MASE 改善百分比 (%)', fontsize=12)
        ax.set_title(f'簇内 θ 分位数 vs no_rule 改善 (direct vs llm)', fontsize=14)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(output_dir, "improvement_chart.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   📊 改善对比图已保存: {plot_path}")

    def _save_final_results(self, all_results: Dict):
        output_dir = os.path.join(self.run_dir, "theta_ablation_results")
        os.makedirs(output_dir, exist_ok=True)

        # 保存 JSON
        json_path = os.path.join(output_dir, "results.json")
        summary = {}
        for key, r in all_results.items():
            if r.get('success', False):
                parts = key.split('_')
                mode = '_'.join(parts[:-1]) if len(parts) > 1 else key
                exec_mode = parts[-1] if len(parts) > 1 else 'unknown'
                summary[key] = {
                    'mode': mode,
                    'exec_mode': exec_mode,
                    'avg_mase': r.get('avg_mase', float('inf')),
                    'avg_mae': r.get('avg_mae', float('inf')),
                    'avg_rmse': r.get('avg_rmse', float('inf')),
                    'avg_smape': r.get('avg_smape', float('inf')),
                    'avg_owa': r.get('avg_owa', float('inf')),
                    'window_count': r.get('window_count', 0),
                    'policy_count': r.get('policy_count', 0),
                    'elapsed': r.get('elapsed', 0)
                }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        # 生成文本报告
        report_path = os.path.join(output_dir, "comparison_report.txt")
        lines = []
        lines.append("=" * 120)
        lines.append("🧪 θ 分位数 Ablation 测试报告 (簇内 Top‑θ)")
        lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"运行目录: {self.run_dir}")
        lines.append(f"指定轮次: round_{self.round_num}")
        lines.append("=" * 120)
        lines.append("")
        lines.append("说明:")
        lines.append("  - 策略按簇（PolicyGraph）分组，每个簇内按 θ 从大到小排序")
        lines.append("  - 分别取每个簇内 Top 10%、30%、50%、100% 的策略，合并去重")
        lines.append("  - 与 no_rule (无策略 LLM 直接预测) 对比")
        lines.append("  - 支持 direct 和 llm 两种执行模式，可并列对比")
        lines.append("")
        lines.append(f"{'模式':<15} | {'执行模式':<10} | {'策略数':<8} | {'MASE':<12} | {'MAE':<12} | {'RMSE':<12} | {'SMAPE':<12} | {'OWA':<12} | {'改善':<12}")
        lines.append("-" * 150)

        # 找出 no_rule 基线
        no_rule_key = None
        for key in all_results:
            if key.startswith('no_rule_'):
                no_rule_key = key
                break
        no_rule_data = all_results.get(no_rule_key, {})
        no_rule_mase = no_rule_data.get('avg_mase', float('inf'))

        # 排序输出
        sorted_keys = sorted(all_results.keys(), key=lambda x: (0 if x.startswith('no_rule') else 1, x))
        for key in sorted_keys:
            data = all_results.get(key, {})
            if not data.get('success', False):
                continue
            mase = data.get('avg_mase', float('inf'))
            if mase == float('inf') or math.isnan(mase):
                continue
            parts = key.split('_')
            mode = '_'.join(parts[:-1]) if len(parts) > 1 else key
            exec_mode = parts[-1] if len(parts) > 1 else 'unknown'
            policy_count = data.get('policy_count', 0)
            mae = data.get('avg_mae', float('inf'))
            rmse = data.get('avg_rmse', float('inf'))
            smape = data.get('avg_smape', float('inf'))
            owa = data.get('avg_owa', float('inf'))
            imp = ""
            if not key.startswith('no_rule_') and no_rule_mase > 0 and no_rule_mase != float('inf'):
                imp = f"{(no_rule_mase - mase) / no_rule_mase * 100:+.2f}%"
            lines.append(f"{mode:<15} | {exec_mode:<10} | {policy_count:<8} | {mase:<12.6f} | {mae:<12.6f} | {rmse:<12.6f} | {smape:<12.6f} | {owa:<12.6f} | {imp:<12}")

        lines.append("-" * 150)

        # 统计信息
        if self.policies:
            theta_vals = [p.logit_weight for p in self.policies]
            lines.append(f"\n📊 θ 分布统计:")
            lines.append(f"   策略总数: {len(self.policies)}")
            lines.append(f"   θ min: {min(theta_vals):.4f}")
            lines.append(f"   θ max: {max(theta_vals):.4f}")
            lines.append(f"   θ mean: {np.mean(theta_vals):.4f}")
            lines.append(f"   θ median: {np.median(theta_vals):.4f}")
            if self.policy_graph:
                lines.append(f"   簇数量: {len(self.policy_graph.clusters)}")

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        print(f"\n📁 结果已保存: {output_dir}")
        print(f"   - results.json (详细数据)")
        print(f"   - comparison_report.txt (文本报告)")
        print(f"   - theta_comparison.png (对比折线图)")
        print(f"   - improvement_chart.png (改善对比图)")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="θ 分位数 Ablation 测试工具 (簇内 Top‑θ，支持 direct/llm 对比)")
    parser.add_argument('--resume', type=str, required=True,
                        help='运行目录（如 llog/cs2）')
    parser.add_argument('--round', type=int, required=True,
                        help='指定轮次（如 34）')
    parser.add_argument('--config', type=str, default=None,
                        help='配置文件路径')
    parser.add_argument('--workers', type=int, default=10,
                        help='并行线程数（默认 10）')
    parser.add_argument('--test-ratio', type=float, default=0.5,
                        help='测试集比例（默认 0.5）')
    parser.add_argument('--exec-mode', type=str, choices=['llm', 'direct'], default='llm',
                        help='执行模式：llm=注入LLM参考，direct=直接执行策略')

    args = parser.parse_args()

    if os.path.exists(args.resume):
        run_dir = args.resume
    else:
        run_dir = os.path.join("llog", args.resume)

    if not os.path.exists(run_dir):
        print(f"❌ 目录不存在: {run_dir}")
        return

    tester = ThetaAblationTester(
        run_dir=run_dir,
        round_num=args.round,
        config_path=args.config,
        test_ratio=args.test_ratio,
        exec_mode=args.exec_mode
    )
    tester.test_workers = args.workers
    tester.run()


if __name__ == '__main__':
    main()