#!/usr/bin/env python
"""
语义匹配 vs RL 参数消融实验

对比以下模式：
1. no_rule：纯 LLM 预测（无策略参考）
2. semantic_top1：语义匹配度最高的 1 个策略 → 注入 LLM
3. semantic_top3_best_rl：语义前 3 个中 θ 最大的策略 → 注入 LLM
4. semantic_top5_best_rl：语义前 5 个中 θ 最大的策略 → 注入 LLM
5. semantic_top7_best_rl：语义前 7 个中 θ 最大的策略 → 注入 LLM

目的：证明 RL 参数（θ）能够从语义相似的候选策略中筛选出更优的策略，
     即 θ 具有学习到的判别能力。

用法：
    python -m experiments.autotune.test_semantic_vs_rl \
        --resume llog/cs2 \
        --round 34 \
        --workers 24 \
        --exec-mode llm

输出：
    llog/cs2/semantic_vs_rl_results/
        results.json
        comparison_report.txt
        comparison_plot.png
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
from src.agents.llm_planner import LLMPlannerAgent
from src.tasks.instance import TaskInstance
from run_benchmark import build_full_registry

WINDOW_TIMEOUT = 120


class SemanticVsRLTester:
    """语义匹配 vs RL 参数消融实验测试器"""

    def __init__(self, run_dir: str, round_num: int, config_path: str = None,
                 test_ratio: float = 0.5, exec_mode: str = 'llm'):
        self.run_dir = run_dir
        self.round_num = round_num
        self.config = load_config(config_path)
        self.output_dir = self.config.get('output_dir', 'storage/autotune_results')
        self.test_workers = 10
        self.test_ratio = test_ratio
        self.exec_mode = exec_mode  # 建议 llm，但保留 direct 用于调试
        self._timeout_counter = 0

        print("   🔧 构建技能注册表...")
        self.full_registry, _ = build_full_registry()
        print(f"   ✅ 注册了 {len(self.full_registry._skills)} 个技能")

        self.state_encoder = StateEncoder(self.config)
        self.model = self._detect_model()
        self.test_df = self._load_test_df()
        self.policies = self._load_round_policies(round_num)
        self._agent_cache = {}
        self._lock = threading.Lock()

        # 定义测试模式
        self.modes = [
            'no_rule',
            'semantic_top1',
            'semantic_top3_best_rl',
            'semantic_top5_best_rl',
            'semantic_top7_best_rl'
        ]

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

    def _load_round_policies(self, round_num: int) -> List[SkillPolicy]:
        path = os.path.join(self.run_dir, f"round_{round_num}", "refined_policies_optimized.json")
        if not os.path.exists(path):
            path = os.path.join(self.run_dir, f"round_{round_num}", "refined_policies_raw.json")
        if not os.path.exists(path):
            print(f"❌ 未找到第 {round_num} 轮策略文件")
            return []

        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            policies = [SkillPolicy.from_dict(p) for p in data.get('policies', [])]
            print(f"📋 加载第 {round_num} 轮策略: {len(policies)} 条")
            return policies
        except Exception as e:
            print(f"⚠️ 加载失败: {e}")
            return []

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

    def _predict_with_policy(self, task: TaskInstance, policy: SkillPolicy,
                             train: np.ndarray, horizon: int, period: int,
                             thread_id: int = 0) -> Optional[np.ndarray]:
        """使用单个策略作为参考（注入 LLM 或直接执行）"""
        if policy is None:
            return None

        if self.exec_mode == 'direct':
            pred = policy.execute(train, horizon, period)
            return np.array(pred) if pred is not None else None
        else:
            # llm 模式：注入策略参考
            agent = self._get_agent_for_thread(thread_id)
            agent.rule_engine = None
            agent._current_rule_strategy = policy.skill_strategy
            pred = agent.predict(task)
            return np.array(pred) if pred is not None else None

    def _select_policy_for_mode(self, mode: str, scored_policies: List[Tuple[SkillPolicy, float]]) -> Optional[SkillPolicy]:
        """
        根据模式从已排序的策略列表中选择策略
        scored_policies: 已按语义分数降序排列的列表 [(policy, score), ...]
        """
        if not scored_policies:
            return None

        if mode == 'semantic_top1':
            return scored_policies[0][0]

        elif mode == 'semantic_top3_best_rl':
            top_k = scored_policies[:3]
            # 选 θ 最大的
            return max(top_k, key=lambda x: x[0].logit_weight)[0]

        elif mode == 'semantic_top5_best_rl':
            top_k = scored_policies[:5]
            return max(top_k, key=lambda x: x[0].logit_weight)[0]

        elif mode == 'semantic_top7_best_rl':
            top_k = scored_policies[:7]
            return max(top_k, key=lambda x: x[0].logit_weight)[0]

        else:
            return None

    def evaluate_single_window(self, idx: int, row: pd.Series,
                               mode: str, thread_id: int = 0) -> Dict:
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
                # 计算所有策略的语义分数
                state = self.state_encoder.encode(train)
                numeric_state = state.get('numeric', {})
                scored = []
                for policy in self.policies:
                    try:
                        score = policy.compute_applicability_score(numeric_state)
                        scored.append((policy, score))
                    except Exception:
                        continue
                scored.sort(key=lambda x: x[1], reverse=True)

                selected_policy = self._select_policy_for_mode(mode, scored)
                if selected_policy is None:
                    pred = None
                else:
                    pred = self._predict_with_policy(task, selected_policy, train, horizon, period, thread_id)

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

    def evaluate_mode(self, mode: str) -> Dict:
        tasks = [(idx, row) for idx, row in self.test_df.iterrows()]
        total = len(tasks)

        pbar_desc = f"   {mode}"

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
                    idx, row, mode, thread_id
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
                'results': []
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
            'results': results
        }

    def run(self):
        print("\n" + "=" * 80)
        print("🧪 语义匹配 vs RL 参数消融实验")
        print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📁 运行目录: {self.run_dir}")
        print(f"🔢 指定轮次: round_{self.round_num}")
        print(f"📋 策略总数: {len(self.policies)}")
        print(f"⚡ 并行线程: {self.test_workers}")
        print(f"🔧 执行模式: {self.exec_mode} ({'直接执行策略' if self.exec_mode=='direct' else '注入LLM参考'})")
        print("=" * 80)

        if not self.policies:
            print("❌ 没有策略，退出")
            return

        all_results = {}
        total_start = time.time()

        for mode in self.modes:
            # 检查缓存
            cache = self._load_cached_results()
            if mode in cache and cache[mode].get('window_count', 0) > 0:
                print(f"\n   📦 使用缓存: {mode}")
                all_results[mode] = copy.deepcopy(cache[mode])
                continue

            start = time.time()
            result = self.evaluate_mode(mode)
            result['elapsed'] = time.time() - start
            all_results[mode] = result
            self._save_intermediate_results(all_results)

        total_elapsed = time.time() - total_start

        self._print_comparison_report(all_results, total_elapsed)
        self._generate_comparison_plots(all_results)
        self._save_final_results(all_results)

    def _load_cached_results(self) -> Dict:
        cache_path = os.path.join(self.run_dir, "semantic_vs_rl_results", "results.json")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return {}

    def _save_intermediate_results(self, all_results: Dict):
        output_dir = os.path.join(self.run_dir, "semantic_vs_rl_results")
        os.makedirs(output_dir, exist_ok=True)

        summary = {}
        for mode, r in all_results.items():
            if r.get('success', False):
                summary[mode] = {
                    'avg_mase': r.get('avg_mase', float('inf')),
                    'avg_mae': r.get('avg_mae', float('inf')),
                    'avg_rmse': r.get('avg_rmse', float('inf')),
                    'avg_smape': r.get('avg_smape', float('inf')),
                    'avg_owa': r.get('avg_owa', float('inf')),
                    'window_count': r.get('window_count', 0)
                }

        if summary:
            json_path = os.path.join(output_dir, "results.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)

    def _print_comparison_report(self, all_results: Dict, total_elapsed: float):
        print("\n" + "=" * 120)
        print("📊 语义匹配 vs RL 参数消融实验报告")
        print("=" * 120)

        no_rule_data = all_results.get('no_rule', {})
        no_rule_mase = no_rule_data.get('avg_mase', float('inf'))

        print(f"\n{'模式':<25} | {'MASE':<12} | {'MAE':<12} | {'RMSE':<12} | {'SMAPE':<12} | {'OWA':<12} | {'vs no_rule':<12}")
        print("-" * 150)

        for mode in self.modes:
            data = all_results.get(mode, {})
            if not data.get('success', False):
                continue
            mase = data.get('avg_mase', float('inf'))
            if mase == float('inf') or math.isnan(mase):
                continue

            mae = data.get('avg_mae', float('inf'))
            rmse = data.get('avg_rmse', float('inf'))
            smape = data.get('avg_smape', float('inf'))
            owa = data.get('avg_owa', float('inf'))

            imp = ""
            if mode != 'no_rule' and no_rule_mase > 0 and no_rule_mase != float('inf'):
                imp = f"{(no_rule_mase - mase) / no_rule_mase * 100:+.2f}%"

            print(f"{mode:<25} | {mase:<12.6f} | {mae:<12.6f} | {rmse:<12.6f} | {smape:<12.6f} | {owa:<12.6f} | {imp:<12}")

        print("-" * 150)
        print(f"\n📊 汇总统计:")
        print(f"   总耗时: {total_elapsed:.1f}s ({total_elapsed/60:.1f}分钟)")
        print(f"   no_rule MASE: {no_rule_mase:.6f}")
        print("   📌 模式说明:")
        print("      - semantic_top1: 语义最匹配的策略")
        print("      - semantic_topK_best_rl: 语义前K个中θ最大的策略")
        print("      - 若 RL 参数有效，则 semantic_topK_best_rl 应优于 semantic_top1")

    def _generate_comparison_plots(self, all_results: Dict):
        output_dir = os.path.join(self.run_dir, "semantic_vs_rl_results")
        os.makedirs(output_dir, exist_ok=True)

        modes = self.modes
        mases = []
        labels = []
        for mode in modes:
            data = all_results.get(mode, {})
            if data.get('success', False):
                mase = data.get('avg_mase', float('inf'))
                if mase != float('inf') and not math.isnan(mase):
                    mases.append(mase)
                    labels.append(mode)

        if not labels:
            print("⚠️ 无足够数据生成图像")
            return

        # 柱状图
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = ['#808080', '#2E86AB', '#F5A623', '#E68A2E', '#D4693A']
        bars = ax.bar(labels, mases, color=colors[:len(labels)], alpha=0.7, edgecolor='black', linewidth=1.5)

        for bar, mase in zip(bars, mases):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{mase:.4f}', ha='center', va='bottom', fontsize=9)

        ax.set_ylabel('MASE', fontsize=12)
        ax.set_title('语义匹配 vs RL 参数消融实验', fontsize=14)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        bar_path = os.path.join(output_dir, "comparison_bar.png")
        plt.savefig(bar_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   📊 柱状图已保存: {bar_path}")

        # 改善百分比图（相对于 no_rule）
        no_rule_mase = all_results.get('no_rule', {}).get('avg_mase', float('inf'))
        if no_rule_mase != float('inf') and not math.isnan(no_rule_mase):
            improvements = []
            imp_labels = []
            for mode in modes:
                if mode == 'no_rule':
                    continue
                data = all_results.get(mode, {})
                if data.get('success', False):
                    mase = data.get('avg_mase', float('inf'))
                    if mase != float('inf') and not math.isnan(mase):
                        imp = (no_rule_mase - mase) / no_rule_mase * 100
                        improvements.append(imp)
                        imp_labels.append(mode)

            if improvements:
                fig, ax = plt.subplots(figsize=(12, 6))
                bars = ax.bar(imp_labels, improvements, color='#2E86AB', alpha=0.7, edgecolor='black', linewidth=1.5)
                ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='基线 (0%)')
                for bar, imp in zip(bars, improvements):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                            f'{imp:.2f}%', ha='center', va='bottom', fontsize=9)

                ax.set_ylabel('MASE 改善百分比 (%)', fontsize=12)
                ax.set_title('相对 no_rule 的改善 (正值表示优于基线)', fontsize=14)
                ax.legend()
                ax.grid(axis='y', alpha=0.3)

                plt.tight_layout()
                imp_path = os.path.join(output_dir, "improvement_bar.png")
                plt.savefig(imp_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"   📊 改善图已保存: {imp_path}")

    def _save_final_results(self, all_results: Dict):
        output_dir = os.path.join(self.run_dir, "semantic_vs_rl_results")
        os.makedirs(output_dir, exist_ok=True)

        json_path = os.path.join(output_dir, "results.json")
        summary = {}
        for mode, r in all_results.items():
            if r.get('success', False):
                summary[mode] = {
                    'avg_mase': r.get('avg_mase', float('inf')),
                    'avg_mae': r.get('avg_mae', float('inf')),
                    'avg_rmse': r.get('avg_rmse', float('inf')),
                    'avg_smape': r.get('avg_smape', float('inf')),
                    'avg_owa': r.get('avg_owa', float('inf')),
                    'window_count': r.get('window_count', 0),
                    'elapsed': r.get('elapsed', 0)
                }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        report_path = os.path.join(output_dir, "comparison_report.txt")
        lines = []
        lines.append("=" * 120)
        lines.append("🧪 语义匹配 vs RL 参数消融实验报告")
        lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"运行目录: {self.run_dir}")
        lines.append(f"指定轮次: round_{self.round_num}")
        lines.append(f"执行模式: {self.exec_mode}")
        lines.append("=" * 120)
        lines.append("")
        lines.append("模式说明:")
        lines.append("  - no_rule: 纯 LLM 预测（无策略参考）")
        lines.append("  - semantic_top1: 语义匹配度最高的 1 个策略 → 注入 LLM")
        lines.append("  - semantic_top3_best_rl: 语义前 3 个中 θ 最大的策略 → 注入 LLM")
        lines.append("  - semantic_top5_best_rl: 语义前 5 个中 θ 最大的策略 → 注入 LLM")
        lines.append("  - semantic_top7_best_rl: 语义前 7 个中 θ 最大的策略 → 注入 LLM")
        lines.append("")
        lines.append("目的：验证 RL 参数（θ）是否能够从语义相似的候选策略中筛选出更优的策略。")
        lines.append("")
        lines.append(f"{'模式':<25} | {'MASE':<12} | {'MAE':<12} | {'RMSE':<12} | {'SMAPE':<12} | {'OWA':<12} | {'vs no_rule':<12}")
        lines.append("-" * 150)

        no_rule_data = all_results.get('no_rule', {})
        no_rule_mase = no_rule_data.get('avg_mase', float('inf'))

        for mode in self.modes:
            data = all_results.get(mode, {})
            if not data.get('success', False):
                continue
            mase = data.get('avg_mase', float('inf'))
            if mase == float('inf') or math.isnan(mase):
                continue
            mae = data.get('avg_mae', float('inf'))
            rmse = data.get('avg_rmse', float('inf'))
            smape = data.get('avg_smape', float('inf'))
            owa = data.get('avg_owa', float('inf'))

            imp = ""
            if mode != 'no_rule' and no_rule_mase > 0 and no_rule_mase != float('inf'):
                imp = f"{(no_rule_mase - mase) / no_rule_mase * 100:+.2f}%"

            lines.append(f"{mode:<25} | {mase:<12.6f} | {mae:<12.6f} | {rmse:<12.6f} | {smape:<12.6f} | {owa:<12.6f} | {imp:<12}")

        lines.append("-" * 150)

        # 统计 θ 分布
        if self.policies:
            theta_vals = [p.logit_weight for p in self.policies]
            lines.append(f"\n📊 θ 分布统计:")
            lines.append(f"   策略总数: {len(self.policies)}")
            lines.append(f"   θ min: {min(theta_vals):.4f}")
            lines.append(f"   θ max: {max(theta_vals):.4f}")
            lines.append(f"   θ mean: {np.mean(theta_vals):.4f}")
            lines.append(f"   θ median: {np.median(theta_vals):.4f}")

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        print(f"\n📁 结果已保存: {output_dir}")
        print(f"   - results.json (详细数据)")
        print(f"   - comparison_report.txt (文本报告)")
        print(f"   - comparison_bar.png (柱状图)")
        print(f"   - improvement_bar.png (改善图)")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="语义匹配 vs RL 参数消融实验")
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
                        help='执行模式：llm=注入LLM参考，direct=直接执行策略（建议 llm）')

    args = parser.parse_args()

    if os.path.exists(args.resume):
        run_dir = args.resume
    else:
        run_dir = os.path.join("llog", args.resume)

    if not os.path.exists(run_dir):
        print(f"❌ 目录不存在: {run_dir}")
        return

    tester = SemanticVsRLTester(
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