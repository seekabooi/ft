#!/usr/bin/env python
"""
Ablation 对比测试：验证 RL 训练中"策略状态管理"的贡献
★ 三种模式对比：
   - active_only: 仅从 ACTIVE 策略中选择（模拟使用 RL 训练参数）
   - active_trial: 从 ACTIVE + TRIAL 策略中选择（验证候补策略的影响）
   - all: 从所有策略中选择（模拟不使用 RL 训练参数）
★ 10 线程并行执行
★ 五指标表格对比（MASE、MAE、RMSE、SMAPE、OWA）+ 改善百分比
★ 图像对比（柱状图、箱线图、改善折线图）
★ 断点续存
★ ★★ 核心修正：round_N 模式可选择注入策略到 LLM 或直接执行策略（--exec-mode）

python -m experiments.autotune.test_ablation_compare --resume run_20260626_133418 --workers 10 --test-ratio 0.5 --exec-mode direct

llog/run_xxx/ablation_compare_results/（新测试文件）或 quick_test_results/（旧测试文件）
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


class AblationCompareTester:
    """Ablation 对比测试器 - 三模式对比"""

    def __init__(self, run_dir: str, config_path: str = None, test_ratio: float = 0.5,
                 exec_mode: str = 'llm'):
        self.run_dir = run_dir
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
        self.round_policies = {}
        self._load_all_round_policies()

        self.results = {}
        self._lock = threading.Lock()

        # ★ 缓存的 agent（每个线程独立创建，但复用模型和注册表）
        self._agent_cache = {}

    def _load_test_df(self) -> pd.DataFrame:
        """加载测试集（方案A：从 B 子集抽取 test_ratio）"""
        csv_path = os.path.join(self.output_dir, "collected_windows.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"❌ 未找到采集数据: {csv_path}")

        df = pd.read_csv(csv_path)

        if 'split' in df.columns:
            test_df = df[df['split'] == 'test'].copy()
            if len(test_df) > 0:
                print(f"📊 使用已有测试集标签: {len(test_df)} 个窗口")
                return test_df

        # 按比例从 B 子集抽取
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
            return []
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return [SkillPolicy.from_dict(p) for p in data.get('policies', [])]
        except Exception as e:
            print(f"   ⚠️ 加载第 {round_num} 轮失败: {e}")
            return []

    def _load_all_round_policies(self):
        print("\n📋 扫描策略文件...")
        print("-" * 50)

        round_num = 1
        max_rounds = 24
        while round_num <= max_rounds:
            round_dir = os.path.join(self.run_dir, f"round_{round_num}")
            policy_path = os.path.join(round_dir, "refined_policies_optimized.json")
            if os.path.exists(policy_path):
                policies = self._load_round_policies(round_num)
                if policies:
                    self.round_policies[f'round_{round_num}'] = policies
                    print(f"   ✅ 第 {round_num:2d} 轮: {len(policies):3d} 条策略")
                round_num += 1
            else:
                if round_num > 2:
                    next_path = os.path.join(self.run_dir, f"round_{round_num + 1}", "refined_policies_optimized.json")
                    if not os.path.exists(os.path.dirname(next_path)):
                        break
                round_num += 1

        if self.round_policies:
            print(f"\n📊 共加载 {len(self.round_policies)} 轮策略: {list(self.round_policies.keys())}")

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
        """创建 LLM Agent（每个线程独立）"""
        return LLMPlannerAgent(
            model=self.model if self.model else "glm-4",
            skill_registry=self.full_registry,
            use_skills=True,
            verbose=False,
            llm_call_interval=3
        )

    def _get_agent_for_thread(self, thread_id: int) -> LLMPlannerAgent:
        """获取线程专属 agent（缓存复用）"""
        if thread_id not in self._agent_cache:
            self._agent_cache[thread_id] = self._create_agent()
        return self._agent_cache[thread_id]

    def _predict_no_rule(self, task: TaskInstance, thread_id: int = 0) -> Optional[np.ndarray]:
        """无策略预测（LLM 直接预测）"""
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
                               filter_mode: str = 'active_only',
                               thread_id: int = 0) -> Optional[np.ndarray]:
        """
        ★★★ 根据 exec_mode 选择策略执行方式 ★★★

        filter_mode 说明：
            - 'active_only'   : 仅从 ACTIVE 策略中选择
            - 'active_trial'  : 从 ACTIVE + TRIAL 策略中选择
            - 'all'           : 从所有策略中选择（ACTIVE + TRIAL + DEPRECATED）

        exec_mode 说明：
            - 'llm'    : 匹配策略后，将其 skill_strategy 注入 LLM，由 LLM 生成预测
            - 'direct' : 直接执行策略（policy.execute()），不经过 LLM
        """
        if not policies:
            return None

        try:
            state = self.state_encoder.encode(train)
            numeric_state = state.get('numeric', {})

            scored = []
            for policy in policies:
                # ★ 根据 filter_mode 过滤策略状态
                if filter_mode == 'active_only':
                    if policy.status not in ['ACTIVE']:
                        continue
                elif filter_mode == 'active_trial':
                    if policy.status not in ['ACTIVE', 'TRIAL']:
                        continue
                # filter_mode == 'all' 时不跳过任何策略

                try:
                    score = policy.compute_applicability_score(numeric_state)
                    scored.append((policy, score))
                except Exception:
                    continue

            if not scored:
                return None

            # 1. 选出最优策略
            best_policy, best_score = max(scored, key=lambda x: x[1])

            if self.exec_mode == 'direct':
                # ★★★ 直接执行策略，不经过 LLM ★★★
                pred = best_policy.execute(train, horizon, period)
                return np.array(pred) if pred is not None else None
            else:
                # ★★★ 原有 LLM 注入模式 ★★★
                agent = self._get_agent_for_thread(thread_id)
                agent.rule_engine = None
                agent._current_rule_strategy = best_policy.skill_strategy
                pred = agent.predict(task)
                return np.array(pred) if pred is not None else None

        except Exception as e:
            return None

    def evaluate_single_window(self, idx: int, row: pd.Series,
                               mode: str, policies: List[SkillPolicy],
                               filter_mode: str, thread_id: int = 0) -> Dict:
        """评估单个窗口的某个模式"""
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
                pred = self._predict_with_policies(task, policies, train, horizon, period,
                                                    filter_mode, thread_id)

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
                      filter_mode: str, thread_id: int = 0) -> Dict:
        """评估一个模式的所有窗口"""
        tasks = [(idx, row) for idx, row in self.test_df.iterrows()]
        total = len(tasks)

        policy_count = len(policies) if policies else 0

        # ★ 过滤模式描述
        filter_desc_map = {
            'active_only': 'ACTIVE only',
            'active_trial': 'ACTIVE+TRIAL',
            'all': 'ALL'
        }
        filter_desc = filter_desc_map.get(filter_mode, filter_mode)

        pbar_desc = f"   {mode:>12} [{filter_desc}]"

        results = []
        mases = []
        maes = []
        rmses = []
        smapes = []
        owas = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.test_workers) as executor:
            futures = {}
            for idx, row in tasks:
                # ★ 每个任务分配一个线程 ID
                thread_id = idx % self.test_workers
                future = executor.submit(
                    self.evaluate_single_window,
                    idx, row, mode, policies, filter_mode, thread_id
                )
                futures[future] = idx

            pbar = tqdm(
                total=total,
                desc=pbar_desc,
                unit="窗口",
                ncols=120,
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
                'filter_mode': filter_mode,
                'success': False,
                'mases': [],
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
            'filter_mode': filter_mode,
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

    def _save_intermediate_results(self, all_results: Dict):
        output_dir = os.path.join(self.run_dir, "ablation_compare_results")
        os.makedirs(output_dir, exist_ok=True)

        summary = {}
        for key, r in all_results.items():
            avg_mase = r.get('avg_mase', float('inf'))
            if avg_mase == float('inf') or math.isnan(avg_mase):
                continue
            summary[key] = {
                'mode': r.get('mode'),
                'filter_mode': r.get('filter_mode'),
                'avg_mase': avg_mase,
                'avg_mae': r.get('avg_mae', float('inf')),
                'avg_rmse': r.get('avg_rmse', float('inf')),
                'avg_smape': r.get('avg_smape', float('inf')),
                'avg_owa': r.get('avg_owa', float('inf')),
                'window_count': r.get('window_count', 0),
            }

        if summary:
            json_path = os.path.join(output_dir, "results.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)

    def _load_cached_results(self) -> Dict:
        cache_path = os.path.join(self.run_dir, "ablation_compare_results", "results.json")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    return copy.deepcopy(json.load(f))
            except:
                pass
        return {}

    def run(self, start_from: str = 'no_rule'):
        """运行三模式对比测试"""
        print("\n" + "=" * 70)
        print("🧪 Ablation 对比测试：验证 RL 策略状态管理的贡献")
        print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📁 运行目录: {self.run_dir}")
        print(f"⚡ 并行线程: {self.test_workers}")
        print(f"🔧 执行模式: {self.exec_mode} ({'直接执行策略' if self.exec_mode=='direct' else '注入LLM参考'})")
        print("=" * 70)

        all_results = {}
        total_start = time.time()

        modes = ['no_rule'] + sorted([m for m in self.round_policies.keys()], key=lambda x: int(x.split('_')[1]))

        # ★ 三种过滤模式
        filter_modes = ['active_only', 'active_trial', 'all']

        cached = self._load_cached_results()

        for mode in modes:
            for filter_mode in filter_modes:
                key = f"{mode}_{filter_mode}"

                if key in cached and cached[key].get('window_count', 0) > 0:
                    print(f"\n   📦 使用缓存结果: {key}")
                    all_results[key] = copy.deepcopy(cached[key])
                    continue

                if mode == 'no_rule':
                    policies = []
                    # no_rule 不受 filter_mode 影响，只跑一次
                    if filter_mode == 'active_trial' or filter_mode == 'all':
                        continue
                else:
                    policies = self.round_policies.get(mode, [])

                start = time.time()
                result = self.evaluate_mode(mode, policies, filter_mode)
                result['elapsed'] = time.time() - start
                all_results[key] = result

                self._save_intermediate_results(all_results)

        total_elapsed = time.time() - total_start

        # ★ 打印多指标对比报告
        self._print_comparison_report(all_results, total_elapsed)

        # ★ 生成对比图像
        self._generate_comparison_plots(all_results)

        # ★ 保存最终结果
        self._save_final_results(all_results)

    def _print_comparison_report(self, all_results: Dict, total_elapsed: float):
        """打印三模式对比报告（★ 五指标完整对比 + 改善百分比）"""
        print("\n" + "=" * 180)
        print("📊 三模式多指标对比报告：ACTIVE only vs ACTIVE+TRIAL vs ALL policies")
        print("=" * 180)

        # ★ 提取 no_rule 基线
        no_rule_key = 'no_rule_active_only'
        no_rule_data = all_results.get(no_rule_key, {})
        no_rule_mase = no_rule_data.get('avg_mase', float('inf'))
        no_rule_valid = no_rule_mase != float('inf') and not math.isnan(no_rule_mase)

        print(f"\n{'轮次':<12} | {'过滤模式':<16} | {'MASE':<12} | {'MAE':<12} | {'RMSE':<12} | {'SMAPE':<12} | {'OWA':<12} | {'vs ALL':<14}")
        print("-" * 180)

        modes = sorted([m for m in all_results.keys() if m != 'no_rule_active_only'],
                       key=lambda x: int(x.split('_')[1]) if x.startswith('round_') else 0)

        for mode_key in modes:
            if not mode_key.startswith('round_'):
                continue

            mode_name = mode_key.split('_')[1] if '_' in mode_key else mode_key

            # ★ 获取三种模式的结果
            active_key = f"round_{mode_name}_active_only"
            trial_key = f"round_{mode_name}_active_trial"
            all_key = f"round_{mode_name}_all"

            active_data = all_results.get(active_key, {})
            trial_data = all_results.get(trial_key, {})
            all_data = all_results.get(all_key, {})

            active_valid = active_data.get('avg_mase', float('inf')) != float('inf')
            trial_valid = trial_data.get('avg_mase', float('inf')) != float('inf')
            all_valid = all_data.get('avg_mase', float('inf')) != float('inf')

            # ★ ACTIVE only 行（改善百分比相对于 ALL）
            if active_valid:
                active_mase = active_data.get('avg_mase', 0)
                active_mae = active_data.get('avg_mae', 0)
                active_rmse = active_data.get('avg_rmse', 0)
                active_smape = active_data.get('avg_smape', 0)
                active_owa = active_data.get('avg_owa', 0)

                imp_mase = ""
                imp_mae = ""
                imp_rmse = ""
                imp_smape = ""
                imp_owa = ""

                if all_valid:
                    all_mase = all_data.get('avg_mase', 0)
                    all_mae = all_data.get('avg_mae', 0)
                    all_rmse = all_data.get('avg_rmse', 0)
                    all_smape = all_data.get('avg_smape', 0)
                    all_owa = all_data.get('avg_owa', 0)

                    if all_mase > 0:
                        imp_mase = f"{(all_mase - active_mase) / all_mase * 100:+.2f}%"
                    if all_mae > 0:
                        imp_mae = f"{(all_mae - active_mae) / all_mae * 100:+.2f}%"
                    if all_rmse > 0:
                        imp_rmse = f"{(all_rmse - active_rmse) / all_rmse * 100:+.2f}%"
                    if all_smape > 0:
                        imp_smape = f"{(all_smape - active_smape) / all_smape * 100:+.2f}%"
                    if all_owa > 0:
                        imp_owa = f"{(all_owa - active_owa) / all_owa * 100:+.2f}%"

                print(f"R{mode_name:<9} | {'ACTIVE only':<16} | {active_mase:<12.6f} | {active_mae:<12.6f} | {active_rmse:<12.6f} | {active_smape:<12.6f} | {active_owa:<12.6f} | {imp_mase:<14}")

            # ★ ACTIVE+TRIAL 行（改善百分比相对于 ALL）
            if trial_valid:
                trial_mase = trial_data.get('avg_mase', 0)
                trial_mae = trial_data.get('avg_mae', 0)
                trial_rmse = trial_data.get('avg_rmse', 0)
                trial_smape = trial_data.get('avg_smape', 0)
                trial_owa = trial_data.get('avg_owa', 0)

                imp_mase = ""
                imp_mae = ""
                imp_rmse = ""
                imp_smape = ""
                imp_owa = ""

                if all_valid:
                    all_mase = all_data.get('avg_mase', 0)
                    all_mae = all_data.get('avg_mae', 0)
                    all_rmse = all_data.get('avg_rmse', 0)
                    all_smape = all_data.get('avg_smape', 0)
                    all_owa = all_data.get('avg_owa', 0)

                    if all_mase > 0:
                        imp_mase = f"{(all_mase - trial_mase) / all_mase * 100:+.2f}%"
                    if all_mae > 0:
                        imp_mae = f"{(all_mae - trial_mae) / all_mae * 100:+.2f}%"
                    if all_rmse > 0:
                        imp_rmse = f"{(all_rmse - trial_rmse) / all_rmse * 100:+.2f}%"
                    if all_smape > 0:
                        imp_smape = f"{(all_smape - trial_smape) / all_smape * 100:+.2f}%"
                    if all_owa > 0:
                        imp_owa = f"{(all_owa - trial_owa) / all_owa * 100:+.2f}%"

                print(f"{'':<12} | {'ACTIVE+TRIAL':<16} | {trial_mase:<12.6f} | {trial_mae:<12.6f} | {trial_rmse:<12.6f} | {trial_smape:<12.6f} | {trial_owa:<12.6f} | {imp_mase:<14}")

            # ★ ALL 行（基准，无改善）
            if all_valid:
                all_mase = all_data.get('avg_mase', 0)
                all_mae = all_data.get('avg_mae', 0)
                all_rmse = all_data.get('avg_rmse', 0)
                all_smape = all_data.get('avg_smape', 0)
                all_owa = all_data.get('avg_owa', 0)
                print(f"{'':<12} | {'ALL':<16} | {all_mase:<12.6f} | {all_mae:<12.6f} | {all_rmse:<12.6f} | {all_smape:<12.6f} | {all_owa:<12.6f} | {'—':<14}")

            # ★ 轮次分隔
            print("-" * 180)

        # ★ 汇总统计
        print(f"\n📊 汇总统计:")
        if no_rule_valid:
            print(f"   no_rule MASE: {no_rule_mase:.6f}")
        print(f"   总耗时: {total_elapsed:.1f}s ({total_elapsed/60:.1f}分钟)")

        # ★ 计算平均改善（所有轮次）
        avg_improvements_active = []
        avg_improvements_trial = []
        for mode_key in modes:
            if not mode_key.startswith('round_'):
                continue
            mode_name = mode_key.split('_')[1]
            active_key = f"round_{mode_name}_active_only"
            trial_key = f"round_{mode_name}_active_trial"
            all_key = f"round_{mode_name}_all"

            active_data = all_results.get(active_key, {})
            trial_data = all_results.get(trial_key, {})
            all_data = all_results.get(all_key, {})

            # ACTIVE vs ALL
            if (active_data.get('avg_mase', float('inf')) != float('inf') and
                all_data.get('avg_mase', float('inf')) != float('inf') and
                all_data.get('avg_mase', 0) > 0):
                imp = (all_data['avg_mase'] - active_data['avg_mase']) / all_data['avg_mase'] * 100
                avg_improvements_active.append(imp)

            # ACTIVE+TRIAL vs ALL
            if (trial_data.get('avg_mase', float('inf')) != float('inf') and
                all_data.get('avg_mase', float('inf')) != float('inf') and
                all_data.get('avg_mase', 0) > 0):
                imp = (all_data['avg_mase'] - trial_data['avg_mase']) / all_data['avg_mase'] * 100
                avg_improvements_trial.append(imp)

        if avg_improvements_active:
            avg_imp = np.mean(avg_improvements_active)
            std_imp = np.std(avg_improvements_active)
            print(f"   平均 MASE 改善 (ACTIVE vs ALL): {avg_imp:+.2f}% ± {std_imp:.2f}%")

        if avg_improvements_trial:
            avg_imp = np.mean(avg_improvements_trial)
            std_imp = np.std(avg_improvements_trial)
            print(f"   平均 MASE 改善 (ACTIVE+TRIAL vs ALL): {avg_imp:+.2f}% ± {std_imp:.2f}%")

    def _generate_comparison_plots(self, all_results: Dict):
        """生成三模式对比图像"""
        output_dir = os.path.join(self.run_dir, "ablation_compare_results")
        os.makedirs(output_dir, exist_ok=True)

        # ★ 提取数据
        modes = sorted([m for m in all_results.keys() if m.startswith('round_')],
                       key=lambda x: int(x.split('_')[1]) if x.startswith('round_') else 0)

        active_mases = []
        trial_mases = []
        all_mases = []
        mode_labels = []

        for mode_key in modes:
            mode_name = mode_key.split('_')[1]
            active_key = f"round_{mode_name}_active_only"
            trial_key = f"round_{mode_name}_active_trial"
            all_key = f"round_{mode_name}_all"

            active_data = all_results.get(active_key, {})
            trial_data = all_results.get(trial_key, {})
            all_data = all_results.get(all_key, {})

            active_valid = active_data.get('avg_mase', float('inf')) != float('inf')
            trial_valid = trial_data.get('avg_mase', float('inf')) != float('inf')
            all_valid = all_data.get('avg_mase', float('inf')) != float('inf')

            if active_valid:
                active_mases.append(active_data['avg_mase'])
                trial_mases.append(trial_data['avg_mase'] if trial_valid else float('nan'))
                all_mases.append(all_data['avg_mase'] if all_valid else float('nan'))
                mode_labels.append(f"R{mode_name}")

        if not mode_labels:
            print("⚠️ 无足够数据生成图像")
            return

        # ★ 图1：柱状图 - 三模式 MASE 对比
        fig, ax = plt.subplots(figsize=(14, 7))

        x = np.arange(len(mode_labels))
        width = 0.25

        active_plot = [m if not np.isnan(m) else None for m in active_mases]
        trial_plot = [m if not np.isnan(m) else None for m in trial_mases]
        all_plot = [m if not np.isnan(m) else None for m in all_mases]

        ax.bar(x - width, active_plot, width, label='ACTIVE only', color='#2E86AB', alpha=0.8)
        ax.bar(x, trial_plot, width, label='ACTIVE+TRIAL', color='#F5A623', alpha=0.8)
        ax.bar(x + width, all_plot, width, label='ALL policies', color='#A23B72', alpha=0.8)

        ax.set_xlabel('轮次', fontsize=12)
        ax.set_ylabel('MASE', fontsize=12)
        ax.set_title('三模式对比：ACTIVE only vs ACTIVE+TRIAL vs ALL policies\n(验证 RL 训练中"策略状态管理"的贡献)', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(mode_labels)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        bar_path = os.path.join(output_dir, "comparison_bar_chart.png")
        plt.savefig(bar_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   📊 柱状图已保存: {bar_path}")

        # ★ 图2：箱线图 - 三模式 MASE 分布对比
        fig, ax = plt.subplots(figsize=(14, 7))

        active_mases_all = []
        trial_mases_all = []
        all_mases_all = []
        positions = []

        for idx, mode_key in enumerate(modes):
            mode_name = mode_key.split('_')[1]
            active_key = f"round_{mode_name}_active_only"
            trial_key = f"round_{mode_name}_active_trial"
            all_key = f"round_{mode_name}_all"

            active_data = all_results.get(active_key, {})
            trial_data = all_results.get(trial_key, {})
            all_data = all_results.get(all_key, {})

            if active_data.get('mases'):
                active_mases_all.append(active_data['mases'])
            if trial_data.get('mases'):
                trial_mases_all.append(trial_data['mases'])
            if all_data.get('mases'):
                all_mases_all.append(all_data['mases'])
            positions.append(idx * 3)

        # ★ 箱线图需要至少 2 组数据才有意义
        if len(active_mases_all) >= 2:
            bp1 = ax.boxplot(active_mases_all, positions=[p - 0.8 for p in positions[:len(active_mases_all)]],
                             widths=0.6, patch_artist=True,
                             boxprops=dict(facecolor='#2E86AB', alpha=0.6),
                             medianprops=dict(color='#2E86AB', linewidth=2),
                             whiskerprops=dict(color='#2E86AB'),
                             capprops=dict(color='#2E86AB'),
                             flierprops=dict(marker='o', markerfacecolor='#2E86AB', markersize=4, alpha=0.5))

        if len(trial_mases_all) >= 2:
            bp2 = ax.boxplot(trial_mases_all, positions=[p for p in positions[:len(trial_mases_all)]],
                             widths=0.6, patch_artist=True,
                             boxprops=dict(facecolor='#F5A623', alpha=0.6),
                             medianprops=dict(color='#F5A623', linewidth=2),
                             whiskerprops=dict(color='#F5A623'),
                             capprops=dict(color='#F5A623'),
                             flierprops=dict(marker='o', markerfacecolor='#F5A623', markersize=4, alpha=0.5))

        if len(all_mases_all) >= 2:
            bp3 = ax.boxplot(all_mases_all, positions=[p + 0.8 for p in positions[:len(all_mases_all)]],
                             widths=0.6, patch_artist=True,
                             boxprops=dict(facecolor='#A23B72', alpha=0.6),
                             medianprops=dict(color='#A23B72', linewidth=2),
                             whiskerprops=dict(color='#A23B72'),
                             capprops=dict(color='#A23B72'),
                             flierprops=dict(marker='o', markerfacecolor='#A23B72', markersize=4, alpha=0.5))

        ax.set_xlabel('轮次', fontsize=12)
        ax.set_ylabel('MASE', fontsize=12)
        ax.set_title('三模式各轮次 MASE 分布对比', fontsize=14)
        ax.set_xticks(positions[:len(active_mases_all)])
        ax.set_xticklabels([f"R{i+1}" for i in range(len(active_mases_all))])

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2E86AB', alpha=0.6, label='ACTIVE only'),
            Patch(facecolor='#F5A623', alpha=0.6, label='ACTIVE+TRIAL'),
            Patch(facecolor='#A23B72', alpha=0.6, label='ALL policies')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        box_path = os.path.join(output_dir, "comparison_boxplot.png")
        plt.savefig(box_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   📊 箱线图已保存: {box_path}")

        # ★ 图3：改善百分比折线图（两种改善：ACTIVE vs ALL, ACTIVE+TRIAL vs ALL）
        fig, ax = plt.subplots(figsize=(12, 6))

        improvements_active = []
        improvements_trial = []
        imp_modes = []

        for idx, mode_key in enumerate(modes):
            if not mode_key.startswith('round_'):
                continue
            mode_name = mode_key.split('_')[1]
            active_key = f"round_{mode_name}_active_only"
            trial_key = f"round_{mode_name}_active_trial"
            all_key = f"round_{mode_name}_all"

            active_data = all_results.get(active_key, {})
            trial_data = all_results.get(trial_key, {})
            all_data = all_results.get(all_key, {})

            # ACTIVE vs ALL
            if (active_data.get('avg_mase', float('inf')) != float('inf') and
                all_data.get('avg_mase', float('inf')) != float('inf') and
                all_data.get('avg_mase', 0) > 0):
                imp = (all_data['avg_mase'] - active_data['avg_mase']) / all_data['avg_mase'] * 100
                improvements_active.append(imp)
            else:
                improvements_active.append(float('nan'))

            # ACTIVE+TRIAL vs ALL
            if (trial_data.get('avg_mase', float('inf')) != float('inf') and
                all_data.get('avg_mase', float('inf')) != float('inf') and
                all_data.get('avg_mase', 0) > 0):
                imp = (all_data['avg_mase'] - trial_data['avg_mase']) / all_data['avg_mase'] * 100
                improvements_trial.append(imp)
            else:
                improvements_trial.append(float('nan'))

            imp_modes.append(f"R{mode_name}")

        if improvements_active or improvements_trial:
            # 过滤掉 nan 值
            valid_indices = [i for i in range(len(imp_modes))
                            if not np.isnan(improvements_active[i]) or not np.isnan(improvements_trial[i])]
            valid_modes = [imp_modes[i] for i in valid_indices]
            valid_active = [improvements_active[i] for i in valid_indices]
            valid_trial = [improvements_trial[i] for i in valid_indices]

            if valid_active:
                ax.plot(valid_modes, valid_active, marker='o', color='#2E86AB',
                        linewidth=2, markersize=8, label='ACTIVE vs ALL')
            if valid_trial:
                ax.plot(valid_modes, valid_trial, marker='s', color='#F5A623',
                        linewidth=2, markersize=8, label='ACTIVE+TRIAL vs ALL')

            ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='基线 (0%)')
            ax.set_xlabel('轮次', fontsize=12)
            ax.set_ylabel('MASE 改善百分比 (%)', fontsize=12)
            ax.set_title('三模式改善百分比对比', fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)

            # ★ 添加数值标注
            for i, v in enumerate(valid_active):
                if not np.isnan(v):
                    ax.text(i, v + 0.5, f"{v:.1f}%", ha='center', fontsize=8, color='#2E86AB')

            plt.tight_layout()
            imp_path = os.path.join(output_dir, "improvement_line_chart.png")
            plt.savefig(imp_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"   📊 改善折线图已保存: {imp_path}")

    def _save_final_results(self, all_results: Dict):
        """保存最终结果"""
        output_dir = os.path.join(self.run_dir, "ablation_compare_results")
        os.makedirs(output_dir, exist_ok=True)

        # ★ 保存 JSON
        json_path = os.path.join(output_dir, "results.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)

        # ★ 生成详细报告
        report_path = os.path.join(output_dir, "comparison_report.txt")
        lines = []
        lines.append("=" * 180)
        lines.append("🧪 Ablation 对比测试报告：ACTIVE only vs ACTIVE+TRIAL vs ALL policies")
        lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"运行目录: {self.run_dir}")
        lines.append("=" * 180)
        lines.append("")
        lines.append("模式说明:")
        lines.append("  - ACTIVE only   : 仅从 ACTIVE 策略中选择（模拟使用 RL 训练参数）")
        lines.append("  - ACTIVE+TRIAL  : 从 ACTIVE + TRIAL 策略中选择（验证候补策略的影响）")
        lines.append("  - ALL policies  : 从所有策略中选择（模拟不使用 RL 训练参数）")
        lines.append("")
        lines.append("改善百分比 = (ALL - 目标模式) / ALL * 100% (正值表示目标模式更优)")
        lines.append("")
        lines.append(f"★ ★★ 执行模式：{self.exec_mode} ({'直接执行策略' if self.exec_mode=='direct' else '注入LLM参考'})")

        lines.append(f"{'轮次':<12} | {'过滤模式':<16} | {'MASE':<12} | {'MAE':<12} | {'RMSE':<12} | {'SMAPE':<12} | {'OWA':<12} | {'vs ALL':<14}")
        lines.append("-" * 180)

        modes = sorted([m for m in all_results.keys() if m.startswith('round_')],
                       key=lambda x: int(x.split('_')[1]) if x.startswith('round_') else 0)

        for mode_key in modes:
            if not mode_key.startswith('round_'):
                continue

            mode_name = mode_key.split('_')[1]

            active_key = f"round_{mode_name}_active_only"
            trial_key = f"round_{mode_name}_active_trial"
            all_key = f"round_{mode_name}_all"

            active_data = all_results.get(active_key, {})
            trial_data = all_results.get(trial_key, {})
            all_data = all_results.get(all_key, {})

            active_valid = active_data.get('avg_mase', float('inf')) != float('inf')
            trial_valid = trial_data.get('avg_mase', float('inf')) != float('inf')
            all_valid = all_data.get('avg_mase', float('inf')) != float('inf')

            # ★ ACTIVE only 行
            if active_valid:
                active_mase = active_data.get('avg_mase', 0)
                active_mae = active_data.get('avg_mae', 0)
                active_rmse = active_data.get('avg_rmse', 0)
                active_smape = active_data.get('avg_smape', 0)
                active_owa = active_data.get('avg_owa', 0)

                imp_mase = ""
                if all_valid and all_data.get('avg_mase', 0) > 0:
                    imp_mase = f"{(all_data['avg_mase'] - active_mase) / all_data['avg_mase'] * 100:+.2f}%"

                lines.append(f"R{mode_name:<9} | {'ACTIVE only':<16} | {active_mase:<12.6f} | {active_mae:<12.6f} | {active_rmse:<12.6f} | {active_smape:<12.6f} | {active_owa:<12.6f} | {imp_mase:<14}")

            # ★ ACTIVE+TRIAL 行
            if trial_valid:
                trial_mase = trial_data.get('avg_mase', 0)
                trial_mae = trial_data.get('avg_mae', 0)
                trial_rmse = trial_data.get('avg_rmse', 0)
                trial_smape = trial_data.get('avg_smape', 0)
                trial_owa = trial_data.get('avg_owa', 0)

                imp_mase = ""
                if all_valid and all_data.get('avg_mase', 0) > 0:
                    imp_mase = f"{(all_data['avg_mase'] - trial_mase) / all_data['avg_mase'] * 100:+.2f}%"

                lines.append(f"{'':<12} | {'ACTIVE+TRIAL':<16} | {trial_mase:<12.6f} | {trial_mae:<12.6f} | {trial_rmse:<12.6f} | {trial_smape:<12.6f} | {trial_owa:<12.6f} | {imp_mase:<14}")

            # ★ ALL 行
            if all_valid:
                all_mase = all_data.get('avg_mase', 0)
                all_mae = all_data.get('avg_mae', 0)
                all_rmse = all_data.get('avg_rmse', 0)
                all_smape = all_data.get('avg_smape', 0)
                all_owa = all_data.get('avg_owa', 0)
                lines.append(f"{'':<12} | {'ALL':<16} | {all_mase:<12.6f} | {all_mae:<12.6f} | {all_rmse:<12.6f} | {all_smape:<12.6f} | {all_owa:<12.6f} | {'—':<14}")

            lines.append("-" * 180)

        # ★ 汇总统计
        avg_improvements_active = []
        avg_improvements_trial = []
        for mode_key in modes:
            if not mode_key.startswith('round_'):
                continue
            mode_name = mode_key.split('_')[1]
            active_key = f"round_{mode_name}_active_only"
            trial_key = f"round_{mode_name}_active_trial"
            all_key = f"round_{mode_name}_all"

            active_data = all_results.get(active_key, {})
            trial_data = all_results.get(trial_key, {})
            all_data = all_results.get(all_key, {})

            if (active_data.get('avg_mase', float('inf')) != float('inf') and
                all_data.get('avg_mase', float('inf')) != float('inf') and
                all_data.get('avg_mase', 0) > 0):
                imp = (all_data['avg_mase'] - active_data['avg_mase']) / all_data['avg_mase'] * 100
                avg_improvements_active.append(imp)

            if (trial_data.get('avg_mase', float('inf')) != float('inf') and
                all_data.get('avg_mase', float('inf')) != float('inf') and
                all_data.get('avg_mase', 0) > 0):
                imp = (all_data['avg_mase'] - trial_data['avg_mase']) / all_data['avg_mase'] * 100
                avg_improvements_trial.append(imp)

        if avg_improvements_active:
            avg_imp = np.mean(avg_improvements_active)
            std_imp = np.std(avg_improvements_active)
            lines.append(f"\n平均 MASE 改善 (ACTIVE vs ALL): {avg_imp:+.2f}% ± {std_imp:.2f}%")

        if avg_improvements_trial:
            avg_imp = np.mean(avg_improvements_trial)
            std_imp = np.std(avg_improvements_trial)
            lines.append(f"平均 MASE 改善 (ACTIVE+TRIAL vs ALL): {avg_imp:+.2f}% ± {std_imp:.2f}%")

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        print(f"\n📁 结果已保存: {output_dir}")
        print(f"   - results.json (详细数据)")
        print(f"   - comparison_report.txt (对比报告)")
        print(f"   - comparison_bar_chart.png (柱状图)")
        print(f"   - comparison_boxplot.png (箱线图)")
        print(f"   - improvement_line_chart.png (改善折线图)")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Ablation 对比测试：验证 RL 策略状态管理的贡献")
    parser.add_argument('--resume', type=str, required=True,
                        help='运行目录（如 run_20260624_043426）')
    parser.add_argument('--config', type=str, default=None,
                        help='配置文件路径')
    parser.add_argument('--workers', type=int, default=10,
                        help='并行线程数（默认 10）')
    parser.add_argument('--start', type=str, default='no_rule',
                        help='起始模式（默认 no_rule）')
    parser.add_argument('--test-ratio', type=float, default=0.5,
                        help='从 B 子集抽取测试集的比例（默认 0.5）')
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

    tester = AblationCompareTester(run_dir, args.config, test_ratio=args.test_ratio,
                                   exec_mode=args.exec_mode)
    tester.test_workers = args.workers
    tester.run(start_from=args.start)


if __name__ == '__main__':
    main()