#!/usr/bin/env python
"""
#python -m experiments.autotune.quick_test --resume run_20260626_133418 --workers 10 --baseline round_1 --test-ratio 0.5

快速测试脚本：独立加载各轮策略，在测试集上评估
★ 10 线程并行执行
★ 详细进度条（窗口级 + 模式级）
★ 多指标对比（MASE、MAE、RMSE、SMAPE、OWA）
★ 支持 --start 参数从指定轮次开始
★ 断点续存（自动跳过已完成轮次）
★ 支持 --baseline 参数（no_rule 或 round_1）
★ 测试集比例可配置（--test-ratio）
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

# ★ 窗口超时时间（秒）
WINDOW_TIMEOUT = 120


class QuickTester:
    def __init__(self, run_dir: str, config_path: str = None, test_ratio: float = 0.5):
        self.run_dir = run_dir
        self.config = load_config(config_path)
        self.output_dir = self.config.get('output_dir', 'storage/autotune_results')
        self.test_workers = 10
        self.test_ratio = test_ratio  # ★ 从 B 子集抽取测试集的比例
        self._timeout_counter = 0

        # ★ 构建共享技能注册表（所有线程复用）
        print("   🔧 构建技能注册表...")
        self.full_registry, _ = build_full_registry()
        print(f"   ✅ 注册了 {len(self.full_registry._skills)} 个技能")

        # ★ 状态编码器（线程安全，只读）
        self.state_encoder = StateEncoder(self.config)

        # ★ 检测可用模型
        self.model = self._detect_model()

        # ★ 加载测试集（支持比例配置）
        self.test_df = self._load_test_df()

        # ★ 加载各轮策略
        self.round_policies = {}
        self._load_all_round_policies()

        # ★ 结果存储
        self.results = {}
        self._lock = threading.Lock()

    def _load_test_df(self) -> pd.DataFrame:
        """
        加载测试集数据
        ★ 方案A：从 B1/B2 子集各取 test_ratio（默认 50%）作为测试集
        """
        csv_path = os.path.join(self.output_dir, "collected_windows.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"❌ 未找到采集数据: {csv_path}")

        df = pd.read_csv(csv_path)

        # ★ 检查是否已有 split 列
        if 'split' in df.columns:
            # 如果已经有 test 标签，直接使用（兼容已有数据）
            test_df = df[df['split'] == 'test'].copy()
            if len(test_df) > 0:
                print(f"📊 使用已有测试集标签: {len(test_df)} 个窗口")
                return test_df

        # ★ 如果没有 test 标签，按比例从 B 子集抽取
        # 识别 B 子集（假设 split 为 'B' 或 'B1'/'B2'）
        b_mask = df['split'].str.startswith('B') if 'split' in df.columns else pd.Series([True] * len(df))
        if 'split' not in df.columns:
            # 如果没有 split 列，按窗口顺序划分
            n = len(df)
            # 假设前 50% 是 A，后 50% 是 B
            a_end = int(n * 0.5)
            b_mask = pd.Series([False] * n)
            b_mask.iloc[a_end:] = True
            df['split'] = ['A'] * a_end + ['B'] * (n - a_end)

        b_df = df[b_mask].copy()
        if len(b_df) == 0:
            print("⚠️ 未找到 B 子集，使用全量数据")
            return df

        # ★ 按 window_id 排序，确保可复现
        b_df = b_df.sort_values('window_id').reset_index(drop=True)

        # ★ 从 B 子集中按比例抽取
        n_b = len(b_df)
        test_size = int(n_b * self.test_ratio)
        test_df = b_df.iloc[:test_size].copy()
        test_df['split'] = 'test'

        print(f"📊 测试集: {len(test_df)} 个窗口 (从 {n_b} 个 B 窗口抽取 {self.test_ratio:.0%})")
        print(f"   窗口 ID: {test_df['window_id'].tolist()[:10]}{'...' if len(test_df) > 10 else ''}")

        return test_df

    def _load_round_policies(self, round_num: int) -> List[SkillPolicy]:
        """加载指定轮次的策略"""
        path = os.path.join(self.run_dir, f"round_{round_num}", "refined_policies_optimized.json")
        if not os.path.exists(path):
            return []

        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            policies_data = data.get('policies', [])
            return [SkillPolicy.from_dict(p) for p in policies_data]
        except Exception as e:
            print(f"   ⚠️ 加载第 {round_num} 轮失败: {e}")
            return []

    def _load_all_round_policies(self):
        """加载所有轮次的策略"""
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
                    # 检查是否有下一轮
                    next_path = os.path.join(self.run_dir, f"round_{round_num + 1}", "refined_policies_optimized.json")
                    if not os.path.exists(os.path.dirname(next_path)):
                        break
                round_num += 1

        if self.round_policies:
            print(f"\n📊 共加载 {len(self.round_policies)} 轮策略: {list(self.round_policies.keys())}")
        else:
            print(f"\n⚠️ 未找到任何策略文件，仅运行 no_rule 基线")

    def _detect_model(self) -> Optional[str]:
        """检测可用模型"""
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

    def _predict_no_rule(self, task: TaskInstance) -> Optional[np.ndarray]:
        """无策略预测（LLM 直接预测）"""
        if self.model is None:
            return None

        try:
            agent = self._create_agent()
            agent.rule_engine = None
            agent._current_rule_strategy = None
            pred = agent.predict(task)
            return np.array(pred) if pred else None
        except Exception:
            return None

    def _predict_with_policies(self, task: TaskInstance, policies: List[SkillPolicy],
                               train: np.ndarray, horizon: int, period: int,
                               verbose: bool = False) -> Optional[np.ndarray]:
        """使用策略池预测"""
        if not policies:
            return None

        try:
            window_id = task.id.replace('test_', '') if task.id else 'unknown'

            state = self.state_encoder.encode(train)
            numeric_state = state.get('numeric', {})

            scored = []
            for policy in policies:
                if policy.status in ['ARCHIVE', 'DELETE']:
                    continue
                try:
                    score = policy.compute_applicability_score(numeric_state)
                    scored.append((policy, score))
                except Exception:
                    continue

            if not scored:
                return None

            best_policy, best_score = max(scored, key=lambda x: x[1])

            if verbose:
                print(f"      [诊断] 窗口 {window_id} 选择: {best_policy.name[:20]} (分数={best_score:.4f})")

            result = best_policy.execute(train, horizon, period)
            return result

        except Exception:
            return None

    def evaluate_single_window(self, idx: int, row: pd.Series,
                               mode: str, policies: List[SkillPolicy],
                               verbose: bool = False) -> Dict:
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
                pred = self._predict_no_rule(task)
            else:
                pred = self._predict_with_policies(task, policies, train, horizon, period, verbose)

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
                'pred': pred.tolist(),
                'actual': test.tolist()
            }

        except Exception as e:
            return {'window_id': window_id, 'success': False, 'error': str(e)}

    def evaluate_mode(self, mode: str, policies: List[SkillPolicy],
                      verbose: bool = False) -> Dict:
        """评估一个模式的所有窗口"""
        tasks = [(idx, row) for idx, row in self.test_df.iterrows()]
        total = len(tasks)

        policy_count = len(policies) if policies else 0

        # ★ 进度条显示
        pbar_desc = f"   {mode:>12}" + (f" ({policy_count}策略)" if policy_count > 0 else " (无策略)")

        results = []
        mases = []
        maes = []
        rmses = []
        smapes = []
        owas = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.test_workers) as executor:
            futures = {}
            for idx, row in tasks:
                future = executor.submit(
                    self.evaluate_single_window,
                    idx, row, mode, policies, verbose
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

                        pbar.set_postfix({
                            '已处理': f"{len(results)}/{total}",
                            'MASE': f"{avg_mase:.4f}" if avg_mase != float('inf') else '...'
                        })
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
                'mases': [],
                'maes': [],
                'rmses': [],
                'smapes': [],
                'owas': [],
                'window_count': 0,
                'avg_mase': float('inf'),
                'avg_mae': float('inf'),
                'avg_rmse': float('inf'),
                'avg_smape': float('inf'),
                'avg_owa': float('inf'),
            }

        valid_mases = [m for m in mases if m != float('inf') and not np.isnan(m)]
        valid_maes = [m for m in maes if m != float('inf') and not np.isnan(m)]
        valid_rmses = [m for m in rmses if m != float('inf') and not np.isnan(m)]
        valid_smapes = [m for m in smapes if m != float('inf') and not np.isnan(m)]
        valid_owas = [m for m in owas if m != float('inf') and not np.isnan(m)]

        avg_mase = np.mean(valid_mases) if valid_mases else float('inf')
        avg_mae = np.mean(valid_maes) if valid_maes else float('inf')
        avg_rmse = np.mean(valid_rmses) if valid_rmses else float('inf')
        avg_smape = np.mean(valid_smapes) if valid_smapes else float('inf')
        avg_owa = np.mean(valid_owas) if valid_owas else float('inf')

        print(f"   ✅ {mode}: MASE={avg_mase:.6f} ({valid_count}/{total} 窗口)")

        return {
            'mode': mode,
            'success': True,
            'mases': valid_mases,
            'maes': valid_maes,
            'rmses': valid_rmses,
            'smapes': valid_smapes,
            'owas': valid_owas,
            'window_count': valid_count,
            'avg_mase': avg_mase,
            'avg_mae': avg_mae,
            'avg_rmse': avg_rmse,
            'avg_smape': avg_smape,
            'avg_owa': avg_owa,
            'results': results
        }

    def _save_intermediate_results(self, all_results: Dict):
        """保存中间结果（断点续存）"""
        output_dir = os.path.join(self.run_dir, "quick_test_results")
        os.makedirs(output_dir, exist_ok=True)

        summary = {}
        for mode, r in all_results.items():
            avg_mase = r.get('avg_mase', float('inf'))
            if avg_mase == float('inf') or math.isnan(avg_mase):
                continue
            summary[mode] = {
                'avg_mase': avg_mase,
                'avg_mae': r.get('avg_mae', float('inf')),
                'avg_rmse': r.get('avg_rmse', float('inf')),
                'avg_smape': r.get('avg_smape', float('inf')),
                'avg_owa': r.get('avg_owa', float('inf')),
                'window_count': r.get('window_count', 0),
                'elapsed': r.get('elapsed', 0),
            }

        if summary:
            json_path = os.path.join(output_dir, "results.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)

    def _load_cached_results(self) -> Dict:
        """加载缓存的中间结果"""
        cache_path = os.path.join(self.run_dir, "quick_test_results", "results.json")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    return copy.deepcopy(json.load(f))
            except:
                pass
        return {}

    def _is_valid_cache(self, cached_data: Dict) -> bool:
        """检查缓存数据是否有效"""
        if not cached_data:
            return False
        avg_mase = cached_data.get('avg_mase', float('inf'))
        window_count = cached_data.get('window_count', 0)
        if math.isnan(avg_mase) or avg_mase == float('inf'):
            return False
        if window_count <= 0:
            return False
        return True

    def run(self, start_from: str = 'no_rule', baseline: str = 'round_1'):
        """运行所有评估"""
        print("\n" + "=" * 70)
        print("🚀 快速测试：独立评估各轮策略")
        print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📁 运行目录: {self.run_dir}")
        print(f"⚡ 并行线程: {self.test_workers}")
        print(f"⏱️  窗口超时: {WINDOW_TIMEOUT}s")
        print(f"📌 起始模式: {start_from}")
        print(f"📌 基线模式: {baseline}")
        print(f"📌 测试集比例: {self.test_ratio:.0%}")
        print("=" * 70)

        all_results = {}
        total_start = time.time()

        # ★ 获取所有模式并按顺序排列
        modes = ['no_rule'] + sorted([m for m in self.round_policies.keys()], key=lambda x: int(x.split('_')[1]))

        # ★ 加载缓存
        cached = self._load_cached_results()

        # ★ 找到起始位置
        start_idx = 0
        for i, mode in enumerate(modes):
            if mode == start_from:
                start_idx = i
                break
        if start_idx == 0 and start_from != 'no_rule':
            print(f"⚠️ 未找到 '{start_from}'，从 no_rule 开始")
            start_idx = 0

        # ★ 从指定位置开始
        for mode in modes[start_idx:]:
            if mode in cached and self._is_valid_cache(cached[mode]):
                print(f"\n   📦 使用缓存结果: {mode}")
                all_results[mode] = copy.deepcopy({
                    'mode': mode,
                    'success': True,
                    'avg_mase': cached[mode].get('avg_mase', float('inf')),
                    'avg_mae': cached[mode].get('avg_mae', float('inf')),
                    'avg_rmse': cached[mode].get('avg_rmse', float('inf')),
                    'avg_smape': cached[mode].get('avg_smape', float('inf')),
                    'avg_owa': cached[mode].get('avg_owa', float('inf')),
                    'window_count': cached[mode].get('window_count', 0),
                    'elapsed': cached[mode].get('elapsed', 0),
                })
                continue

            if mode == 'no_rule':
                policies = []
            else:
                policies = self.round_policies.get(mode, [])

            start = time.time()
            result = self.evaluate_mode(mode, policies, verbose=False)
            result['elapsed'] = time.time() - start
            all_results[mode] = result

            self._save_intermediate_results(all_results)

        total_elapsed = time.time() - total_start

        if self._timeout_counter > 0:
            print(f"\n⚠️ 共有 {self._timeout_counter} 个窗口超时（> {WINDOW_TIMEOUT}s）")

        # ★ 打印增强的对比报告
        self._print_enhanced_report(all_results, total_elapsed, baseline)

        # ★ 保存最终结果
        self._save_enhanced_results(all_results, baseline)

    def _print_enhanced_report(self, all_results: Dict, total_elapsed: float, baseline: str):
        """打印增强的多指标对比报告"""
        print("\n" + "=" * 100)
        print("📊 Ablation 多指标对比报告")
        print(f"   基线: {baseline}")
        print("=" * 100)

        # ★ 获取基线数据
        if baseline not in all_results:
            baseline = 'no_rule'
            print(f"⚠️ 基线 '{baseline}' 不存在，回退到 'no_rule'")

        baseline_data = all_results.get(baseline, {})
        baseline_mase = baseline_data.get('avg_mase', float('inf'))
        baseline_mae = baseline_data.get('avg_mae', float('inf'))
        baseline_rmse = baseline_data.get('avg_rmse', float('inf'))
        baseline_smape = baseline_data.get('avg_smape', float('inf'))
        baseline_owa = baseline_data.get('avg_owa', float('inf'))
        baseline_valid = baseline_mase != float('inf') and not math.isnan(baseline_mase)

        # ★ 获取 no_rule 数据（用于二次对比）
        no_rule_data = all_results.get('no_rule', {})
        no_rule_mase = no_rule_data.get('avg_mase', float('inf'))
        no_rule_valid = no_rule_mase != float('inf') and not math.isnan(no_rule_mase)

        print(f"\n{'模式':<14} | {'MASE':<12} | {'MAE':<12} | {'RMSE':<12} | {'SMAPE':<12} | {'OWA':<12} | {'窗口':<6} | {'vs ' + baseline + '':<12} | {'vs no_rule':<12}")
        print("-" * 120)

        modes = ['no_rule'] + sorted([m for m in all_results.keys() if m != 'no_rule'], key=lambda x: int(x.split('_')[1]))

        best_mode = 'no_rule'
        best_mase = no_rule_mase if no_rule_valid else float('inf')

        for mode in modes:
            r = all_results.get(mode, {})
            mase = r.get('avg_mase', float('inf'))
            mae = r.get('avg_mae', float('inf'))
            rmse = r.get('avg_rmse', float('inf'))
            smape = r.get('avg_smape', float('inf'))
            owa = r.get('avg_owa', float('inf'))
            count = r.get('window_count', 0)
            elapsed = r.get('elapsed', 0)

            # ★ 计算相对基线的改善
            imp_baseline = None
            if baseline_valid and mase != float('inf') and not math.isnan(mase):
                imp_baseline = (baseline_mase - mase) / baseline_mase * 100

            # ★ 计算相对 no_rule 的改善
            imp_no_rule = None
            if no_rule_valid and mase != float('inf') and not math.isnan(mase):
                imp_no_rule = (no_rule_mase - mase) / no_rule_mase * 100

            # ★ 高亮最佳
            is_best = False
            if mase != float('inf') and not math.isnan(mase) and mase < best_mase:
                best_mase = mase
                best_mode = mode
                is_best = True

            # ★ 标记是否为基线
            if mode == baseline:
                mode_display = f"{mode}*"  # * 标记基线
            elif is_best and mode != baseline:
                mode_display = f"{mode}🏆"
            else:
                mode_display = mode

            # ★ 格式化输出
            mase_str = f"{mase:.6f}" if mase != float('inf') and not math.isnan(mase) else "N/A"
            mae_str = f"{mae:.6f}" if mae != float('inf') and not math.isnan(mae) else "N/A"
            rmse_str = f"{rmse:.6f}" if rmse != float('inf') and not math.isnan(rmse) else "N/A"
            smape_str = f"{smape:.6f}" if smape != float('inf') and not math.isnan(smape) else "N/A"
            owa_str = f"{owa:.6f}" if owa != float('inf') and not math.isnan(owa) else "N/A"

            imp_baseline_str = f"{imp_baseline:>+6.2f}%" if imp_baseline is not None else "N/A"
            imp_no_rule_str = f"{imp_no_rule:>+6.2f}%" if imp_no_rule is not None else "N/A"

            # ★ 基线行不显示改善百分比（本身是0%）
            if mode == baseline:
                imp_baseline_str = "  —   "

            # ★ 判断状态
            if mode == baseline:
                status = "基线"
            elif imp_baseline is not None and imp_baseline > 2:
                status = "✅"
            elif imp_baseline is not None and imp_baseline > 0:
                status = "📈"
            else:
                status = "  "

            print(f"{mode_display:<14} | {mase_str:<12} | {mae_str:<12} | {rmse_str:<12} | {smape_str:<12} | {owa_str:<12} | {count:<6} | {imp_baseline_str:<12} | {imp_no_rule_str:<12}")

            # ★ 如果是基线，打印分隔线
            if mode == baseline:
                print("-" * 120)

        print("=" * 120)

        # ★ 最佳轮次
        if best_mase != float('inf') and not math.isnan(best_mase):
            print(f"\n🏆 最佳策略 (MASE): {best_mode} (MASE={best_mase:.6f})")
            if baseline_valid and best_mode != baseline:
                imp = (baseline_mase - best_mase) / baseline_mase * 100
                print(f"   相比基线 {baseline} 改善: {imp:+.2f}%")
            if no_rule_valid and best_mode != 'no_rule':
                imp = (no_rule_mase - best_mase) / no_rule_mase * 100
                print(f"   相比 no_rule 改善: {imp:+.2f}%")
        else:
            print(f"\n⚠️ 无有效最佳策略")

        print(f"\n⏱️  总耗时: {total_elapsed:.1f}s ({total_elapsed/60:.1f}分钟)")

    def _save_enhanced_results(self, all_results: Dict, baseline: str):
        """保存增强的结果"""
        output_dir = os.path.join(self.run_dir, "quick_test_results")
        os.makedirs(output_dir, exist_ok=True)

        summary = {}
        for mode, r in all_results.items():
            summary[mode] = {
                'avg_mase': r.get('avg_mase', float('inf')),
                'avg_mae': r.get('avg_mae', float('inf')),
                'avg_rmse': r.get('avg_rmse', float('inf')),
                'avg_smape': r.get('avg_smape', float('inf')),
                'avg_owa': r.get('avg_owa', float('inf')),
                'window_count': r.get('window_count', 0),
                'elapsed': r.get('elapsed', 0),
                'mases': r.get('mases', []),
                'maes': r.get('maes', []),
                'rmses': r.get('rmses', []),
                'smapes': r.get('smapes', []),
                'owas': r.get('owas', []),
            }

        # ★ 保存 JSON
        json_path = os.path.join(output_dir, "results.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        # ★ 保存详细 CSV（每个窗口每个模式的 MASE）
        detail_rows = []
        for mode, r in all_results.items():
            if not r.get('results'):
                continue
            for win_result in r['results']:
                if win_result.get('success', False):
                    detail_rows.append({
                        'window_id': win_result.get('window_id'),
                        'mode': mode,
                        'mase': win_result.get('mase', float('inf')),
                        'mae': win_result.get('mae', float('inf')),
                        'rmse': win_result.get('rmse', float('inf')),
                        'smape': win_result.get('smape', float('inf')),
                        'owa': win_result.get('owa', float('inf')),
                    })

        if detail_rows:
            detail_df = pd.DataFrame(detail_rows)
            detail_path = os.path.join(output_dir, "detailed_results.csv")
            detail_df.to_csv(detail_path, index=False)
            print(f"   📁 详细结果: {detail_path}")

        # ★ 保存报告
        report_path = os.path.join(output_dir, "report.txt")
        lines = []
        lines.append("=" * 100)
        lines.append("📊 SPLS Ablation 测试报告")
        lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"运行目录: {self.run_dir}")
        lines.append(f"基线: {baseline}")
        lines.append(f"测试集窗口数: {len(self.test_df)}")
        lines.append("=" * 100)

        # ★ 获取基线数据
        baseline_data = all_results.get(baseline, {})
        baseline_mase = baseline_data.get('avg_mase', float('inf'))
        baseline_valid = baseline_mase != float('inf') and not math.isnan(baseline_mase)

        no_rule_data = all_results.get('no_rule', {})
        no_rule_mase = no_rule_data.get('avg_mase', float('inf'))
        no_rule_valid = no_rule_mase != float('inf') and not math.isnan(no_rule_mase)

        lines.append(f"\n{'模式':<14} | {'MASE':<12} | {'MAE':<12} | {'RMSE':<12} | {'SMAPE':<12} | {'OWA':<12} | {'窗口':<6} | {'vs ' + baseline + '':<12} | {'vs no_rule':<12}")
        lines.append("-" * 120)

        best_mode = 'no_rule'
        best_mase = no_rule_mase if no_rule_valid else float('inf')

        for mode in ['no_rule'] + sorted([m for m in all_results.keys() if m != 'no_rule'], key=lambda x: int(x.split('_')[1])):
            r = all_results.get(mode, {})
            mase = r.get('avg_mase', float('inf'))
            mae = r.get('avg_mae', float('inf'))
            rmse = r.get('avg_rmse', float('inf'))
            smape = r.get('avg_smape', float('inf'))
            owa = r.get('avg_owa', float('inf'))
            count = r.get('window_count', 0)

            if mase != float('inf') and not math.isnan(mase) and mase < best_mase:
                best_mase = mase
                best_mode = mode

            mase_str = f"{mase:.6f}" if mase != float('inf') and not math.isnan(mase) else "N/A"
            mae_str = f"{mae:.6f}" if mae != float('inf') and not math.isnan(mae) else "N/A"
            rmse_str = f"{rmse:.6f}" if rmse != float('inf') and not math.isnan(rmse) else "N/A"
            smape_str = f"{smape:.6f}" if smape != float('inf') and not math.isnan(smape) else "N/A"
            owa_str = f"{owa:.6f}" if owa != float('inf') and not math.isnan(owa) else "N/A"

            imp_baseline = None
            if baseline_valid and mase != float('inf') and not math.isnan(mase):
                imp_baseline = (baseline_mase - mase) / baseline_mase * 100

            imp_no_rule = None
            if no_rule_valid and mase != float('inf') and not math.isnan(mase):
                imp_no_rule = (no_rule_mase - mase) / no_rule_mase * 100

            imp_baseline_str = f"{imp_baseline:>+6.2f}%" if imp_baseline is not None else "N/A"
            imp_no_rule_str = f"{imp_no_rule:>+6.2f}%" if imp_no_rule is not None else "N/A"

            if mode == baseline:
                imp_baseline_str = "  —   "

            lines.append(f"{mode:<14} | {mase_str:<12} | {mae_str:<12} | {rmse_str:<12} | {smape_str:<12} | {owa_str:<12} | {count:<6} | {imp_baseline_str:<12} | {imp_no_rule_str:<12}")

        lines.append("=" * 120)

        if best_mase != float('inf') and not math.isnan(best_mase):
            lines.append(f"\n🏆 最佳策略: {best_mode} (MASE={best_mase:.6f})")
        else:
            lines.append(f"\n⚠️ 无有效最佳策略")

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        print(f"\n📁 结果已保存: {output_dir}")
        print(f"   - results.json (详细数据)")
        print(f"   - report.txt (对比报告)")
        print(f"   - detailed_results.csv (每个窗口每个模式的详细结果)")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="快速测试各轮策略（增强版）")
    parser.add_argument('--resume', type=str, required=True,
                        help='运行目录（如 run_20260624_043426）')
    parser.add_argument('--config', type=str, default=None,
                        help='配置文件路径')
    parser.add_argument('--workers', type=int, default=10,
                        help='并行线程数（默认 10）')
    parser.add_argument('--start', type=str, default='no_rule',
                        help='起始模式：no_rule, round_1, round_2, ... (默认 no_rule)')
    parser.add_argument('--baseline', type=str, default='round_1',
                        help='基线模式：no_rule 或 round_1（默认 round_1）')
    parser.add_argument('--test-ratio', type=float, default=0.5,
                        help='从 B 子集抽取测试集的比例（默认 0.5，即 50%）')
    parser.add_argument('--no-cache', action='store_true',
                        help='禁用缓存，强制重新运行所有模式')

    args = parser.parse_args()

    if os.path.exists(args.resume):
        run_dir = args.resume
    else:
        run_dir = os.path.join("llog", args.resume)

    if not os.path.exists(run_dir):
        print(f"❌ 目录不存在: {run_dir}")
        return

    # ★ 验证基线参数
    if args.baseline not in ['no_rule', 'round_1']:
        print(f"⚠️ 基线 '{args.baseline}' 无效，使用默认 'round_1'")
        args.baseline = 'round_1'

    tester = QuickTester(run_dir, args.config, test_ratio=args.test_ratio)
    tester.test_workers = args.workers
    tester.run(start_from=args.start, baseline=args.baseline)


if __name__ == '__main__':
    main()