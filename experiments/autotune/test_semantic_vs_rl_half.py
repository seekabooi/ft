#!/usr/bin/env python
"""
语义匹配 vs RL 参数消融实验（半窗口版）- 无注解/无描述版

仅使用测试集的前半或后半 25 个窗口。
★ 所有模式（no_rule, semantic_top1, semantic_top30, semantic_top60）重新计算
★ 不添加特征注解，不添加技能描述（LLM 仅凭技能名称和原始特征决策）
★ 输出目录独立（llog/cs2_half_no_desc），不影响旧结果
★ 测试完成后自动生成折线图和各指标对比表格

用法：
    python -m experiments.autotune.test_semantic_vs_rl_half_no_desc \
        --resume llog/cs2 \
        --round 57 \
        --half-mode first \
        --workers 12

参数：
    --resume      原始运行目录（如 llog/cs2）
    --round       指定轮次（如 57）
    --half-mode   选择测试集位置：first（前25个窗口）或 second（后25个窗口）
    --workers     并行线程数
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
import re
import shutil

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 复用现有组件
from experiments.autotune.utils import (
    load_config, load_window_data, compute_all_metrics, extract_features
)
from experiments.autotune.skill_policy import SkillPolicy
from experiments.autotune.state_encoder import StateEncoder
from experiments.autotune.prompts import build_strategy_generation_prompt
from experiments.autotune.inducer_candidate import _safe_extract_json, _extract_strategies_from_text
from src.agents.llm_client import LLMClient
from src.agents.llm_planner import LLMPlannerAgent
from src.tasks.instance import TaskInstance
from run_benchmark import build_full_registry

WINDOW_TIMEOUT = 120

# ★★★ 全局信号量：控制并发 LLM 请求数（避免 429 速率限制） ★★★
_LLM_SEMAPHORE = threading.Semaphore(8)


class SemanticVsRLTesterHalfNoDesc:
    """语义匹配 vs RL 参数消融实验（半窗口版）- 无注解无描述"""

    def __init__(self, run_dir: str, round_num: int, half_mode: str = 'first',
                 config_path: str = None, test_ratio: float = 0.5, workers: int = 12):
        self.run_dir = run_dir
        self.round_num = round_num
        self.half_mode = half_mode
        self.config = load_config(config_path)
        self.output_dir = self.config.get('output_dir', 'storage/autotune_results')
        self.test_workers = workers
        self.test_ratio = test_ratio
        self._timeout_counter = 0

        # ★★★ 新输出目录：在原目录下加 _half_no_desc ★★★
        self.run_dir_out = run_dir + "_half_no_desc"
        os.makedirs(self.run_dir_out, exist_ok=True)

        # 构建技能注册表
        print("   🔧 构建技能注册表...")
        self.full_registry, self.all_skills = build_full_registry()
        self.skill_names = [s.name for s in self.all_skills]
        blacklist = set(self.config.get('skill_filter', {}).get('blacklist', []))
        self.skill_names = [s for s in self.skill_names if s not in blacklist]
        print(f"   ✅ 可用技能数: {len(self.skill_names)}")

        self.state_encoder = StateEncoder(self.config)
        self.model = self._detect_model()
        self.test_df = self._load_test_df()
        self.policies = self._load_round_policies(round_num)
        self._llm_client_cache = {}
        self._lock = threading.Lock()

        # ★★★ 测试模式 ★★★
        self.modes = [
            'no_rule',
            'semantic_top1',
            'semantic_top30_theta_max',
            'semantic_top60_theta_max',
        ]

        self._formatter_agent = LLMPlannerAgent(
            model=self.model if self.model else "glm-4",
            skill_registry=self.full_registry,
            verbose=False,
            use_skills=True
        )

        self.log_file_path = os.path.join(self.run_dir_out, "semantic_vs_rl_detailed.log")
        self._log_file = open(self.log_file_path, 'a', encoding='utf-8')

        self._print_big(f"🚀 启动测试（无注解无描述，{half_mode} 25 个窗口）")

    def _print_big(self, msg: str):
        border = "=" * 80
        print("\n" + border)
        print(" " * 10 + msg)
        print(border + "\n")
        self._log(msg)

    def _log(self, msg: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full_msg = f"[{timestamp}] {msg}"
        print(full_msg)
        self._log_file.write(full_msg + '\n')
        self._log_file.flush()

    def _load_test_df(self) -> pd.DataFrame:
        """加载测试集，并根据 half_mode 取前 25 或后 25 个窗口"""
        csv_path = os.path.join(self.output_dir, "collected_windows.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"❌ 未找到采集数据: {csv_path}")

        df = pd.read_csv(csv_path)

        if 'split' in df.columns:
            test_df = df[df['split'] == 'test'].copy()
        else:
            n = len(df)
            a_end = int(n * 0.5)
            b_mask = pd.Series([False] * n)
            b_mask.iloc[a_end:] = True
            df['split'] = ['A'] * a_end + ['B'] * (n - a_end)
            b_df = df[b_mask].copy().sort_values('window_id').reset_index(drop=True)
            n_b = len(b_df)
            test_size = int(n_b * self.test_ratio)
            test_df = b_df.iloc[:test_size].copy()

        # 按 window_id 排序
        test_df = test_df.sort_values('window_id').reset_index(drop=True)

        if self.half_mode == 'first':
            test_df = test_df.head(25).copy()
        elif self.half_mode == 'second':
            test_df = test_df.tail(25).copy()
        else:
            raise ValueError(f"half_mode 必须为 'first' 或 'second'，得到 {self.half_mode}")

        print(f"📊 使用测试集 {'前' if self.half_mode=='first' else '后'} 25 个窗口（共 {len(test_df)} 个）")
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
        models = ["glm-4"]
        for model in models:
            try:
                client = LLMClient(model=model, verbose=False)
                resp = client.call_with_retry("请回复'OK'", max_retries=1)
                if resp and resp.choices and resp.choices[0].message.content:
                    print(f"   ✅ 使用模型: {model}")
                    return model
            except:
                continue
        models = ["glm-4.5-air", "glm-4.7"]
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

    def _get_llm_client(self, thread_id: int) -> LLMClient:
        if thread_id not in self._llm_client_cache:
            self._llm_client_cache[thread_id] = LLMClient(
                model=self.model if self.model else "glm-4",
                verbose=False
            )
        return self._llm_client_cache[thread_id]

    def _format_reference_strategy(self, policy: SkillPolicy) -> str:
        if policy is None:
            return "无"
        return self._formatter_agent._format_strategy(policy.skill_strategy)

    def _generate_strategy_from_llm(self, features: Dict, horizon: int,
                                   reference_policy: Optional[SkillPolicy] = None,
                                   thread_id: int = 0,
                                   window_id: int = None) -> Optional[Dict]:
        with _LLM_SEMAPHORE:
            # ★★★ 不使用特征注解，直接用原始特征（无中文解释） ★★★
            base_prompt = build_strategy_generation_prompt(
                features=features,
                trajectory=[],
                window_id=window_id if window_id is not None else 0,
                horizon=horizon
            )
            prompt = base_prompt

            # 追加技能列表（仅名称，无描述）
            skill_list_str = ', '.join(self.skill_names)
            prompt += f"\n\n★★★★★ 可用技能列表（必须从以下名称中选择，不得使用列表外的任何名称）：\n{skill_list_str}\n"
            prompt += "\n⚠️ 请只生成一个候选策略（candidate_strategies 数组只包含一个对象）。\n"

            # 如果有参考策略，追加参考信息（但不再强调特征注解，因为没有注解）
            if reference_policy is not None:
                ref_desc = self._format_reference_strategy(reference_policy)
                prompt += (
                    "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    "📌 参考策略（供参考，可借鉴其思路，但鼓励创新）：\n"
                    f"{ref_desc}\n"
                    "   请基于当前窗口的原始特征值独立判断，不要照搬参考策略。\n"
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                )

            client = self._get_llm_client(thread_id)
            try:
                self._log(f"      ⏳ [窗口 {window_id}] 向 LLM 发送请求...")
                resp = client.call_with_retry(prompt, max_retries=2)
                content = resp.choices[0].message.content
                self._log(f"      ✅ [窗口 {window_id}] LLM 响应完成 (长度: {len(content)})")

                data = _safe_extract_json(content)
                strategies = data.get('candidate_strategies', [])
                if strategies and len(strategies) > 0:
                    strategy = strategies[0]
                    if strategy.get('stages'):
                        self._log(f"      📋 [窗口 {window_id}] 解析到策略: {strategy.get('name', '未命名')} (阶段数: {len(strategy['stages'])})")
                        return strategy

                strategies = _extract_strategies_from_text(content)
                if strategies and len(strategies) > 0:
                    self._log(f"      📋 [窗口 {window_id}] 正则解析到策略: {strategies[0].get('name', '未命名')}")
                    return strategies[0]

                self._log(f"      ⚠️ [窗口 {window_id}] 解析策略失败")
                return None
            except Exception as e:
                self._log(f"      ❌ [窗口 {window_id}] LLM 策略生成失败: {e}")
                return None

    def _execute_strategy(self, strategy: Dict, train: np.ndarray,
                          horizon: int, period: int, window_id: int = None) -> Optional[np.ndarray]:
        try:
            from experiments.autotune.skill_policy import SkillPolicy
            import hashlib, time
            temp_policy = SkillPolicy(
                policy_id=hashlib.md5(f"temp_{time.time()}".encode()).hexdigest()[:8],
                name="temp_policy",
                skill_strategy=strategy,
                avg_mase=1.0
            )
            pred = temp_policy.execute(train, horizon, period)
            if pred is not None and len(pred) == horizon:
                return pred
            self._log(f"      ⚠️ [窗口 {window_id}] 策略执行返回无效预测")
            return None
        except Exception as e:
            self._log(f"      ⚠️ [窗口 {window_id}] 策略执行失败: {e}")
            return None

    def _predict_with_mode(self, mode: str, train: np.ndarray, horizon: int,
                           period: int, thread_id: int = 0,
                           window_id: int = None) -> Optional[np.ndarray]:
        features = extract_features(train)
        self._log(f"   📊 [窗口 {window_id}] 特征: trend={features.get('trend_strength',0):.3f}, season={features.get('seasonal_strength',0):.3f}, cv={features.get('cv',0):.3f}")

        # no_rule 无参考
        if mode == 'no_rule':
            self._log(f"   🎯 [窗口 {window_id}] 模式: no_rule (无参考)")
            strategy = self._generate_strategy_from_llm(features, horizon, None, thread_id, window_id)
            if strategy is None:
                self._log(f"   ❌ [窗口 {window_id}] 策略生成失败，使用均值回退")
                return None
            return self._execute_strategy(strategy, train, horizon, period, window_id)

        # 有参考模式
        scored = []
        for policy in self.policies:
            if policy.status in ['ARCHIVE', 'DELETE']:
                continue
            score = policy.compute_applicability_score(features)
            scored.append((policy, score))

        if not scored:
            self._log(f"   ⚠️ [窗口 {window_id}] 无可用策略计算语义，使用均值回退")
            return None

        scored.sort(key=lambda x: x[1], reverse=True)

        if mode == 'semantic_top1':
            k = 1
        elif mode == 'semantic_top30_theta_max':
            k = max(1, int(len(scored) * 0.30))
        elif mode == 'semantic_top60_theta_max':
            k = max(1, int(len(scored) * 0.60))
        else:
            k = 1

        candidate_pool = scored[:k]
        ref_policy = max(candidate_pool, key=lambda x: x[0].logit_weight)[0]
        ref_score = max(candidate_pool, key=lambda x: x[0].logit_weight)[1]
        ref_theta = ref_policy.logit_weight

        self._log(f"   🎯 [窗口 {window_id}] 模式: {mode} | 候选池大小: {k}/{len(scored)} | 参考策略: {ref_policy.name[:20]} (语义={ref_score:.3f}, θ={ref_theta:.3f})")

        strategy = self._generate_strategy_from_llm(features, horizon, ref_policy, thread_id, window_id)
        if strategy is None:
            self._log(f"   ❌ [窗口 {window_id}] 策略生成失败，使用均值回退")
            return None

        return self._execute_strategy(strategy, train, horizon, period, window_id)

    def evaluate_single_window(self, idx: int, row: pd.Series,
                               mode: str, thread_id: int = 0) -> Dict:
        window_id = row.get('window_id', 'unknown')
        window_data_path = row.get('window_data_path')

        self._log(f"\n{'='*60}")
        self._log(f"🔹 窗口 {window_id} | 模式: {mode} | 线程: {thread_id}")
        self._log(f"{'='*60}")

        if not window_data_path or not os.path.exists(window_data_path):
            self._log(f"   ❌ 数据路径不存在: {window_data_path}")
            return {'window_id': window_id, 'success': False, 'error': '路径不存在'}

        try:
            wdata = load_window_data(window_data_path)
            train = wdata['train']
            test = wdata['test']
            period = wdata.get('period', 365)
            mase_scale = wdata.get('mase_scale', 1.0)
            horizon = wdata.get('horizon', 7)

            self._log(f"   📈 训练长度: {len(train)}, 测试长度: {len(test)}, horizon: {horizon}, period: {period}")

            pred = self._predict_with_mode(mode, train, horizon, period, thread_id, window_id)

            if pred is None or len(pred) != len(test):
                self._log(f"   ⚠️ 预测无效，使用均值回退")
                pred = np.full(len(test), np.mean(train))

            metrics = compute_all_metrics(pred, test, mase_scale)
            mase = metrics.get('mase', float('inf'))
            self._log(f"   ✅ 完成: MASE={mase:.6f}")

            return {
                'window_id': window_id,
                'success': True,
                'mase': mase,
                'mae': metrics.get('mae', float('inf')),
                'rmse': metrics.get('rmse', float('inf')),
                'smape': metrics.get('smape', float('inf')),
                'owa': metrics.get('owa', float('inf')),
            }

        except Exception as e:
            self._log(f"   ❌ 异常: {e}")
            return {'window_id': window_id, 'success': False, 'error': str(e)}

    def evaluate_mode(self, mode: str) -> Dict:
        tasks = [(idx, row) for idx, row in self.test_df.iterrows()]
        total = len(tasks)

        self._print_big(f"🚀 开始评估模式: {mode} (共 {total} 个窗口)")

        self._log(f"\n{'='*80}")
        self._log(f"📊 开始评估模式: {mode} (共 {total} 个窗口)")
        self._log(f"{'='*80}")

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
                desc=f"   {mode} 进度",
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
                        pbar.set_postfix({
                            'MASE': f"{avg_mase:.4f}" if avg_mase != float('inf') else '...',
                            '完成': len(results)
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

        self._log(f"\n📊 模式 {mode} 完成: 有效窗口 {valid_count}/{total}")

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

    def _save_intermediate_results(self, all_results: Dict):
        output_dir = os.path.join(self.run_dir_out, "semantic_vs_rl_results")
        os.makedirs(output_dir, exist_ok=True)
        json_path = os.path.join(output_dir, "results.json")

        if os.path.exists(json_path):
            backup_path = os.path.join(output_dir, "results_backup.json")
            try:
                shutil.copy2(json_path, backup_path)
            except:
                pass

        existing = {}
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    existing = json.load(f)
            except:
                pass

        for mode, r in all_results.items():
            if r.get('success', False):
                existing[mode] = {
                    'avg_mase': r.get('avg_mase', float('inf')),
                    'avg_mae': r.get('avg_mae', float('inf')),
                    'avg_rmse': r.get('avg_rmse', float('inf')),
                    'avg_smape': r.get('avg_smape', float('inf')),
                    'avg_owa': r.get('avg_owa', float('inf')),
                    'window_count': r.get('window_count', 0)
                }

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)

    def run(self):
        self._log("\n" + "=" * 80)
        self._log("🧪 语义匹配 vs RL 参数消融实验（半窗口无注解版）")
        self._log(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._log(f"📁 原始运行目录: {self.run_dir}")
        self._log(f"📁 输出目录: {self.run_dir_out}")
        self._log(f"🔢 指定轮次: round_{self.round_num}")
        self._log(f"📋 策略总数: {len(self.policies)}")
        self._log(f"📊 测试窗口数: {len(self.test_df)}")
        self._log(f"⚡ 并行线程: {self.test_workers}")
        self._log("🔧 执行模式: 策略生成 → 本地执行（与训练一致）")
        self._log("")
        self._log("★ ★★ 无特征注解，无技能描述，LLM 仅凭技能名称和原始特征决策")
        self._log("★ ★★ 所有模式重新计算（无缓存）")
        self._log("=" * 80)

        if not self.policies:
            self._log("❌ 没有策略，退出")
            return

        all_results = {}
        total_start = time.time()

        for mode in self.modes:
            self._log(f"\n   🔄 开始计算: {mode}")
            start = time.time()
            result = self.evaluate_mode(mode)
            result['elapsed'] = time.time() - start
            all_results[mode] = result
            self._save_intermediate_results(all_results)

        total_elapsed = time.time() - total_start

        self._generate_comparison_report(all_results, total_elapsed)
        self._generate_window_line_chart(all_results)
        self._save_final_results(all_results)

        self._print_big("✅ 所有测试完成！结果已保存至 " + self.run_dir_out)

    def _generate_comparison_report(self, all_results: Dict, total_elapsed: float):
        output_dir = os.path.join(self.run_dir_out, "semantic_vs_rl_results")
        os.makedirs(output_dir, exist_ok=True)

        rows = []
        for mode in self.modes:
            data = all_results.get(mode, {})
            if data.get('success', False):
                rows.append({
                    'mode': mode,
                    'avg_mase': data.get('avg_mase', float('inf')),
                    'avg_mae': data.get('avg_mae', float('inf')),
                    'avg_rmse': data.get('avg_rmse', float('inf')),
                    'avg_smape': data.get('avg_smape', float('inf')),
                    'avg_owa': data.get('avg_owa', float('inf')),
                    'window_count': data.get('window_count', 0)
                })

        if not rows:
            self._log("⚠️ 无数据生成报告")
            return

        # 生成文本报告
        report_lines = []
        report_lines.append("=" * 120)
        report_lines.append("📊 语义匹配 vs RL 参数消融实验报告（半窗口无注解版）")
        report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"运行目录: {self.run_dir_out}")
        report_lines.append("=" * 120)
        report_lines.append("")
        report_lines.append("模式说明:")
        report_lines.append("  - no_rule: 无参考策略，无特征注解")
        report_lines.append("  - semantic_top1: 语义最匹配的策略作为参考")
        report_lines.append("  - semantic_top30_theta_max: 语义前30%中θ最大的策略作为参考")
        report_lines.append("  - semantic_top60_theta_max: 语义前60%中θ最大的策略作为参考")
        report_lines.append("")
        report_lines.append("★ 无特征注解，无技能描述，LLM 仅凭原始特征和技能名称决策")
        report_lines.append("")
        report_lines.append(f"{'模式':<28} | {'窗口数':<8} | {'MASE':<12} | {'MAE':<12} | {'RMSE':<12} | {'SMAPE':<12} | {'OWA':<12}")
        report_lines.append("-" * 140)

        for r in rows:
            mase = r['avg_mase']
            if mase == float('inf') or math.isnan(mase):
                continue
            report_lines.append(
                f"{r['mode']:<28} | {r['window_count']:<8} | {mase:<12.6f} | {r['avg_mae']:<12.6f} | "
                f"{r['avg_rmse']:<12.6f} | {r['avg_smape']:<12.6f} | {r['avg_owa']:<12.6f}"
            )

        report_lines.append("-" * 140)
        report_lines.append(f"\n总耗时: {total_elapsed:.1f}s ({total_elapsed/60:.1f}分钟)")

        report_path = os.path.join(output_dir, "comparison_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        self._log(f"📄 对比报告已保存: {report_path}")

        self._plot_bar_chart(rows, output_dir)

    def _plot_bar_chart(self, rows: List[Dict], output_dir: str):
        modes = [r['mode'] for r in rows]
        mases = [r['avg_mase'] for r in rows if r['avg_mase'] != float('inf')]

        if not modes or not mases:
            return

        colors = ['#808080', '#2E86AB', '#F5A623', '#E68A2E']
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(modes, mases, color=colors[:len(modes)], alpha=0.7, edgecolor='black', linewidth=1.5)

        for bar, mase in zip(bars, mases):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{mase:.4f}', ha='center', va='bottom', fontsize=9)

        ax.set_ylabel('MASE', fontsize=12)
        ax.set_title('各模式平均 MASE 对比（半窗口无注解版）', fontsize=14)
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=15, ha='right')
        plt.tight_layout()

        bar_path = os.path.join(output_dir, "comparison_bar.png")
        plt.savefig(bar_path, dpi=150, bbox_inches='tight')
        plt.close()
        self._log(f"📊 柱状图已保存: {bar_path}")

    def _generate_window_line_chart(self, all_results: Dict):
        output_dir = os.path.join(self.run_dir_out, "semantic_vs_rl_results")
        os.makedirs(output_dir, exist_ok=True)

        mode_window_data = {}
        for mode, data in all_results.items():
            if data.get('success', False) and 'results' in data:
                window_mases = {}
                for res in data['results']:
                    wid = res.get('window_id')
                    mase = res.get('mase')
                    if wid is not None and mase is not None and mase != float('inf'):
                        window_mases[int(wid)] = mase
                if window_mases:
                    mode_window_data[mode] = window_mases

        if not mode_window_data:
            self._log("⚠️ 无窗口级数据生成折线图")
            return

        all_wids = set()
        for wdata in mode_window_data.values():
            all_wids.update(wdata.keys())
        all_wids = sorted(all_wids)

        colors = {
            'no_rule': '#808080',
            'semantic_top1': '#2E86AB',
            'semantic_top30_theta_max': '#F5A623',
            'semantic_top60_theta_max': '#E68A2E',
        }
        display_names = {
            'no_rule': 'no_rule',
            'semantic_top1': 'semantic_top1',
            'semantic_top30_theta_max': 'top30% + θ_max',
            'semantic_top60_theta_max': 'top60% + θ_max',
        }

        fig, ax = plt.subplots(figsize=(14, 7))
        for mode, wdata in mode_window_data.items():
            sorted_items = sorted(wdata.items())
            wids = [w[0] for w in sorted_items]
            mases = [w[1] for w in sorted_items]
            color = colors.get(mode, '#000000')
            label = display_names.get(mode, mode)
            ax.plot(wids, mases, marker='o', color=color, linewidth=2, markersize=4, label=label)

        ax.set_xlabel('窗口ID', fontsize=12)
        ax.set_ylabel('MASE', fontsize=12)
        ax.set_title('各模式窗口 MASE 对比折线图（半窗口无注解版）', fontsize=14)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        line_path = os.path.join(output_dir, "window_comparison.png")
        plt.savefig(line_path, dpi=150, bbox_inches='tight')
        plt.close()
        self._log(f"📊 折线图已保存: {line_path}")

    def _save_final_results(self, all_results: Dict):
        output_dir = os.path.join(self.run_dir_out, "semantic_vs_rl_results")
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

        self._log(f"📁 最终结果已保存: {json_path}")


class Tee:
    def __init__(self, filename, mode='a'):
        self.file = open(filename, mode, encoding='utf-8')
        self.stdout = sys.stdout
        self.stderr = sys.stderr

    def write(self, message):
        self.file.write(message)
        self.file.flush()
        self.stdout.write(message)
        self.stdout.flush()

    def flush(self):
        self.file.flush()
        self.stdout.flush()

    def close(self):
        self.file.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="语义匹配 vs RL 参数消融实验（半窗口无注解版）")
    parser.add_argument('--resume', type=str, required=True,
                        help='原始运行目录（如 llog/cs2）')
    parser.add_argument('--round', type=int, required=True,
                        help='指定轮次（如 57）')
    parser.add_argument('--half-mode', type=str, default='first', choices=['first', 'second'],
                        help='选择测试集前25个窗口(first)还是后25个窗口(second)')
    parser.add_argument('--config', type=str, default=None,
                        help='配置文件路径')
    parser.add_argument('--workers', type=int, default=12,
                        help='并行线程数（默认 12）')
    parser.add_argument('--test-ratio', type=float, default=0.5,
                        help='测试集比例（默认 0.5）')

    args = parser.parse_args()

    if os.path.exists(args.resume):
        run_dir = args.resume
    else:
        run_dir = os.path.join("llog", args.resume)

    if not os.path.exists(run_dir):
        print(f"❌ 目录不存在: {run_dir}")
        return

    run_dir_out = run_dir + "_half_no_desc"
    os.makedirs(run_dir_out, exist_ok=True)

    log_file = os.path.join(run_dir_out, "semantic_vs_rl_full.log")
    tee = Tee(log_file, 'a')
    sys.stdout = tee
    sys.stderr = tee

    try:
        tester = SemanticVsRLTesterHalfNoDesc(
            run_dir=run_dir,
            round_num=args.round,
            half_mode=args.half_mode,
            config_path=args.config,
            test_ratio=args.test_ratio,
            workers=args.workers
        )
        tester.run()
    finally:
        sys.stdout = tee.stdout
        sys.stderr = tee.stderr
        tee.close()
        print(f"\n✅ 全量日志已保存至: {log_file}")


if __name__ == '__main__':
    main()