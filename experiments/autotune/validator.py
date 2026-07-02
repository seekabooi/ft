# experiments/autotune/validator.py
"""
Policy Evaluation Oracle - v6 强化学习版
★ 支持 RL 采样策略评估
★ 增强诊断功能 + 策略健康评估
★ 支持并行评估（多线程）
★ 预测缓存，避免重复计算
★ ★ 2026-06-26 评估时支持 distribution_model 参数（RL 采样）
★ ★ ★ 2026-07-01 增加 active_only 参数，支持只评估 ACTIVE 策略（过滤其他状态）
★ ★ ★ 2026-07-02 增加 no_rule 基线模式（空策略列表 → 季节性朴素预测）
★ ★ ★ 2026-07-02 增加 use_rl_sampling 显式控制，支持 no_llm 模式
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import hashlib
import random

from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from experiments.autotune.utils import ProgressLogger, load_window_data, compute_mase, extract_features
from experiments.autotune.skill_policy import SkillPolicy


class PolicyEvaluationOracle:
    """策略评估 Oracle - v6 强化学习版（含诊断 + 预测缓存 + 进度条 + RL 采样）"""

    def __init__(self, config: Dict, logger: ProgressLogger):
        self.config = config
        self.logger = logger
        self.output_dir = config.get('output_dir', 'storage/autotune_results')
        # 预测缓存：key = (policy_id, window_data_path, horizon, period)
        self._pred_cache = {}
        self._cache_lock = threading.Lock()
        # 缓存最大条目数（防止内存溢出）
        self._max_cache_size = 2000

        # ★★★ RL 配置 ★★★
        rl_cfg = config.get('rl', {})
        self.use_rl_sampling = rl_cfg.get('enabled', True)
        self.top_k_sampling = rl_cfg.get('top_k_sampling', 3)
        self.exploration = rl_cfg.get('exploration', 0.1)

    def _get_cache_key(self, policy_id: str, window_data_path: str, horizon: int, period: int) -> str:
        """生成缓存键"""
        return f"{policy_id}:{window_data_path}:{horizon}:{period}"

    def _get_prediction(self, policy: SkillPolicy, window_data_path: str,
                        horizon: int, period: int) -> Optional[np.ndarray]:
        """获取策略在指定窗口上的预测结果，优先从缓存读取"""
        cache_key = self._get_cache_key(policy.policy_id, window_data_path, horizon, period)

        with self._cache_lock:
            if cache_key in self._pred_cache:
                return self._pred_cache[cache_key]

        try:
            wdata = load_window_data(window_data_path)
            train = wdata['train']
            pred = policy.execute(train, horizon, period)
            if pred is not None:
                with self._cache_lock:
                    if len(self._pred_cache) < self._max_cache_size:
                        self._pred_cache[cache_key] = pred
                    else:
                        keys = list(self._pred_cache.keys())[:len(self._pred_cache) // 2]
                        for k in keys:
                            del self._pred_cache[k]
                        self._pred_cache[cache_key] = pred
                return pred
        except Exception:
            pass
        return None

    def evaluate(self, policies: List[SkillPolicy], dataset_name: str,
                 split: Optional[str] = None,
                 parallel: bool = True,
                 workers: int = 4,
                 distribution_model=None,
                 active_only: bool = False,
                 use_rl_sampling: Optional[bool] = None) -> Dict[str, Any]:
        """
        评估策略（含诊断），支持并行处理 + RL 采样

        Args:
            policies: 策略列表（若为空，则自动进入 no_rule 基线模式，使用季节性朴素预测）
            dataset_name: 数据集名称（仅用于日志）
            split: 数据划分
            parallel: 是否并行
            workers: 并行线程数
            distribution_model: 可选，用于 RL 采样评估（仅在 use_rl_sampling=True 时生效）
            active_only: 是否只评估 ACTIVE 策略（True=只保留 status=='ACTIVE'，False=评估全部策略）
            use_rl_sampling: 是否使用 RL 采样（若为 None，则使用配置中的默认值）
        """
        # 决定是否使用 RL 采样
        if use_rl_sampling is None:
            use_rl_sampling = self.use_rl_sampling

        # ★★★ 基线模式：policies 为空 → no_rule ★★★
        if not policies:
            self.logger.log("🔍 no_rule 模式：使用季节性朴素预测作为基线")
            return self._evaluate_baseline(dataset_name, split, parallel, workers)

        # ★★★ 根据 active_only 过滤策略 ★★★
        original_count = len(policies)
        if active_only:
            policies = [p for p in policies if p.status == 'ACTIVE']
            if not policies:
                self.logger.log("⚠️ active_only=True 且没有 ACTIVE 策略，返回空报告")
                return self._empty_report()
            if original_count != len(policies):
                self.logger.log(f"🔍 active_only=True，过滤掉 {original_count - len(policies)} 条非 ACTIVE 策略，保留 {len(policies)} 条 ACTIVE 策略")
        else:
            self.logger.log(f"🔍 评估全部 {len(policies)} 条策略（ALL 模式）")

        # 记录当前模式
        mode_label = "rule (RL采样)" if use_rl_sampling else "no_llm (无RL采样)"
        self.logger.log(f"🔍 使用模式: {mode_label}")

        csv_path = os.path.join(self.output_dir, "collected_windows.csv")
        if not os.path.exists(csv_path):
            return self._empty_report()

        df = pd.read_csv(csv_path)
        if split is not None:
            df = df[df['split'] == split]
            self.logger.log(f"🔍 评估 split='{split}'，共 {len(df)} 个窗口")

        if len(df) == 0 or not policies:
            return self._empty_report()

        if not parallel or len(df) < 2:
            return self._evaluate_serial(policies, df, distribution_model, use_rl_sampling)
        else:
            self.logger.log(f"   ⚡ 并行评估 {len(df)} 个窗口 (workers={workers})")
            return self._evaluate_parallel(policies, df, workers, distribution_model, use_rl_sampling)

    # ---------- 基线评估（no_rule） ----------
    def _evaluate_baseline(self, dataset_name: str, split: Optional[str],
                           parallel: bool, workers: int) -> Dict:
        """使用季节性朴素预测作为基线，不依赖任何策略"""
        csv_path = os.path.join(self.output_dir, "collected_windows.csv")
        if not os.path.exists(csv_path):
            return self._empty_report()

        df = pd.read_csv(csv_path)
        if split is not None:
            df = df[df['split'] == split]
        if len(df) == 0:
            return self._empty_report()

        window_results = []
        mases = []
        no_rule_mases = []  # 基线本身也作为 no_rule

        def process_baseline(row):
            try:
                wpath = row.get('window_data_path')
                if not wpath or not os.path.exists(wpath):
                    return None
                wdata = load_window_data(wpath)
                train = wdata['train']
                test = wdata['test']
                period = wdata.get('period', 365)
                mase_scale = wdata.get('mase_scale', 1.0)
                horizon = wdata.get('horizon', 7)

                # 季节性朴素预测：用最后一个周期的对应值
                # 简单实现：用最后 period 个值循环
                if len(train) >= period:
                    pred = np.array([train[-period + (i % period)] for i in range(horizon)])
                else:
                    # 若数据不足，用最后一个值
                    pred = np.full(horizon, train[-1])
                mase = compute_mase(pred, test, mase_scale)
                return {
                    'window_id': row.get('window_id'),
                    'mase': mase,
                    'no_rule_mase': mase,  # 基线本身就是 no_rule
                    'improvement': 0.0,
                    'policy_id': 'baseline',
                    'policy_name': 'SeasonalNaive',
                    'applicability_score': 0.0
                }
            except Exception:
                return None

        if parallel and len(df) > 1:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(process_baseline, row) for _, row in df.iterrows()]
                for future in tqdm(futures, desc="   🔍 基线评估进度", unit="窗口"):
                    result = future.result()
                    if result is not None:
                        window_results.append(result)
                        mases.append(result['mase'])
                        no_rule_mases.append(result['mase'])
        else:
            for _, row in tqdm(df.iterrows(), desc="   🔍 基线评估进度", unit="窗口"):
                result = process_baseline(row)
                if result is not None:
                    window_results.append(result)
                    mases.append(result['mase'])
                    no_rule_mases.append(result['mase'])

        if not mases:
            return self._empty_report()

        avg_mase = np.mean(mases)
        avg_no_rule = avg_mase
        improvement = 0.0
        std_mase = np.std(mases) if len(mases) > 1 else 0.0

        report = {
            'avg_mase': avg_mase,
            'avg_no_rule': avg_no_rule,
            'improvement_score': 0.0,
            'stability_score': 1.0 / (1.0 + std_mase / (avg_mase + 0.001)) if avg_mase > 0 else 0,
            'total_windows': len(mases),
            'worst_3': sorted(window_results, key=lambda x: x['mase'], reverse=True)[:3],
            'window_results': window_results,
            'mases': mases,
            'policy_usage': {'baseline': len(mases)},
            'policy_scores': {}
        }
        self.logger.log(f"✅ 基线评估: avg_mase={avg_mase:.4f}")
        return report

    # ---------- 串行评估 ----------
    def _evaluate_serial(self, policies: List[SkillPolicy], df: pd.DataFrame,
                         distribution_model, use_rl_sampling: bool) -> Dict:
        """串行评估"""
        window_results = []
        mases = []
        no_rule_mases = []
        matched_policy_ids = []
        policy_scores = defaultdict(list)

        for _, row in df.iterrows():
            window_data_path = row.get('window_data_path')
            if not window_data_path or not os.path.exists(window_data_path):
                continue

            try:
                wdata = load_window_data(window_data_path)
                train = wdata['train']
                test = wdata['test']
                period = wdata.get('period', 365)
                mase_scale = wdata.get('mase_scale', 1.0)
                horizon = wdata.get('horizon', 7)

                features = extract_features(train)

                scored = []
                for policy in policies:
                    if policy.status in ['ARCHIVE', 'DELETE']:
                        continue
                    score = policy.compute_applicability_score(features)
                    scored.append((policy, score))
                    policy_scores[policy.policy_id].append(score)

                if not scored:
                    continue

                # 根据 use_rl_sampling 选择策略
                if use_rl_sampling and distribution_model is not None:
                    dist = distribution_model.get_distribution(features, policies)
                    policy_ids = list(dist.keys())
                    probs = list(dist.values())
                    if random.random() < self.exploration:
                        sampled_id = random.choice(policy_ids)
                    else:
                        sampled_id = random.choices(policy_ids, weights=probs, k=1)[0]
                    best_policy = next(p for p in policies if p.policy_id == sampled_id)
                    best_score = dist.get(sampled_id, 0.0)
                else:
                    # no_llm：直接取最高适用性分数
                    best_policy, best_score = max(scored, key=lambda x: x[1])

                if best_policy and best_score > 0.3:
                    pred = self._get_prediction(best_policy, window_data_path, horizon, period)
                    if pred is not None and len(pred) == len(test):
                        mase = compute_mase(pred, test, mase_scale)
                        mases.append(mase)
                        no_rule_mase = row.get('best_mase', mase)
                        no_rule_mases.append(no_rule_mase)
                        matched_policy_ids.append(best_policy.policy_id)

                        window_results.append({
                            'window_id': row.get('window_id'),
                            'mase': mase,
                            'no_rule_mase': no_rule_mase,
                            'improvement': (no_rule_mase - mase) / no_rule_mase if no_rule_mase > 0 else 0,
                            'policy_id': best_policy.policy_id,
                            'policy_name': best_policy.name,
                            'applicability_score': best_score
                        })
            except Exception:
                continue

        return self._build_report(window_results, mases, no_rule_mases, matched_policy_ids, policy_scores)

    # ---------- 并行评估 ----------
    def _evaluate_parallel(self, policies: List[SkillPolicy], df: pd.DataFrame,
                           workers: int, distribution_model, use_rl_sampling: bool) -> Dict:
        """并行评估"""
        tasks = []
        for idx, row in df.iterrows():
            window_data_path = row.get('window_data_path')
            if not window_data_path or not os.path.exists(window_data_path):
                continue
            tasks.append((idx, row, window_data_path))

        results_lock = threading.Lock()
        window_results = []
        mases = []
        no_rule_mases = []
        matched_policy_ids = []
        policy_scores = defaultdict(list)

        def process_one(task):
            idx, row, window_data_path = task
            try:
                wdata = load_window_data(window_data_path)
                train = wdata['train']
                test = wdata['test']
                period = wdata.get('period', 365)
                mase_scale = wdata.get('mase_scale', 1.0)
                horizon = wdata.get('horizon', 7)

                features = extract_features(train)

                scored = []
                local_scores = defaultdict(list)
                for policy in policies:
                    if policy.status in ['ARCHIVE', 'DELETE']:
                        continue
                    score = policy.compute_applicability_score(features)
                    scored.append((policy, score))
                    local_scores[policy.policy_id].append(score)

                if not scored:
                    return None

                if use_rl_sampling and distribution_model is not None:
                    dist = distribution_model.get_distribution(features, policies)
                    policy_ids = list(dist.keys())
                    probs = list(dist.values())
                    if random.random() < self.exploration:
                        sampled_id = random.choice(policy_ids)
                    else:
                        sampled_id = random.choices(policy_ids, weights=probs, k=1)[0]
                    best_policy = next(p for p in policies if p.policy_id == sampled_id)
                    best_score = dist.get(sampled_id, 0.0)
                else:
                    best_policy, best_score = max(scored, key=lambda x: x[1])

                if not best_policy or best_score <= 0.3:
                    return None

                pred = self._get_prediction(best_policy, window_data_path, horizon, period)
                if pred is None or len(pred) != len(test):
                    return None

                mase = compute_mase(pred, test, mase_scale)
                no_rule_mase = row.get('best_mase', mase)

                result = {
                    'window_id': row.get('window_id'),
                    'mase': mase,
                    'no_rule_mase': no_rule_mase,
                    'improvement': (no_rule_mase - mase) / no_rule_mase if no_rule_mase > 0 else 0,
                    'policy_id': best_policy.policy_id,
                    'policy_name': best_policy.name,
                    'applicability_score': best_score,
                    'policy_scores': local_scores
                }
                return result
            except Exception:
                return None

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_one, task): task for task in tasks}
            total_tasks = len(futures)

            with tqdm(total=total_tasks, desc="   🔍 评估进度", unit="窗口", ncols=100) as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        with results_lock:
                            window_results.append({
                                'window_id': result['window_id'],
                                'mase': result['mase'],
                                'no_rule_mase': result['no_rule_mase'],
                                'improvement': result['improvement'],
                                'policy_id': result['policy_id'],
                                'policy_name': result['policy_name'],
                                'applicability_score': result['applicability_score']
                            })
                            mases.append(result['mase'])
                            no_rule_mases.append(result['no_rule_mase'])
                            matched_policy_ids.append(result['policy_id'])
                            for pid, scores in result['policy_scores'].items():
                                policy_scores[pid].extend(scores)
                    pbar.update(1)

        self.logger.log(f"   ✅ 并行评估完成，有效窗口: {len(window_results)}/{len(tasks)}")
        return self._build_report(window_results, mases, no_rule_mases, matched_policy_ids, policy_scores)

    # ---------- 报告构建 ----------
    def _build_report(self, window_results, mases, no_rule_mases, matched_policy_ids, policy_scores):
        if not mases:
            return self._empty_report()

        avg_mase = np.mean(mases)
        avg_no_rule = np.mean(no_rule_mases) if no_rule_mases else avg_mase
        std_mase = np.std(mases)
        improvement = (avg_no_rule - avg_mase) / avg_no_rule if avg_no_rule > 0 else 0

        policy_usage = {}
        for pid in matched_policy_ids:
            policy_usage[pid] = policy_usage.get(pid, 0) + 1

        report = {
            'avg_mase': avg_mase,
            'avg_no_rule': avg_no_rule,
            'improvement_score': improvement,
            'stability_score': 1.0 / (1.0 + std_mase / (avg_mase + 0.001)) if avg_mase > 0 else 0,
            'total_windows': len(mases),
            'worst_3': sorted(window_results, key=lambda x: x['mase'], reverse=True)[:3],
            'window_results': window_results,
            'mases': mases,
            'policy_usage': policy_usage,
            'policy_scores': dict(policy_scores)
        }

        self.logger.log(f"✅ 评估: avg_mase={avg_mase:.4f}, 改善={improvement * 100:.2f}%")
        return report

    def _empty_report(self) -> Dict:
        return {
            'avg_mase': float('inf'),
            'avg_no_rule': float('inf'),
            'improvement_score': 0.0,
            'stability_score': 0.0,
            'total_windows': 0,
            'worst_3': [],
            'window_results': [],
            'mases': [],
            'policy_usage': {},
            'policy_scores': {}
        }