# experiments/autotune/tuner_eval.py
"""SPLS AutoTuner - 评估相关方法（Ablation + 测试集）
★ 测试时 Top‑2 集成投票
"""

import os
import time
import threading
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from experiments.autotune.utils import load_window_data, compute_all_metrics
from experiments.autotune.skill_policy import SkillPolicy
from experiments.autotune.tuner_utils import detect_available_model
from src.agents.llm_planner import LLMPlannerAgent
from src.tasks.instance import TaskInstance
from run_benchmark import build_full_registry


def run_ablation_parallel(logger, loop, policy_snapshots: Dict, num_rounds: int,
                           dataset_name: str, window_size: int,
                           horizon: int, test_df: pd.DataFrame,
                           test_parallel: bool, test_workers: int) -> Dict:
    """并行版 Ablation 测试（模式级并行）+ 每轮耗时统计 + ★★★ Top‑2 集成 ★★★"""
    full_registry, _ = build_full_registry()

    model_candidates = ["glm-4", "glm-4.5-air"]
    selected_model, error = detect_available_model(logger, model_candidates)

    if selected_model is None:
        logger.log(f"   ⚠️ 没有可用模型，使用均值回退")
        return run_ablation_fallback(test_df)

    logger.log(f"   ✅ 使用模型: {selected_model}")

    agent = LLMPlannerAgent(
        model=selected_model,
        skill_registry=full_registry,
        log_file=None,
        use_skills=True,
        rules_file=None,
        llm_call_interval=3,
        verbose=False
    )
    agent.rule_engine = None

    modes = ['no_rule']
    for round_num in range(1, num_rounds + 1):
        modes.append(f'round_{round_num}')

    # ★★★ 新增集成模式 ★★★
    modes.append('ensemble_top2')

    results = {mode: {'mases': [], 'metrics': []} for mode in modes}
    window_metrics = []

    mode_times = {mode: 0.0 for mode in modes}
    mode_counts = {mode: 0 for mode in modes}
    mode_times_lock = threading.Lock()

    total_windows = len(test_df)

    def process_window(idx, row):
        window_id = row.get('window_id', 'unknown')
        window_data_path = row.get('window_data_path')

        if not window_data_path or not os.path.exists(window_data_path):
            return None

        try:
            wdata = load_window_data(window_data_path)
            train = wdata['train']
            test = wdata['test']
            period = wdata.get('period', 365)
            mase_scale = wdata.get('mase_scale', 1.0)

            task = TaskInstance(
                id=f"ablation_{window_id}",
                dataset_id=dataset_name,
                template_id="fixed_origin",
                question="",
                question_type="numerical",
                history=train.tolist(),
                horizon=horizon,
                frequency="daily",
                prediction_target={},
                resolution_date=pd.Timestamp.now(),
                difficulty_level=1,
                ground_truth_extractor="",
                dates=None,
                target_date=""
            )

            window_metric = {'window_id': window_id}
            window_mases = {}
            window_mode_times = {}

            # 预计算所有轮次的策略排序（按适用性）
            round_policies = {}
            for mode in modes:
                if mode == 'no_rule':
                    continue
                elif mode == 'ensemble_top2':
                    continue
                else:
                    round_num = int(mode.split('_')[1])
                    policies = policy_snapshots.get(f'round_{round_num}', [])
                    if policies:
                        state = loop.state_encoder.encode(train)
                        numeric_state = state.get('numeric', {})
                        scored = []
                        for policy in policies:
                            if policy.status in ['ARCHIVE', 'DELETE']:
                                continue
                            score = policy.compute_applicability_score(numeric_state)
                            scored.append((policy, score))
                        scored.sort(key=lambda x: x[1], reverse=True)
                        round_policies[mode] = scored
                    else:
                        round_policies[mode] = []

            for mode in modes:
                start_time = time.time()

                if mode == 'no_rule':
                    agent._current_rule_strategy = None
                    try:
                        pred = agent.predict(task)
                    except:
                        pred = None
                elif mode == 'ensemble_top2':
                    # ★★★ Top‑2 集成投票 ★★★
                    # 遍历所有轮次，收集每个轮次的 top‑2 策略预测，然后加权平均
                    # 简单实现：取所有轮次中适用性最高的前两个策略（跨轮）
                    all_scored = []
                    for rnd in range(1, num_rounds + 1):
                        rmode = f'round_{rnd}'
                        if rmode in round_policies:
                            # 取该轮次的前2
                            top2 = round_policies[rmode][:2]
                            for policy, score in top2:
                                all_scored.append((policy, score, rmode))
                    # 全局排序，取前2
                    all_scored.sort(key=lambda x: x[1], reverse=True)
                    top2_policies = all_scored[:2]
                    if len(top2_policies) == 2:
                        # 加权平均（权重为 softmax 适用性分数）
                        scores = [s[1] for s in top2_policies]
                        exp_scores = np.exp(np.array(scores))
                        weights = exp_scores / np.sum(exp_scores)
                        preds = []
                        for (policy, _, _), w in zip(top2_policies, weights):
                            # 执行预测
                            period2 = period
                            pred = policy.execute(train, horizon, period2)
                            if pred is not None and len(pred) == horizon:
                                preds.append(pred * w)
                        if preds:
                            pred = np.sum(preds, axis=0)
                        else:
                            pred = np.full(horizon, np.mean(train))
                    else:
                        # 如果不足2个，回退到最佳策略
                        if all_scored:
                            best_policy = all_scored[0][0]
                            pred = best_policy.execute(train, horizon, period)
                            if pred is None:
                                pred = np.full(horizon, np.mean(train))
                        else:
                            pred = np.full(horizon, np.mean(train))
                    agent._current_rule_strategy = None  # 集成模式不依赖 agent
                else:
                    # 普通轮次模式：使用最佳策略
                    scored = round_policies.get(mode, [])
                    if scored:
                        best_policy = scored[0][0]
                        if best_policy and best_policy.state_condition:
                            agent._current_rule_strategy = best_policy.skill_strategy
                        else:
                            agent._current_rule_strategy = None
                    else:
                        agent._current_rule_strategy = None
                    try:
                        pred = agent.predict(task)
                    except:
                        pred = None

                # 处理预测
                if pred is not None and len(pred) == len(test):
                    metrics = compute_all_metrics(np.array(pred), test, mase_scale)
                    window_mases[mode] = metrics.get('mase', float('inf'))
                    window_metric[mode] = metrics
                else:
                    mean_pred = np.full(len(test), np.mean(train))
                    metrics = compute_all_metrics(mean_pred, test, mase_scale)
                    window_mases[mode] = metrics.get('mase', float('inf'))
                    window_metric[mode] = metrics

                elapsed = time.time() - start_time
                window_mode_times[mode] = elapsed

            return {
                'window_id': window_id,
                'window_metric': window_metric,
                'window_mases': window_mases,
                'window_mode_times': window_mode_times
            }
        except Exception as e:
            return {'window_id': window_id, 'error': str(e), 'window_metric': {'window_id': window_id}}

    if test_parallel and total_windows > 1:
        logger.log(f"   ⚡ 并行处理 {total_windows} 个窗口 (workers={test_workers})...")
        with ThreadPoolExecutor(max_workers=test_workers) as executor:
            futures = {}
            for idx, row in test_df.iterrows():
                future = executor.submit(process_window, idx, row)
                futures[future] = idx

            pbar = tqdm(total=total_windows, desc="Ablation 进度", unit="窗口", ncols=100)
            for future in as_completed(futures):
                result = future.result()
                if result and 'window_metric' in result:
                    window_metrics.append(result['window_metric'])
                    for mode in modes:
                        mase = result['window_mases'].get(mode, float('inf'))
                        if mase != float('inf') and not np.isnan(mase):
                            results[mode]['mases'].append(mase)
                        elapsed = result.get('window_mode_times', {}).get(mode, 0.0)
                        with mode_times_lock:
                            mode_times[mode] += elapsed
                            mode_counts[mode] += 1
                pbar.update(1)
            pbar.close()
    else:
        pbar = tqdm(total=total_windows, desc="Ablation 进度", unit="窗口", ncols=100)
        for idx, row in test_df.iterrows():
            result = process_window(idx, row)
            if result and 'window_metric' in result:
                window_metrics.append(result['window_metric'])
                for mode in modes:
                    mase = result['window_mases'].get(mode, float('inf'))
                    if mase != float('inf') and not np.isnan(mase):
                        results[mode]['mases'].append(mase)
                    elapsed = result.get('window_mode_times', {}).get(mode, 0.0)
                    mode_times[mode] += elapsed
                    mode_counts[mode] += 1
            pbar.update(1)
        pbar.close()

    summary = {}
    for mode in modes:
        mases = results[mode]['mases']
        if mases:
            valid_mases = [m for m in mases if not np.isnan(m) and m != float('inf')]
            if valid_mases:
                summary[mode] = {
                    'mase': np.mean(valid_mases),
                    'mae': np.mean([m.get('mae', 0) for m in results[mode]['metrics'] if m]),
                    'rmse': np.mean([m.get('rmse', 0) for m in results[mode]['metrics'] if m]),
                    'smape': np.mean([m.get('smape', 0) for m in results[mode]['metrics'] if m]),
                    'owa': np.mean([m.get('owa', 0) for m in results[mode]['metrics'] if m]),
                }

    # 打印耗时统计
    logger.log("\n" + "=" * 70)
    logger.log("📊 Ablation 各模式耗时统计")
    logger.log("=" * 70)
    logger.log(f"{'模式':<15} | {'总耗时':<12} | {'窗口数':<8} | {'平均每窗口':<12}")
    logger.log("-" * 60)
    for mode in modes:
        total_sec = mode_times.get(mode, 0.0)
        count = mode_counts.get(mode, 0)
        avg_sec = total_sec / count if count > 0 else 0.0
        logger.log(f"{mode:<15} | {total_sec:>8.2f}s   | {count:>6}   | {avg_sec:>8.2f}s")
    logger.log("=" * 70)

    return summary


def run_ablation_fallback(test_df: pd.DataFrame) -> Dict:
    """Ablation 回退（无可用模型时使用均值）"""
    from experiments.autotune.utils import load_window_data, compute_all_metrics

    results = {}
    mases = {'no_rule': []}

    for _, row in test_df.iterrows():
        window_data_path = row.get('window_data_path')
        if not window_data_path or not os.path.exists(window_data_path):
            continue
        try:
            wdata = load_window_data(window_data_path)
            train = wdata['train']
            test = wdata['test']
            period = wdata.get('period', 365)
            mase_scale = wdata.get('mase_scale', 1.0)

            mean_pred = np.full(len(test), np.mean(train))
            metrics = compute_all_metrics(mean_pred, test, mase_scale)
            mases['no_rule'].append(metrics.get('mase', 0))
        except:
            continue

    if mases['no_rule']:
        results['no_rule'] = {'mase': np.mean(mases['no_rule']), 'mae': 0, 'rmse': 0, 'smape': 0, 'owa': 0}

    return results


def print_ablation_summary(logger, ablation_results: Dict):
    """打印 Ablation 摘要，特别突出 ensemble_top2"""
    logger.log("\n📊 多版本 Ablation 摘要:")
    logger.log("─" * 80)

    baseline = ablation_results.get('no_rule', {})
    baseline_mase = baseline.get('mase', 0)

    for mode, data in ablation_results.items():
        if mode == 'no_rule':
            logger.log(f"   no_rule: MASE={data.get('mase', 0):.6f}")
        else:
            mase = data.get('mase', 0)
            if baseline_mase > 0:
                improvement = (baseline_mase - mase) / baseline_mase * 100
                status = "✅" if improvement > 0 else "⚠️"
                logger.log(f"   {mode}: MASE={mase:.6f} ({status} 改善 {improvement:+.2f}%)")

    logger.log("─" * 80)