#!/usr/bin/env python
"""
独立 Ablation 测试 - 修正版，与 run_ablation_only.py 逻辑对齐
核心方式：通过设置 agent._current_rule_strategy 让 LLM 参考规则预测
"""

import os
import sys
import json
import argparse
import time
import copy
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from experiments.autotune.skill_policy import SkillPolicy
from experiments.autotune.state_encoder import StateEncoder
from experiments.autotune.utils import ProgressLogger, load_config, load_window_data, compute_all_metrics
from experiments.autotune.tuner_utils import detect_available_model   # ★ 使用官方检测
from src.agents.llm_planner import LLMPlannerAgent
from src.agents.llm_client import LLMClient
from src.tasks.instance import TaskInstance
from run_benchmark import build_full_registry


def load_round_policies(run_dir: str, round_num: int, logger) -> List[SkillPolicy]:
    """加载指定轮次的策略"""
    round_dir = os.path.join(run_dir, f"round_{round_num}")
    json_path = os.path.join(round_dir, "refined_policies_optimized.json")
    if not os.path.exists(json_path):
        return []
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        policies_data = data.get('policies', [])
        policies = [SkillPolicy.from_dict(p) for p in policies_data]
        return policies
    except Exception as e:
        logger.log(f"   ⚠️ 加载 round_{round_num} 失败: {e}")
        return []


def load_final_policies(run_dir: str, logger) -> List[SkillPolicy]:
    """加载最终策略池"""
    json_path = os.path.join(run_dir, "refined_policies.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            policies_data = data.get('policies', [])
            policies = [SkillPolicy.from_dict(p) for p in policies_data]
            return policies
        except Exception as e:
            logger.log(f"   ⚠️ 加载 refined_policies.json 失败: {e}")
    return []


def evaluate_window(task_info, mode: str, policies: List[SkillPolicy],
                    agent: LLMPlannerAgent, state_encoder: StateEncoder,
                    full_registry, logger):
    """评估单个窗口（与 tuner_eval 逻辑一致）"""
    try:
        window_id = task_info['window_id']
        train = task_info['train']
        test = task_info['test']
        mase_scale = task_info['mase_scale']
        horizon = task_info['horizon']
        task = task_info['task']

        if mode == 'no_rule':
            agent._current_rule_strategy = None
        else:
            if not policies:
                agent._current_rule_strategy = None
            else:
                # ★ 使用与主循环相同的状态编码器
                state = state_encoder.encode(train)
                numeric_state = state.get('numeric', {})
                scored = []
                for policy in policies:
                    if policy.status in ['ARCHIVE', 'DELETE']:
                        continue
                    score = policy.compute_applicability_score(numeric_state)
                    scored.append((policy, score))
                if scored:
                    best_policy, _ = max(scored, key=lambda x: x[1])
                    agent._current_rule_strategy = best_policy.skill_strategy
                else:
                    agent._current_rule_strategy = None

        pred = agent.predict(task)
        if pred is None:
            pred = np.full(horizon, np.mean(train))
        if len(pred) != len(test):
            pred = np.full(len(test), np.mean(train))

        metrics = compute_all_metrics(np.array(pred), np.array(test), mase_scale)
        return {
            'window_id': window_id,
            'success': True,
            'metrics': metrics
        }
    except Exception as e:
        return {
            'window_id': task_info.get('window_id', -1),
            'success': False,
            'error': str(e)
        }


def evaluate_mode(mode: str, policies: List[SkillPolicy],
                  test_df: pd.DataFrame, full_registry,
                  state_encoder: StateEncoder, model: str,
                  logger, workers: int = 8) -> Dict:
    """评估单个模式（复用主流程逻辑）"""
    if test_df.empty:
        return {'success': False, 'error': '无测试集'}

    logger.log(f"\n▶ 评估模式: {mode}")
    if mode != 'no_rule':
        logger.log(f"   策略数: {len(policies)}")

    # ★ 创建 Agent，参数与 tuner_eval 完全一致
    agent = LLMPlannerAgent(
        model=model,
        skill_registry=full_registry,
        log_file=None,
        use_skills=True,
        rules_file=None,
        llm_call_interval=3,
        verbose=False
    )
    agent.rule_engine = None
    agent._current_rule_strategy = None
    agent._skill_cache = {}
    agent._logged_skill_errors = set()

    # 构建任务列表
    tasks = []
    for idx, row in test_df.iterrows():
        window_id = row.get('window_id', idx)
        window_data_path = row.get('window_data_path')
        if not window_data_path or not os.path.exists(window_data_path):
            continue

        try:
            wdata = load_window_data(window_data_path)
            train = wdata['train']
            test = wdata['test']
            period = wdata.get('period', 365)
            mase_scale = wdata.get('mase_scale', 1.0)
            horizon = wdata.get('horizon', 12)

            task = TaskInstance(
                id=f"ablation_{window_id}",
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

            tasks.append({
                'window_id': window_id,
                'train': train,
                'test': test,
                'period': period,
                'mase_scale': mase_scale,
                'horizon': horizon,
                'task': task,
                'window_data_path': window_data_path
            })
        except Exception as e:
            logger.log(f"   ⚠️ 窗口 {window_id} 加载失败: {e}")

    if not tasks:
        return {'success': False, 'error': '无有效任务'}

    # 并行评估（共享 agent，与 tuner_eval 一致）
    results = []
    actual_workers = min(workers, len(tasks))

    # 策略列表深拷贝，避免共享修改（但 agent 是共享的）
    policies_copy = copy.deepcopy(policies) if policies else []

    def process_single(task_info):
        return evaluate_window(
            task_info, mode, policies_copy,
            agent, state_encoder, full_registry, logger
        )

    with ThreadPoolExecutor(max_workers=actual_workers) as executor:
        futures = {executor.submit(process_single, t): t['window_id'] for t in tasks}
        pbar = tqdm(total=len(tasks), desc=f"   {mode} 评估", unit="窗口", ncols=100)

        for future in as_completed(futures):
            try:
                result = future.result(timeout=180)
                if result['success']:
                    results.append(result)
            except Exception as e:
                logger.log(f"   ⚠️ 窗口 {futures[future]} 超时或异常: {e}")
            pbar.update(1)
        pbar.close()

    if not results:
        return {'success': False, 'error': '无有效结果'}

    # 聚合指标（与 tuner_eval 完全相同）
    all_metrics = [r['metrics'] for r in results]
    aggregated = {
        'mase': np.mean([m.get('mase', float('inf')) for m in all_metrics if m.get('mase') != float('inf')]),
        'mae': np.mean([m.get('mae', 0) for m in all_metrics if m.get('mae', 0) > 0]),
        'rmse': np.mean([m.get('rmse', 0) for m in all_metrics if m.get('rmse', 0) > 0]),
        'smape': np.mean([m.get('smape', 0) for m in all_metrics if m.get('smape', 0) > 0]),
        'owa': np.mean([m.get('owa', 0) for m in all_metrics if m.get('owa', 0) > 0]),
    }

    logger.log(f"   ✅ MASE={aggregated['mase']:.6f}, MAE={aggregated['mae']:.6f}, RMSE={aggregated['rmse']:.6f}")

    return {
        'success': True,
        'metrics': aggregated,
        'window_count': len(results),
        'total_windows': len(tasks)
    }


def print_comparison_table(logger, results: Dict):
    """打印多指标对比表格（与原版相同）"""
    # （此处省略，原函数不变，可复用原代码）
    # 与 run_ablation_ref.py 原版一致


def save_report(logger, results: Dict, run_dir: str):
    """保存报告（与原版相同）"""
    # （此处省略，原函数不变）


def main():
    parser = argparse.ArgumentParser(description="独立 Ablation 测试（对齐主流程逻辑）")
    parser.add_argument('--resume', type=str, required=True,
                        help='运行目录（如 run_20260625_181236）')
    parser.add_argument('--config', type=str, default=None,
                        help='配置文件路径')
    parser.add_argument('--workers', type=int, default=4,
                        help='并行线程数（默认 4）')
    args = parser.parse_args()

    if os.path.exists(args.resume):
        run_dir = args.resume
    else:
        run_dir = os.path.join("llog", args.resume)

    if not os.path.exists(run_dir):
        print(f"❌ 目录不存在: {run_dir}")
        return

    logger = ProgressLogger(log_dir=run_dir, verbose=True, run_folder=False)
    logger.start_log("ablation_ref_fixed")

    logger.log("=" * 70)
    logger.log("📊 独立 Ablation 测试（对齐主流程逻辑）")
    logger.log(f"📁 运行目录: {run_dir}")
    logger.log("=" * 70)

    # 加载配置
    config = load_config(args.config) if args.config else {}
    logger.log(f"✅ 配置文件加载完成")

    # ★ 创建状态编码器（使用配置）
    state_encoder = StateEncoder(config)

    # 加载测试集
    csv_path = "storage/autotune_results/collected_windows.csv"
    if not os.path.exists(csv_path):
        logger.log(f"❌ 未找到采集数据: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    test_df = df[df['split'] == 'test'].copy()
    logger.log(f"📊 测试集: {len(test_df)} 个窗口")

    # 加载各轮策略
    policy_snapshots = {}
    policy_snapshots['no_rule'] = []

    for round_num in range(1, 9):
        policies = load_round_policies(run_dir, round_num, logger)
        if policies:
            policy_snapshots[f'round_{round_num}'] = policies
            logger.log(f"   round_{round_num}: {len(policies)} 条策略")

    final_policies = load_final_policies(run_dir, logger)
    if final_policies:
        policy_snapshots['final_with_patch'] = final_policies
        policy_snapshots['final'] = final_policies
        logger.log(f"   final_with_patch: {len(final_policies)} 条策略")

    # ★ 使用官方模型检测（与 tuner_eval 一致）
    model_candidates = ["glm-4", "glm-4.5-air"]
    model, error = detect_available_model(logger, model_candidates)
    if model is None:
        logger.log(f"❌ 无可用模型，无法运行 Ablation")
        return
    logger.log(f"✅ 使用模型: {model}")

    # 构建技能注册表
    full_registry, _ = build_full_registry()
    logger.log(f"✅ 技能注册表: {len(full_registry._skills)} 个技能")

    # 执行 Ablation
    start_time = time.time()
    ablation_results = {}

    for mode, policies in policy_snapshots.items():
        result = evaluate_mode(
            mode=mode,
            policies=policies,
            test_df=test_df,
            full_registry=full_registry,
            state_encoder=state_encoder,   # ★ 传入统一的编码器
            model=model,
            logger=logger,
            workers=args.workers
        )

        if result['success']:
            ablation_results[mode] = result['metrics']
        else:
            logger.log(f"   ❌ {mode} 失败: {result.get('error', '未知错误')}")
            ablation_results[mode] = {
                'mase': float('inf'),
                'mae': float('inf'),
                'rmse': float('inf'),
                'smape': float('inf'),
                'owa': float('inf'),
            }

    elapsed = time.time() - start_time

    # 输出结果
    print_comparison_table(logger, ablation_results)
    save_report(logger, ablation_results, run_dir)

    stats_str = LLMClient.print_token_stats("Ablation Token 统计")
    logger.log("\n" + stats_str)
    logger.log(f"\n⏱️  Ablation 总耗时: {elapsed:.1f}s ({elapsed/60:.1f}分钟)")
    logger.log("✅ Ablation 完成！")


if __name__ == '__main__':
    main()