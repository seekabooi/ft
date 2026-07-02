# compare_sliding.py
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from typing import List, Dict, Tuple

from src.dataset.registry import DatasetRegistry
from src.dataset.loader import load_dataset
from src.agents.llm_planner import LLMPlannerAgent
from src.tasks.instance import TaskInstance
from run_benchmark import build_full_registry
from src.skills.data_profiler import DataProfiler
from experiments.autotune.utils import load_config


def load_data(dataset_name: str) -> Tuple[np.ndarray, str]:
    registry = DatasetRegistry()
    ds_config = registry.get(dataset_name)
    if not ds_config:
        raise ValueError(f"Dataset {dataset_name} not found")
    df = load_dataset(ds_config)
    target_col = ds_config['target_column']
    series = df[target_col].values
    freq = ds_config.get('frequency', 'daily')
    return series, freq


def compute_all_metrics(pred: np.ndarray, true: np.ndarray, mase_scale: float) -> Dict:
    """计算多种评价指标"""
    errors = pred - true
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))
    # MAPE: 平均绝对百分比误差
    mape = np.mean(np.abs(errors / (true + 1e-8))) * 100
    # sMAPE: 对称平均绝对百分比误差
    smape = np.mean(2.0 * np.abs(errors) / (np.abs(pred) + np.abs(true) + 1e-8)) * 100
    # MASE: 平均绝对缩放误差
    mase = mae / mase_scale if mase_scale != 0 else np.nan
    # OWA: 整体加权平均（近似，使用MASE和RMSSE的均等加权，但RMSSE未计算，这里用RMSE替代）
    # 为简化，OWA = (MASE + RMSE/scale) / 2，但scale不同，故不使用
    # 改为计算RMSSE（基于季节性朴素缩放因子）
    # 使用与MASE相同的缩放因子
    rmsse = np.sqrt(np.mean(errors ** 2)) / mase_scale if mase_scale != 0 else np.nan
    # OWA = (MASE + RMSSE) / 2
    owa = (mase + rmsse) / 2 if not np.isnan(mase) and not np.isnan(rmsse) else np.nan
    return {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'sMAPE': smape,
        'MASE': mase,
        'RMSSE': rmsse,
        'OWA': owa
    }


def predict_window_with_trajectory(agent, train: np.ndarray, horizon: int, freq: str):
    """返回 (预测值, 轨迹列表)"""
    task = TaskInstance(
        id="sliding_compare",
        dataset_id="compare",
        template_id="fixed_origin",
        question="",
        question_type="numerical",
        history=train.tolist(),
        horizon=horizon,
        frequency=freq,
        prediction_target={},
        resolution_date=datetime.now(),
        difficulty_level=1,
        ground_truth_extractor="",
        dates=None,
        target_date=""
    )
    pred, trajectory = agent.predict_with_trajectory(task)
    return np.array(pred), trajectory


def main():
    parser = argparse.ArgumentParser(description="滑动窗口对比：规则模式 vs 原始LLM")
    parser.add_argument('--dataset', type=str, default='melbourne_temp')
    parser.add_argument('--config', type=str, default='experiments/autotune/config.yaml',
                        help='配置文件路径')
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)
    comparison_config = config.get('comparison', {})
    window_size = comparison_config.get('window_size', 600)
    horizon = comparison_config.get('horizon', 7)
    step = comparison_config.get('step', 150)
    start = comparison_config.get('start', 0)          # 窗口起点下限
    end = comparison_config.get('end', None)           # 窗口起点上限（可选）
    rules_file = comparison_config.get('rules_file', 'storage/autotune_results/refined_rules.json')
    output_dir = comparison_config.get('output_dir', 'storage')
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, 'comparison_detailed.log')

    # 加载数据
    print("加载数据...")
    series, freq = load_data(args.dataset)
    n = len(series)

    # 生成窗口起点列表
    max_start = n - window_size - horizon
    if end is not None and end < max_start:
        max_start = min(end, max_start)
    if start > max_start:
        print(f"起始点{start}超出数据范围，调整至{max_start}")
        start = max_start
    starts = list(range(start, max_start + 1, step))
    if not starts:
        print("数据太短，无法生成任何窗口。")
        return
    print(f"生成 {len(starts)} 个滑动窗口 (大小={window_size}, 步长={step}, 起点范围={start}~{max_start})")

    # 构建技能注册表
    full_registry, _ = build_full_registry()

    # 两个模式的预测结果和轨迹存储
    results_rule = []
    trajectories_rule = []
    results_no_rule = []
    trajectories_no_rule = []

    # 创建Agent（规则模式）
    agent_rule = LLMPlannerAgent(
        model="glm-4",
        skill_registry=full_registry,
        log_file=None,          # 不写入日志文件，由我们手动记录
        use_skills=True,
        verbose=False,          # 静默模式
        rules_file=rules_file
    )

    # 创建Agent（无规则模式）
    agent_no_rule = LLMPlannerAgent(
        model="glm-4",
        skill_registry=full_registry,
        log_file=None,
        use_skills=True,
        verbose=False,          # 静默模式
        rules_file=None
    )

    # 打开详细日志文件
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"对比测试详细日志\n")
        f.write(f"数据集: {args.dataset}, 窗口大小: {window_size}, 步长: {step}, 预测步数: {horizon}\n")
        f.write(f"窗口数量: {len(starts)}\n")
        f.write("="*80 + "\n")

    # 先跑规则模式
    print("运行规则模式...")
    for idx, start_pos in enumerate(tqdm(starts, desc="规则模式")):
        train = series[start_pos:start_pos + window_size]
        test = series[start_pos + window_size:start_pos + window_size + horizon]
        # 计算MASE缩放因子
        period = DataProfiler._auto_period(train, freq=freq)
        if period > 1 and len(train) >= 2 * period:
            seasonal_errors = np.abs(train[period:] - train[:-period])
            mase_scale = np.mean(seasonal_errors) if len(seasonal_errors) > 0 else 1.0
        else:
            naive_errors = np.abs(np.diff(train))
            mase_scale = np.mean(naive_errors) if len(naive_errors) > 0 else 1.0

        # 预测并获取轨迹
        pred_rule, traj_rule = predict_window_with_trajectory(agent_rule, train, horizon, freq)
        metrics_rule = compute_all_metrics(pred_rule, test, mase_scale)
        results_rule.append(metrics_rule)
        trajectories_rule.append(traj_rule)

        # 写入详细日志
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n窗口 {idx+1}/{len(starts)} (起点={start_pos}, 规则模式):\n")
            f.write(f"  预测值: {pred_rule.tolist()}\n")
            f.write(f"  真实值: {test.tolist()}\n")
            f.write(f"  指标: {json.dumps(metrics_rule, indent=2)}\n")
            f.write(f"  轨迹: {json.dumps(traj_rule, indent=2)}\n")

    # 再跑无规则模式
    print("运行无规则模式...")
    for idx, start_pos in enumerate(tqdm(starts, desc="无规则模式")):
        train = series[start_pos:start_pos + window_size]
        test = series[start_pos + window_size:start_pos + window_size + horizon]
        period = DataProfiler._auto_period(train, freq=freq)
        if period > 1 and len(train) >= 2 * period:
            seasonal_errors = np.abs(train[period:] - train[:-period])
            mase_scale = np.mean(seasonal_errors) if len(seasonal_errors) > 0 else 1.0
        else:
            naive_errors = np.abs(np.diff(train))
            mase_scale = np.mean(naive_errors) if len(naive_errors) > 0 else 1.0

        pred_no_rule, traj_no_rule = predict_window_with_trajectory(agent_no_rule, train, horizon, freq)
        metrics_no_rule = compute_all_metrics(pred_no_rule, test, mase_scale)
        results_no_rule.append(metrics_no_rule)
        trajectories_no_rule.append(traj_no_rule)

        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n窗口 {idx+1}/{len(starts)} (起点={start_pos}, 无规则模式):\n")
            f.write(f"  预测值: {pred_no_rule.tolist()}\n")
            f.write(f"  真实值: {test.tolist()}\n")
            f.write(f"  指标: {json.dumps(metrics_no_rule, indent=2)}\n")
            f.write(f"  轨迹: {json.dumps(traj_no_rule, indent=2)}\n")

    # 汇总对比
    df_rule = pd.DataFrame(results_rule)
    df_no_rule = pd.DataFrame(results_no_rule)

    summary_rule = df_rule.agg(['mean', 'std']).round(4)
    summary_no_rule = df_no_rule.agg(['mean', 'std']).round(4)

    compare = pd.DataFrame({
        'Rule_Mean': summary_rule.loc['mean'],
        'Rule_Std': summary_rule.loc['std'],
        'NoRule_Mean': summary_no_rule.loc['mean'],
        'NoRule_Std': summary_no_rule.loc['std'],
        'Improvement_%': ((summary_no_rule.loc['mean'] - summary_rule.loc['mean']) /
                          summary_no_rule.loc['mean'] * 100).round(2)
    })

    # 输出到终端
    print("\n" + "=" * 80)
    print("📊 滑动窗口对比结果（规则模式 vs 原始LLM）")
    print(f"窗口数量: {len(starts)}")
    print("=" * 80)
    print(compare.to_string())
    print("=" * 80)

    # 同时写入日志
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write("📊 滑动窗口对比结果（规则模式 vs 原始LLM）\n")
        f.write(f"窗口数量: {len(starts)}\n")
        f.write("=" * 80 + "\n")
        f.write(compare.to_string() + "\n")
        f.write("=" * 80 + "\n")

    # 保存CSV
    output_csv = os.path.join(output_dir, 'comparison_results.csv')
    compare.to_csv(output_csv)
    print(f"📁 结果已保存至: {output_csv}")
    print(f"📁 详细日志已保存至: {log_file}")


if __name__ == "__main__":
    main()