# compare_three_modes.py
"""
三模式完整对比（多指标 + 测试集 + 配置文件）
- 支持 config.yaml 中的 three_mode_comparison 配置
- 默认使用 collected_windows.csv 的 test 集，可切换为 raw 模式
- 输出 7 个评估指标

用法:
    python compare_three_modes.py --dataset melbourne_temp
    python compare_three_modes.py --dataset melbourne_temp --config experiments/autotune/config.yaml
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm

# 添加项目根目录
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.agents.llm_planner import LLMPlannerAgent
from src.tasks.instance import TaskInstance
from experiments.autotune.utils import (
    load_window_data, compute_mase, compute_smape,
    compute_rmse, compute_owa, extract_features, load_config
)
from experiments.autotune.rule_engine import RuleEngine
from run_benchmark import build_full_registry


def load_rules(file_path):
    """加载规则文件，如果不存在返回空列表"""
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('rules', [])
    return []


def compute_all_metrics(pred: np.ndarray, actual: np.ndarray, mase_scale: float) -> dict:
    """计算所有评估指标"""
    if len(pred) != len(actual) or len(pred) == 0:
        return {
            'rmse': np.nan, 'mae': np.nan, 'mape': np.nan,
            'smape': np.nan, 'mase': np.nan, 'rm_sse': np.nan, 'owa': np.nan
        }

    pred = np.array(pred)
    actual = np.array(actual)
    errors = pred - actual
    abs_errors = np.abs(errors)

    rmse = np.sqrt(np.mean(errors ** 2))
    mae = np.mean(abs_errors)
    mape = np.mean(abs_errors / (np.abs(actual) + 1e-8)) * 100
    smape = compute_smape(pred, actual)
    mase = compute_mase(pred, actual, mase_scale)

    n = len(actual)
    if n > 1:
        diff = np.diff(actual)
        denominator = np.sqrt(np.mean(diff ** 2)) + 1e-8
        rm_sse = np.sqrt(np.mean(errors ** 2)) / denominator
    else:
        rm_sse = np.nan

    owa = compute_owa(pred, actual)

    return {
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'smape': smape,
        'mase': mase,
        'rm_sse': rm_sse,
        'owa': owa
    }


def evaluate_window(train, test, period, mase_scale, horizon, agent, rule_engine=None):
    """评估单个窗口，返回两种模式的指标"""
    from experiments.autotune.inducer import RuleInducer

    # 1. 无规则预测
    task = TaskInstance(
        id=f"compare",
        dataset_id="compare",
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
    pred_no_rule = agent.predict(task)
    pred_no_rule = np.array(pred_no_rule)
    metrics_no = compute_all_metrics(pred_no_rule, test, mase_scale) if len(pred_no_rule) == len(test) else None

    # 2. 规则预测
    metrics_rule = None
    if rule_engine:
        features = extract_features(train)
        strategy = rule_engine.get_strategy(features)
        if strategy:
            inducer = RuleInducer({}, None)
            pred_rule = inducer._predict_with_strategy(train, horizon, period, strategy)
            if pred_rule is not None and len(pred_rule) == len(test):
                metrics_rule = compute_all_metrics(pred_rule, test, mase_scale)

    return metrics_no, metrics_rule


def main():
    parser = argparse.ArgumentParser(description="三模式多指标对比")
    parser.add_argument('--dataset', type=str, default='melbourne_temp', help='数据集名称')
    parser.add_argument('--config', type=str, default=None, help='配置文件路径（默认 experiments/autotune/config.yaml）')
    parser.add_argument('--output', type=str, default=None, help='输出CSV路径')
    args = parser.parse_args()

    # ★★★ 修正：使用 load_config 的默认行为（自动加载 experiments/autotune/config.yaml） ★★★
    # 如果提供了 --config，则使用指定路径；否则不传参让 load_config 自动找默认位置
    if args.config:
        config = load_config(args.config)
    else:
        config = load_config()  # 自动加载 experiments/autotune/config.yaml

    print(f"📋 配置文件: {args.config or 'experiments/autotune/config.yaml (默认)'}")

    # 读取 three_mode_comparison 配置
    three_cfg = config.get('three_mode_comparison', {})

    # 调试：打印当前 data_source
    data_source = three_cfg.get('data_source', 'collected')
    print(f"📋 data_source: {data_source}")

    output_csv = args.output or three_cfg.get('output_csv', 'storage/three_modes_comparison.csv')
    gen_rules_path = three_cfg.get('generated_rules', 'storage/autotune_results/generated_rules.json')
    ref_rules_path = three_cfg.get('refined_rules', 'storage/autotune_results/refined_rules.json')

    print("=" * 90)
    print(f"📊 三模式多指标对比: {args.dataset}")
    print("=" * 90)

    # 根据 data_source 决定数据来源
    if data_source == 'raw':
        # 从原始数据集重新生成窗口
        window_size = three_cfg.get('window_size', 600)
        step = three_cfg.get('step', 150)
        horizon = three_cfg.get('horizon', 7)
        start = three_cfg.get('start', 0)
        end = three_cfg.get('end', 3000)

        print(f"📋 数据源: raw (从原始数据重新生成)")
        print(f"📋 参数: window_size={window_size}, step={step}, horizon={horizon}, start={start}, end={end}")

        from src.dataset.registry import DatasetRegistry
        from src.dataset.loader import load_dataset
        registry = DatasetRegistry()
        ds_config = registry.get(args.dataset)
        if not ds_config:
            print(f"❌ 数据集 {args.dataset} 不存在")
            return
        df = load_dataset(ds_config)
        target_col = ds_config['target_column']
        series = df[target_col].values
        freq = ds_config.get('frequency', 'daily')

        windows = []
        for origin in range(start, end + 1, step):
            if origin + window_size + horizon > len(series):
                break
            train = series[origin:origin + window_size]
            test = series[origin + window_size:origin + window_size + horizon]
            from experiments.autotune.utils import detect_period
            period = detect_period(train, freq)
            mase_scale = 1.0
            windows.append({
                'train': train,
                'test': test,
                'period': period,
                'mase_scale': mase_scale,
                'horizon': horizon,
                'window_id': origin
            })
        print(f"📋 生成了 {len(windows)} 个窗口")
        all_windows = windows
    else:
        # 使用 collected_windows.csv 的 test 集
        print(f"📋 数据源: collected (使用 collected_windows.csv 的 test 集)")
        csv_path = "storage/autotune_results/collected_windows.csv"
        if not os.path.exists(csv_path):
            print(f"❌ 未找到采集数据: {csv_path}")
            print("   请先运行: python -m experiments.autotune.main --dataset melbourne_temp --horizon 7 --compare")
            return

        df_collected = pd.read_csv(csv_path)
        test_df = df_collected[df_collected['split'] == 'test']
        if len(test_df) == 0:
            print("❌ 没有测试集窗口")
            return
        print(f"📋 使用 {len(test_df)} 个测试集窗口")

        all_windows = []
        for _, row in test_df.iterrows():
            wpath = row['window_data_path']
            if not wpath or not os.path.exists(wpath):
                continue
            wdata = load_window_data(wpath)
            all_windows.append({
                'train': wdata['train'],
                'test': wdata['test'],
                'period': wdata.get('period', 365),
                'mase_scale': wdata.get('mase_scale', 1.0),
                'horizon': wdata.get('horizon', 7),
                'window_id': row.get('window_id', 'unknown')
            })

    print(f"📋 最终窗口数: {len(all_windows)}")

    # 加载规则
    gen_rules = load_rules(gen_rules_path)
    ref_rules = load_rules(ref_rules_path)

    print(f"📋 初始规则数: {len(gen_rules)}")
    print(f"📋 优化规则数: {len(ref_rules)}")

    # 构建 Agent
    full_registry, _ = build_full_registry()
    agent = LLMPlannerAgent(
        model="glm-4",
        skill_registry=full_registry,
        log_file=None,
        use_skills=True,
        rules_file=None
    )

    rule_engine_gen = RuleEngine({'rules': gen_rules}) if gen_rules else None
    rule_engine_ref = RuleEngine({'rules': ref_rules}) if ref_rules else None

    # 逐窗口评估
    all_metrics = {
        'no_rule': [],
        'generated': [],
        'refined': []
    }

    for w in tqdm(all_windows, desc="评估窗口"):
        train = w['train']
        test = w['test']
        period = w['period']
        mase_scale = w['mase_scale']
        horizon = w['horizon']

        metrics_no, _ = evaluate_window(train, test, period, mase_scale, horizon, agent)
        _, metrics_gen = evaluate_window(train, test, period, mase_scale, horizon, agent, rule_engine_gen)
        _, metrics_ref = evaluate_window(train, test, period, mase_scale, horizon, agent, rule_engine_ref)

        if metrics_no:
            all_metrics['no_rule'].append(metrics_no)
        if metrics_gen:
            all_metrics['generated'].append(metrics_gen)
        if metrics_ref:
            all_metrics['refined'].append(metrics_ref)

    # 汇总统计
    metric_names = ['rmse', 'mae', 'mape', 'smape', 'mase', 'rm_sse', 'owa']
    display_names = ['RMSE', 'MAE', 'MAPE', 'sMAPE', 'MASE', 'RMSSE', 'OWA']

    results = {}
    for mode in ['no_rule', 'generated', 'refined']:
        if not all_metrics[mode]:
            continue
        mode_data = {}
        for m in metric_names:
            vals = [d[m] for d in all_metrics[mode] if not np.isnan(d[m])]
            if vals:
                mode_data[m] = {'mean': np.mean(vals), 'std': np.std(vals)}
            else:
                mode_data[m] = {'mean': np.nan, 'std': np.nan}
        results[mode] = mode_data

    # 打印表格
    print("\n" + "=" * 90)
    print("📊 多指标对比结果（测试集泛化能力）")
    print("=" * 90)
    print(f"窗口数: {len(all_metrics['no_rule'])}")
    print("-" * 90)
    print(
        f"{'指标':<10} | {'NoRule_Mean':<12} | {'NoRule_Std':<12} | {'Gen_Mean':<12} | {'Gen_Std':<12} | {'Ref_Mean':<12} | {'Ref_Std':<12} | {'Gen_vs_No%':<12} | {'Ref_vs_No%':<12} | {'Ref_vs_Gen%':<12}")
    print("-" * 90)

    for idx, m in enumerate(metric_names):
        no = results.get('no_rule', {}).get(m, {'mean': np.nan, 'std': np.nan})
        gen = results.get('generated', {}).get(m, {'mean': np.nan, 'std': np.nan})
        ref = results.get('refined', {}).get(m, {'mean': np.nan, 'std': np.nan})

        gen_imp = (no['mean'] - gen['mean']) / no['mean'] * 100 if no['mean'] > 0 else np.nan
        ref_imp = (no['mean'] - ref['mean']) / no['mean'] * 100 if no['mean'] > 0 else np.nan
        ref_vs_gen = (gen['mean'] - ref['mean']) / gen['mean'] * 100 if gen['mean'] > 0 else np.nan

        if m in ['mape', 'smape']:
            fmt = "{:.2f}"
        else:
            fmt = "{:.4f}"

        print(
            f"{display_names[idx]:<10} | {fmt.format(no['mean']):<12} | {fmt.format(no['std']):<12} | {fmt.format(gen['mean']):<12} | {fmt.format(gen['std']):<12} | {fmt.format(ref['mean']):<12} | {fmt.format(ref['std']):<12} | {gen_imp:>11.2f}% | {ref_imp:>11.2f}% | {ref_vs_gen:>11.2f}%")

    print("=" * 90)

    # 保存 CSV
    rows = []
    for idx, m in enumerate(metric_names):
        no = results.get('no_rule', {}).get(m, {'mean': np.nan, 'std': np.nan})
        gen = results.get('generated', {}).get(m, {'mean': np.nan, 'std': np.nan})
        ref = results.get('refined', {}).get(m, {'mean': np.nan, 'std': np.nan})
        rows.append({
            'metric': display_names[idx],
            'NoRule_Mean': no['mean'],
            'NoRule_Std': no['std'],
            'GenRule_Mean': gen['mean'],
            'GenRule_Std': gen['std'],
            'RefRule_Mean': ref['mean'],
            'RefRule_Std': ref['std'],
            'Gen_vs_No_%': (no['mean'] - gen['mean']) / no['mean'] * 100 if no['mean'] > 0 else np.nan,
            'Ref_vs_No_%': (no['mean'] - ref['mean']) / no['mean'] * 100 if no['mean'] > 0 else np.nan,
            'Ref_vs_Gen_%': (gen['mean'] - ref['mean']) / gen['mean'] * 100 if gen['mean'] > 0 else np.nan
        })

    df_results = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
    df_results.to_csv(output_csv, index=False)
    print(f"\n📁 详细结果已保存: {output_csv}")


if __name__ == '__main__':
    main()