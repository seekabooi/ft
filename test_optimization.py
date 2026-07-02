# test_optimization.py
import sys
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiments.autotune.utils import load_config, ProgressLogger
from experiments.autotune.performance_auditor import PerformanceAuditor
from experiments.autotune.iterative_refiner import IterativeRefiner

def test_performance(rules_file, dataset_name="melbourne_temp", horizon=7):
    """测试规则在验证集上的表现"""
    from src.dataset.registry import DatasetRegistry
    from src.dataset.loader import load_dataset
    from run_benchmark import build_full_registry
    from src.agents.llm_planner import LLMPlannerAgent
    from src.tasks.instance import TaskInstance
    from experiments.autotune.utils import extract_features, compute_mase, detect_period

    # 加载数据
    registry = DatasetRegistry()
    ds_config = registry.get(dataset_name)
    df = load_dataset(ds_config)
    target_col = ds_config['target_column']
    series = df[target_col].values

    # 简单切分：取最后一段
    train_size = 600
    test_size = 100
    horizon = 7
    train = series[-train_size - test_size - horizon:-test_size - horizon]
    test = series[-test_size - horizon:-horizon]
    future = series[-horizon:]

    # 加载规则
    with open(rules_file, 'r') as f:
        rules_dict = json.load(f)
    rules = rules_dict.get('rules', [])

    from experiments.autotune.rule_engine import RuleEngine
    rule_engine = RuleEngine({'rules': rules, 'default': rules_dict.get('default', {})})

    # 使用规则预测
    full_registry, _ = build_full_registry()
    features = extract_features(train)
    strategy = rule_engine.get_strategy(features)
    if strategy:
        # 固定策略预测
        from experiments.autotune.utils import compute_mase, detect_period
        period = detect_period(train)
        from experiments.autotune.inducer import _predict_with_strategy  # 借用函数
        pred = _predict_with_strategy(train, horizon, period, strategy, full_registry)
        if pred is not None:
            mase = compute_mase(pred, future, 1.0)
            return mase
    return None

def compare():
    logger = ProgressLogger(verbose=True)
    logger.log("="*70)
    logger.log("📊 测试优化前后规则效果对比")
    logger.log("="*70)

    original_rules_file = "storage/autotune_results/generated_rules.json"
    refined_rules_file = "storage/autotune_results/refined_rules.json"

    if not os.path.exists(original_rules_file):
        logger.log("⚠️ 原始规则文件不存在，请先运行调优")
        return

    mase_original = test_performance(original_rules_file)
    if os.path.exists(refined_rules_file):
        mase_refined = test_performance(refined_rules_file)
    else:
        logger.log("⚠️ 优化后规则文件不存在，请先运行 iter_refine")
        mase_refined = None

    logger.log(f"📊 原始规则 MASE: {mase_original if mase_original else 'N/A'}")
    if mase_refined is not None:
        logger.log(f"📊 优化后规则 MASE: {mase_refined:.4f}")
        improvement = (mase_original - mase_refined) / mase_original * 100 if mase_original else 0
        logger.log(f"📈 改善: {improvement:.2f}%")
    else:
        logger.log("📊 优化后规则 MASE: 未测试")

if __name__ == "__main__":
    compare()