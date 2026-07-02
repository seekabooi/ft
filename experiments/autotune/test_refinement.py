# experiments/autotune/test_refinement.py
import os
import json
import argparse
import pandas as pd
import numpy as np
from experiments.autotune.utils import ProgressLogger, load_config, load_window_data, compute_mase
from experiments.autotune.rule_engine import RuleEngine


def test_rules(dataset_name: str, rules_file: str, output_dir: str):
    """测试给定规则在验证集上的平均MASE"""
    logger = ProgressLogger(verbose=True)
    logger.log(f"🧪 测试规则: {rules_file}")

    # 加载规则（指定 UTF-8 编码）
    with open(rules_file, 'r', encoding='utf-8') as f:
        rules_data = json.load(f)
    rules = rules_data.get('rules', [])

    # 加载采集数据
    csv_path = os.path.join(output_dir, "collected_windows.csv")
    if not os.path.exists(csv_path):
        logger.log("❌ 采集数据不存在")
        return None

    df = pd.read_csv(csv_path)
    rule_engine = RuleEngine({'rules': rules})

    mases = []
    for idx, row in df.iterrows():
        window_data_path = row['window_data_path']
        if not window_data_path or not os.path.exists(window_data_path):
            continue
        try:
            wdata = load_window_data(window_data_path)
            train = wdata['train']
            test = wdata['test']
            period = wdata.get('period', 365)
            mase_scale = wdata.get('mase_scale', 1.0)

            from experiments.autotune.utils import extract_features
            features = extract_features(train)
            strategy = rule_engine.get_strategy(features)
            if strategy is None:
                continue

            # 预测
            pred = predict_with_strategy(train, len(test), period, strategy)
            if pred is not None and len(pred) == len(test):
                mase = compute_mase(pred, test, mase_scale)
                mases.append(mase)
        except Exception as e:
            pass

    if mases:
        avg_mase = np.mean(mases)
        logger.log(f"✅ 平均MASE: {avg_mase:.4f} (基于 {len(mases)} 个窗口)")
        return avg_mase
    else:
        logger.log("⚠️ 无有效结果")
        return None


def predict_with_strategy(train, horizon, period, strategy):
    # 同performance_auditor中的实现
    try:
        from src.skills.registry import SkillRegistry
        from run_benchmark import build_full_registry
        full_registry, _ = build_full_registry()
        stages = strategy.get('stages', [])
        if not stages:
            return None
        predictions = []
        current_hist = train.copy()
        for stage in stages:
            steps = stage.get('steps', 0)
            weights = stage.get('weights', {})
            for _ in range(steps):
                pred_val = 0.0
                total_w = 0.0
                for skill_name, weight in weights.items():
                    skill = full_registry.get(skill_name)
                    if skill and weight > 0:
                        try:
                            forecast = skill.execute(current_hist, 1, period=period)
                            if forecast is not None and len(forecast) > 0:
                                pred_val += forecast[0] * weight
                                total_w += weight
                        except:
                            pass
                if total_w > 0:
                    pred_val /= total_w
                else:
                    pred_val = np.mean(current_hist[-5:]) if len(current_hist) >= 5 else np.mean(current_hist)
                predictions.append(pred_val)
                current_hist = np.append(current_hist, pred_val)
        return np.array(predictions[:horizon])
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='melbourne_temp')
    parser.add_argument('--original_rules', type=str, default='storage/autotune_results/generated_rules.json')
    parser.add_argument('--refined_rules', type=str, default='storage/autotune_results/refined_rules.json')
    parser.add_argument('--output_dir', type=str, default='storage/autotune_results')
    args = parser.parse_args()

    print("\n" + "="*60)
    print("📊 测试改进前后规则效果")
    print("="*60)

    avg_original = test_rules(args.dataset, args.original_rules, args.output_dir)
    avg_refined = test_rules(args.dataset, args.refined_rules, args.output_dir)

    if avg_original is not None and avg_refined is not None:
        improvement = (avg_original - avg_refined) / avg_original * 100
        print(f"\n📈 改进前MASE: {avg_original:.4f}")
        print(f"📈 改进后MASE: {avg_refined:.4f}")
        print(f"📈 改善: {improvement:.2f}%")


if __name__ == '__main__':
    main()