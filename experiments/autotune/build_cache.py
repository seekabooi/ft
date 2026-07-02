#!/usr/bin/env python
"""
独立缓存构建脚本 - 用于为指定子集（B1/B2）预构建 RL 预测缓存
可在训练中断后手动执行，以便后续轮次快速加载

用法：
    python -m experiments.autotune.build_cache --dataset melbourne_temp --horizon 12 --subset B2 --resume llog/cs2
"""

import os
import sys
import argparse
import pickle
import pandas as pd

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from experiments.autotune.utils import ProgressLogger, load_config
from experiments.autotune.skill_policy import SkillPolicy
from experiments.autotune.tuner_train import build_rl_cache
from experiments.autotune.checkpoint_manager import CheckpointManager


def get_subset_data(config: dict, dataset_name: str, subset: str) -> pd.DataFrame:
    """
    根据配置和数据集名称，返回指定子集的【训练】窗口 DataFrame（已排除测试集）

    subset: 'B1' 或 'B2'

    测试集划分逻辑（与 tuner_core.py 一致）：
        - 每个 B 子集的前 1/3 窗口被抽取为测试集
        - 剩余 2/3 窗口用于演化训练
        - B1: 76 窗口 → 25 测试 + 51 训练
        - B2: 76 窗口 → 25 测试 + 51 训练
    """
    # 加载收集的窗口 CSV
    output_dir = config.get('output_dir', 'storage/autotune_results')
    csv_path = os.path.join(output_dir, "collected_windows.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"收集的窗口文件不存在: {csv_path}")

    df = pd.read_csv(csv_path)

    # 根据配置划分 A 和 B
    first_round_ratio = config.get('data_split', {}).get('first_round_ratio', 0.5)
    b_subset_count = config.get('evolution', {}).get('b_subset_count', 2)

    # 按索引排序，确保划分一致性
    df = df.sort_values('window_id').reset_index(drop=True)
    total = len(df)
    split_point = int(total * first_round_ratio)

    # A 部分（第1轮归纳用）
    df_a = df.iloc[:split_point].copy()
    # B 部分（演化训练用）
    df_b = df.iloc[split_point:].copy()

    # 将 B 划分为多个子集（顺序划分）
    b_size = len(df_b) // b_subset_count
    b_subsets = []
    for i in range(b_subset_count):
        start = i * b_size
        end = (i + 1) * b_size if i < b_subset_count - 1 else len(df_b)
        b_subsets.append(df_b.iloc[start:end].copy())

    # 选择对应的子集
    if subset.upper() == 'B1':
        df_subset = b_subsets[0]
    elif subset.upper() == 'B2':
        if len(b_subsets) < 2:
            raise ValueError("配置中 b_subset_count 小于 2，无法选择 B2")
        df_subset = b_subsets[1]
    else:
        raise ValueError(f"不支持的子集: {subset}，请使用 B1 或 B2")

    # ★★★ 关键修正：排除测试集（每个子集的前 1/3） ★★★
    # 测试集比例固定为 1/3（与 tuner_core.py 中的逻辑一致）
    test_ratio = 1.0 / 3.0
    test_count = int(len(df_subset) * test_ratio)

    # 训练集 = 跳过前 test_count 个窗口（与 tuner_core.py 一致）
    # 在 tuner_core.py 中：测试集从每个 B 子集的前 1/3 抽取，剩余用于演化
    df_train = df_subset.iloc[test_count:].copy()

    return df_train


def main():
    parser = argparse.ArgumentParser(description="手动构建 RL 预测缓存")
    parser.add_argument('--dataset', required=True, help="数据集名称")
    parser.add_argument('--horizon', type=int, required=True, help="预测步长")
    parser.add_argument('--subset', required=True, choices=['B1', 'B2'], help="要构建缓存的子集")
    parser.add_argument('--resume', required=True, help="运行目录（如 llog/cs2）")
    parser.add_argument('--workers', type=int, default=8, help="并行线程数")
    args = parser.parse_args()

    # 设置日志
    logger = ProgressLogger(log_dir=args.resume, verbose=True, run_folder=False)
    logger.log("=" * 70)
    logger.log(f"🚀 手动缓存构建工具")
    logger.log(f"📁 运行目录: {args.resume}")
    logger.log(f"📊 数据集: {args.dataset}, horizon={args.horizon}, 子集={args.subset}")
    logger.log("=" * 70)

    # 加载配置
    config = load_config()

    # 从检查点加载策略
    checkpoint_path = os.path.join(args.resume, "checkpoint.json")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")

    # ★★★ 修正：传入 logger ★★★
    cm = CheckpointManager(args.resume, logger)
    checkpoint = cm.load()
    if checkpoint is None:
        raise ValueError("无法加载检查点")

    policies_data = checkpoint.get('current_policies', [])
    if not policies_data:
        raise ValueError("检查点中没有策略数据")

    # ★★★ 兼容两种格式：字典列表 或 SkillPolicy 对象列表 ★★★
    if isinstance(policies_data[0], dict):
        policies = [SkillPolicy.from_dict(p) for p in policies_data]
    else:
        # 假设已经是 SkillPolicy 对象列表
        policies = policies_data

    logger.log(f"📋 加载策略: {len(policies)} 条")

    # 获取子集训练数据（已排除测试集）
    df_subset = get_subset_data(config, args.dataset, args.subset)
    logger.log(f"📊 子集 {args.subset} 训练窗口: {len(df_subset)} 个（已排除测试集）")

    if len(df_subset) == 0:
        raise ValueError(f"子集 {args.subset} 没有训练窗口，请检查数据划分配置")

    # 构建缓存
    logger.log(f"⏳ 开始构建缓存（并行 workers={args.workers}）...")
    cache = build_rl_cache(policies, df_subset, args.horizon, logger, workers=args.workers)

    # 保存缓存
    subset_lower = args.subset.lower()
    cache_file = os.path.join(args.resume, f"rl_cache_{subset_lower}.pkl")
    with open(cache_file, 'wb') as f:
        pickle.dump(cache, f)

    logger.log(f"✅ 缓存构建完成，共 {len(cache)} 项")
    logger.log(f"💾 已保存到: {cache_file}")
    logger.log("=" * 70)


if __name__ == '__main__':
    main()