"""SPLS AutoTuner - 工具方法"""

import os
import json
import hashlib
from typing import Dict, List, Tuple, Optional

from experiments.autotune.utils import format_weight_for_display
from src.agents.llm_client import LLMClient


def get_policies_hash(policies) -> str:
    """计算策略池的简单哈希，用于检测是否变化"""
    ids = sorted([p.policy_id for p in policies])
    return hashlib.md5(''.join(ids).encode()).hexdigest()


def save_round_policies(llog_dir: str, round_num: int, policies: List, version: str):
    """保存某轮策略"""
    round_dir = os.path.join(llog_dir, f"round_{round_num}")
    os.makedirs(round_dir, exist_ok=True)
    if version == "raw":
        fname = "refined_policies_raw.json"
    else:
        fname = "refined_policies_optimized.json"
    path = os.path.join(round_dir, fname)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({'policies': [p.to_dict() for p in policies]}, f, ensure_ascii=False, indent=2)


def log_config_summary(logger, config: Dict, num_rounds: int):
    """打印配置摘要"""
    policy_cfg = config.get('policy_pool', {})
    train_cfg = config.get('training', {})
    split_cfg = config.get('data_split', {})
    evo_cfg = config.get('evolution', {})
    trouble_cfg = config.get('trouble_patch', {})
    parallel_cfg = config.get('parallel', {})

    logger.log("\n📋 策略池配置:")
    logger.log(f"   min: {policy_cfg.get('min_policies', 5)}")
    logger.log(f"   target: {policy_cfg.get('target_policies', 8)}")
    logger.log(f"   soft_max: {policy_cfg.get('soft_max', 10)}")
    logger.log(f"   hard_max: {policy_cfg.get('hard_max', 15)}")

    logger.log(f"\n📋 训练配置:")
    logger.log(f"   轮数: {num_rounds}")
    logger.log(f"   自动优化: {train_cfg.get('auto_optimize', True)}")

    logger.log(f"\n📋 数据划分配置:")
    logger.log(f"   第一轮归纳比例: {split_cfg.get('first_round_ratio', 0.50)}")
    logger.log(f"   B部分比例: {split_cfg.get('B_ratio', 0.50)}")
    logger.log(f"   B子集数: {evo_cfg.get('b_subset_count', 4)}")
    logger.log(f"   B子集顺序: {evo_cfg.get('b_subset_indices', [2,3,0])}")
    logger.log(f"   测试模式: {split_cfg.get('test_mode', 'custom')}")

    logger.log(f"\n📋 事后补丁配置:")
    logger.log(f"   启用: {trouble_cfg.get('enabled', True)}")
    logger.log(f"   MASE阈值: {trouble_cfg.get('mase_threshold', 1.0)}")
    logger.log(f"   提升阈值: {trouble_cfg.get('improvement_threshold', 0.08):.0%}")

    logger.log(f"\n📋 并行配置:")
    logger.log(f"   测试并行: {parallel_cfg.get('test_parallel', True)}")
    logger.log(f"   测试线程数: {parallel_cfg.get('test_workers', 4)}")


def detect_available_model(logger, model_list: list) -> Tuple[Optional[str], Optional[str]]:
    """检测可用模型，返回 (model_name, error)"""
    logger.log("   🔍 正在检测可用模型...")

    for model_name in model_list:
        logger.log(f"     测试 {model_name}...")
        try:
            from src.agents.llm_client import LLMClient
            test_client = LLMClient(model=model_name, verbose=False)
            resp = test_client.call_with_retry("请回复'OK'", max_retries=1)

            if resp and resp.choices and resp.choices[0].message.content:
                content = resp.choices[0].message.content.strip()
                if content:
                    logger.log(f"      ✅ 模型 {model_name} 可用 (响应: {content[:20]})")
                    return model_name, None
                else:
                    logger.log(f"      ⚠️ 模型 {model_name} 返回空内容")
            else:
                logger.log(f"      ⚠️ 模型 {model_name} 响应异常")
        except Exception as e:
            logger.log(f"      ❌ 模型 {model_name} 不可用: {e}")

    logger.log("\n   📋 所有模型均不可用")
    return None, "所有模型均不可用"


def print_policy_trajectory(logger, policy, prefix="📌 当前策略"):
    """打印策略轨迹"""
    if not policy:
        logger.log(f"   {prefix}: 无策略")
        return

    stages = policy.skill_strategy.get('stages', [])
    logger.log(f"   {prefix}: {policy.name}")
    logger.log(f"      ID: {policy.policy_id}")
    logger.log(f"      Feature Groups: {policy.feature_groups}")

    if stages:
        logger.log(f"      📊 策略组合:")
        for j, stage in enumerate(stages):
            steps = stage.get('steps', 0)
            weights = stage.get('weights', {})
            w_str = ', '.join([f"{k}:{format_weight_for_display(v)}" for k, v in weights.items()])
            logger.log(f"         阶段{j+1}: {steps}步 → {{{w_str}}}")
    else:
        logger.log(f"      ⚠️ 无策略组合")

    if policy.state_condition:
        cond_str = ' AND '.join([f"{k} {v}" for k, v in policy.state_condition.items()])
        logger.log(f"      🎯 条件: {cond_str}")
    else:
        logger.log(f"      🎯 条件: 通用")

    logger.log(f"      📈 性能: avg_mase={policy.avg_mase:.6f}, utility={policy.utility_ema:.6f}")


def print_file_manifest(logger, llog_dir: str):
    """打印文件清单"""
    logger.log("\n" + "=" * 70)
    logger.log("📁 本次执行产生的文件:")
    logger.log("=" * 70)

    if os.path.exists(llog_dir):
        for item in sorted(os.listdir(llog_dir)):
            item_path = os.path.join(llog_dir, item)
            if os.path.isdir(item_path):
                logger.log(f"   📁 {item}/")
                for f in sorted(os.listdir(item_path)):
                    f_path = os.path.join(item_path, f)
                    if os.path.isfile(f_path):
                        size = os.path.getsize(f_path)
                        size_str = f"{size:,} bytes" if size < 1024 else f"{size/1024:.1f} KB"
                        logger.log(f"      ✅ {f} ({size_str})")
            else:
                size = os.path.getsize(item_path)
                size_str = f"{size:,} bytes" if size < 1024 else f"{size/1024:.1f} KB"
                logger.log(f"   ✅ {item} ({size_str})")

    logger.log("=" * 70)