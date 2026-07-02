"""SPLS AutoTuner - 补丁相关方法
★ 2026-06-25 重写：补丁策略聚类后直接加入全局池，不再进行全局平均评估过滤
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime

from experiments.autotune.utils import load_window_data, compute_mase
from experiments.autotune.skill_policy import SkillPolicy
from experiments.autotune.policy_graph import PolicyGraph


def patch_trouble_windows(logger, inducer, config: Dict, policies: List[SkillPolicy],
                           dataset_name: str, horizon: int):
    """
    针对困难窗口池生成补丁策略，聚类后直接加入全局池（状态=TRIAL）
    不再做全局平均 MASE 过滤，让系统后续通过状态匹配自然选择。
    """
    trouble_cfg = config.get('trouble_patch', {})
    if not trouble_cfg.get('enabled', True):
        logger.log("ℹ️ 事后补丁已禁用")
        return

    trouble_pool = inducer.get_trouble_pool()
    if not trouble_pool:
        logger.log("ℹ️ 没有困难窗口，跳过事后补丁")
        return

    min_trouble = trouble_cfg.get('min_trouble_windows', 2)
    if len(trouble_pool) < min_trouble:
        logger.log(f"ℹ️ 困难窗口数 ({len(trouble_pool)}) < {min_trouble}，跳过事后补丁")
        return

    logger.log(f"\n{'=' * 70}")
    logger.log(f"🔧 事后补丁：针对 {len(trouble_pool)} 个困难窗口生成补丁策略")
    logger.log(f"   策略：聚类后直接加入全局池（状态=TRIAL）")
    logger.log(f"{'=' * 70}")

    # 构建困难窗口 DataFrame
    trouble_dfs = []
    for w in trouble_pool:
        path = w.get('window_data_path')
        if path and os.path.exists(path):
            trouble_dfs.append({
                'window_id': w.get('window_id'),
                'origin': w.get('origin'),
                'window_size': w.get('window_size'),
                'window_data_path': path,
                'best_mase': w.get('mase')
            })

    if not trouble_dfs:
        logger.log("⚠️ 无法加载任何困难窗口数据，跳过补丁")
        return

    trouble_df = pd.DataFrame(trouble_dfs)
    logger.log(f"📊 成功加载 {len(trouble_df)} 个困难窗口")

    # ★★★ 核心修改：调用 inducer，获得聚类后的代表策略 ★★★
    try:
        result = inducer.induce(trouble_df, force_regenerate=True)
        new_policies_data = result.get('policies', [])
        new_policies = [SkillPolicy.from_dict(p) for p in new_policies_data]
        new_graph_data = result.get('policy_graph')
    except Exception as e:
        logger.log(f"   ⚠️ 补丁策略生成失败: {e}")
        import traceback
        traceback.print_exc()
        return

    if not new_policies:
        logger.log("   ℹ️ 未生成补丁策略")
        return

    # ★★★ 直接加入全局策略池，不再做全局评估过滤 ★★★
    logger.log(f"   📋 生成 {len(new_policies)} 个聚类代表策略，直接加入全局池")
    if new_graph_data:
        new_graph = PolicyGraph.from_dict(new_graph_data)
        logger.log(f"   包含 {len(new_graph.clusters)} 个簇")

    added = 0
    existing_ids = {p.policy_id for p in policies}

    for candidate in new_policies:
        # 检查是否已存在相同 ID（避免重复）
        if candidate.policy_id in existing_ids:
            logger.log(f"      ⚠️ 策略 {candidate.policy_id} 已存在，跳过")
            continue

        # ★ 标记为 TRIAL 状态（给后续演化轮次处理，如果还有的话）
        candidate.status = 'TRIAL'
        # 重命名，便于追踪来源
        if not candidate.name.startswith('patch_'):
            candidate.name = f"patch_{candidate.name}"
        candidate.created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # 确保 avg_mase 有效
        if candidate.avg_mase is None or np.isnan(candidate.avg_mase) or np.isinf(candidate.avg_mase):
            candidate.avg_mase = 1.0
            candidate.error_mean = 1.0

        policies.append(candidate)
        added += 1
        existing_ids.add(candidate.policy_id)
        logger.log(f"      ✅ 加入补丁策略: {candidate.name} (avg_mase={candidate.avg_mase:.4f}, cluster={candidate.cluster_id})")

    if added > 0:
        logger.log(f"   ✅ 事后补丁完成，新增 {added} 条策略（已聚类，直接进入全局池）")
        # ★ 清空困难池，避免重复处理
        inducer.clear_trouble_pool()
    else:
        logger.log("   ℹ️ 没有新策略加入（可能全部与现有策略重复）")


def evaluate_policy_on_df(policy: SkillPolicy, df: pd.DataFrame) -> Optional[float]:
    """评估策略在DataFrame窗口上的平均MASE（保留供其他用途）"""
    from experiments.autotune.utils import load_window_data, compute_mase
    mases = []

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

            pred = policy.execute(train, horizon, period)
            if pred is not None and len(pred) == len(test):
                mase = compute_mase(pred, test, mase_scale)
                mases.append(mase)
        except Exception:
            continue

    if not mases:
        return None

    return np.mean(mases)