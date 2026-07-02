# experiments/autotune/tuner_train.py
"""
SPLS AutoTuner - 训练循环（第1轮归纳 + 后续轮次演化）
★ 增加全局预测缓存构建（一次性，后续秒读）
★ 动态学习率自适应（根据 MASE 趋势调整）
★ ★ ★ 2026-06-28 将缓存传递给 evolver（质量门控）
★ ★ ★ 2026-06-28 设置 loop.current_round（试用期冻结）
★ ★ ★ ★ 2026-06-29 修复增量更新逻辑：先保存旧策略ID，再计算新增策略，仅对新策略构建缓存
★ ★ ★ ★ ★ 2026-06-29 增加缓存完整性检查：确保所有策略都在缓存中，缺失则自动补充
★ ★ ★ ★ ★ ★ 2026-06-29 增加智能跳过已完成 Re-Induction 的轮次，直接从缓存更新和 RL 训练开始
★ ★ ★ ★ ★ ★ ★ 2026-07-01 按子集分离缓存（B1/B2），支持手动构建和自动加载
★ ★ ★ ★ ★ ★ ★ ★ 2026-07-01 改用 ThreadPoolExecutor 避免 Windows 内存问题，线程数从配置读取
★ ★ ★ ★ ★ ★ ★ ★ ★ 2026-07-01 兼容旧版 rl_cache.pkl：若新缓存不存在，自动从旧缓存迁移（避免重复构建）
★ ★ ★ ★ ★ ★ ★ ★ ★ ★ 2026-07-01 增加断点续传：增量缓存构建中断后可从断点继续
★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ 2026-07-01 关键修复：Re-Induction 完成后立即保存检查点，防止中断后策略丢失
★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ 2026-07-01 关键修复2：保存检查点时保存 policies 副本，避免引用问题
★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ 2026-07-XX 每轮开始前清理困难池中的已解决窗口
★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ 2026-07-XX 修复增量缓存重复构建：增量更新时先加载完整缓存文件，避免重复计算
★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ 2026-07-XX 增加子阶段级别断点续传：增量缓存构建完成后保存检查点，中断后可从缓存构建完成状态继续
★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ 2026-07-XX 清除 .partial 断点续跑机制，中断后直接从本轮缓存构建环节开始，进度条修正
★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ 2026-07-XX 增加 skip_incremental_cache_for_recovery 配置：跳过增量缓存构建，直接进入 RL 训练（用于恢复被卡住的轮次）
★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ 2026-07-01 关键修复：退休机制执行后立即保存 checkpoint，确保 DEPRECATED 状态持久化（已移除退休）
★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ 2026-07-01 增加醒目的退休统计日志（已移除）
★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ 2026-08-XX 移除复活策略冻结期初始化（复活策略不冻结）
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Set, Tuple
from tqdm import tqdm
import pickle
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
import threading
import traceback

from experiments.autotune.skill_policy import SkillPolicy
from experiments.autotune.tuner_utils import save_round_policies
from experiments.autotune.utils import load_window_data, compute_mase
from experiments.autotune.policy_graph import PolicyGraph


# ★★★ 顶层函数：供线程池调用 ★★★
def _process_cache_task(task):
    """
    处理单个缓存任务，供线程池调用
    task: (policy, wpath, horizon_val)
    ★★★ 增加详细错误日志 ★★★
    """
    policy, wpath, horizon_val = task
    policy_name = policy.name[:20] if policy.name else "unknown"
    try:
        print(f"      🔄 [任务开始] {policy_name} @ {os.path.basename(wpath)}")
        
        wdata = load_window_data(wpath)
        train = wdata['train']
        test = wdata['test']
        period = wdata.get('period', 365)
        horizon = wdata.get('horizon', horizon_val)
        mase_scale = wdata.get('mase_scale', 1.0)

        pred = policy.execute(train, horizon, period)
        if pred is not None and len(pred) == len(test):
            mase = compute_mase(pred, test, mase_scale)
            print(f"      ✅ [任务完成] {policy_name} @ {os.path.basename(wpath)} MASE={mase:.4f}")
            return (policy.policy_id, wpath), {'pred': pred, 'mase': mase}
        else:
            pred_len = len(pred) if pred is not None else 0
            print(f"      ⚠️ [任务失败] {policy_name} @ {os.path.basename(wpath)} 预测长度不匹配 (pred={pred_len}, test={len(test)})")
            return None
    except Exception as e:
        print(f"      ❌ [任务异常] {policy_name} @ {os.path.basename(wpath)}")
        print(f"         {type(e).__name__}: {e}")
        print(f"         {traceback.format_exc()[:500]}")
        return None


def build_rl_cache(policies: List[SkillPolicy], df_b: pd.DataFrame,
                   horizon_val: int, logger, workers: int = 24,
                   cache_file: Optional[str] = None) -> Dict:
    """
    构建 RL 预测缓存：(policy_id, window_data_path) -> {'pred': array, 'mase': float}
    使用线程池并行加速（避免 Windows 下多进程内存爆炸）。
    """
    base_cache = {}
    if cache_file is not None and os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                base_cache = pickle.load(f)
            logger.log(f"   📂 加载已有完整缓存: {len(base_cache)} 项")
        except Exception as e:
            logger.log(f"   ⚠️ 加载已有缓存失败: {e}，将重新构建")
            base_cache = {}

    tasks = []
    task_key_set = set()
    
    for _, row in df_b.iterrows():
        wpath = row.get('window_data_path')
        if not wpath or not os.path.exists(wpath):
            continue
        for policy in policies:
            if policy.status in ['ARCHIVE', 'DELETE']:
                continue
            key = (policy.policy_id, wpath)
            if key not in task_key_set:
                task_key_set.add(key)
                tasks.append((policy, wpath, horizon_val))

    if not tasks:
        logger.log("   ⚠️ 无可用任务，缓存为空")
        return {}

    remaining_tasks = []
    for task in tasks:
        key = (task[0].policy_id, task[1])
        if key not in base_cache:
            remaining_tasks.append(task)

    if not remaining_tasks:
        logger.log(f"   ✅ 所有任务已完成 (共 {len(base_cache)} 项)")
        return base_cache

    base_count = len(base_cache)
    total_remaining = len(remaining_tasks)
    total_all = len(tasks)
    logger.log(f"   📦 总任务数: {total_all}, 已有缓存: {base_count}, 需要计算: {total_remaining}")

    cache = base_cache.copy()
    completed_count = 0
    
    TASK_TIMEOUT = 120
    failed_tasks = []
    success_count = 0

    logger.log(f"   ⚡ 开始并行计算（{workers} 线程，每个任务超时 {TASK_TIMEOUT}s）...")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_task = {executor.submit(_process_cache_task, task): task for task in remaining_tasks}
        
        pbar = tqdm(total=total_remaining, desc="   构建缓存进度", unit="项", ncols=100)
        
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            policy, wpath, horizon_val = task
            policy_name = policy.name[:20] if policy.name else "unknown"
            wpath_basename = os.path.basename(wpath)
            
            try:
                result = future.result(timeout=TASK_TIMEOUT)
                if result is not None:
                    key, value = result
                    cache[key] = value
                    success_count += 1
                    completed_count += 1
                    pbar.set_postfix({
                        '成功': success_count,
                        '失败': len(failed_tasks),
                        '总缓存': len(cache)
                    })
                else:
                    failed_tasks.append((policy_name, wpath_basename, "返回None"))
                    pbar.set_postfix({
                        '成功': success_count,
                        '失败': len(failed_tasks),
                        '总缓存': len(cache)
                    })
            except TimeoutError:
                future.cancel()
                failed_tasks.append((policy_name, wpath_basename, f"超时({TASK_TIMEOUT}s)"))
                logger.log(f"      ❌ [超时] {policy_name} @ {wpath_basename} 超过 {TASK_TIMEOUT}s，跳过")
                pbar.set_postfix({
                    '成功': success_count,
                    '失败': len(failed_tasks),
                    '总缓存': len(cache)
                })
            except Exception as e:
                failed_tasks.append((policy_name, wpath_basename, f"{type(e).__name__}: {e}"))
                logger.log(f"      ❌ [异常] {policy_name} @ {wpath_basename}: {type(e).__name__}: {e}")
                pbar.set_postfix({
                    '成功': success_count,
                    '失败': len(failed_tasks),
                    '总缓存': len(cache)
                })
            
            pbar.update(1)
        pbar.close()

    if failed_tasks:
        logger.log(f"   ⚠️ 有 {len(failed_tasks)} 个任务失败:")
        for fname, fpath, reason in failed_tasks[:10]:
            logger.log(f"      - {fname} @ {fpath}: {reason}")
        if len(failed_tasks) > 10:
            logger.log(f"      ... 还有 {len(failed_tasks) - 10} 个失败")

    logger.log(f"   ✅ 缓存构建完成，共 {len(cache)} 项（本次新增 {success_count} 项）")
    return cache


def run_training_loop(logger, loop, collector, inducer, evolver, checkpoint_manager,
                       config: Dict, policies: List[SkillPolicy],
                       df_a, df_b, all_b_subsets, b_subset_indices,
                       name: str, horizon_val: int,
                       completed_rounds: int, current_b_subset_idx: int,
                       num_rounds: int, llog_dir: str,
                       auto_optimize: bool, save_all_rounds: bool,
                       round_results: Dict, policy_snapshots: Dict,
                       b_subsets: List,
                       policy_graph: Optional[PolicyGraph] = None) -> List[SkillPolicy]:
    """
    执行训练循环（第1轮归纳 + 后续演化轮次）
    返回最终的 policies
    """
    # ★★★ 读取自动构建缓存配置（默认 True） ★★★
    auto_build_cache = config.get('auto_build_cache', True)

    # ★★★ ★★★ ★★★ 读取跳过增量缓存恢复开关 ★★★ ★★★ ★★★
    skip_incremental_cache = config.get('skip_incremental_cache_for_recovery', False)
    if skip_incremental_cache:
        logger.log(f"   ⚠️ [恢复模式] skip_incremental_cache_for_recovery=True，将跳过增量缓存构建，直接进入 RL 训练")

    # ★★★ 读取缓存构建线程数（默认 24） ★★★
    cache_workers = config.get('parallel', {}).get('cache_workers', 24)

    start_round = completed_rounds + 1

    logger.log(f"\n{'═' * 70}")
    logger.log(f"  🔄 从第 {start_round} 轮继续训练（共 {num_rounds} 轮）")
    logger.log(f"  📋 策略池当前规模: {len(policies)} 条策略")
    logger.log(f"  📦 缓存构建线程数: {cache_workers}")
    if skip_incremental_cache:
        logger.log(f"  ⚠️ 增量缓存构建已禁用（恢复模式）")
    logger.log(f"{'═' * 70}")

    base_lr = config.get('rl', {}).get('learning_rate', 0.002)

    # 第1轮：A归纳
    if start_round == 1 and not policies:
        if len(df_a) == 0:
            logger.log("⚠️ A部分为空，无法进行归纳")
            return policies

        logger.log("\n" + "█" * 70)
        logger.log("█  📍 阶段 1/2: 策略归纳（第 1 轮）")
        logger.log("█  ──────────────────────────────────────────────────────────────")
        logger.log("█  📊 A部分: {} 个窗口，每窗口生成 2 个候选策略".format(len(df_a)))
        logger.log("█" + "█" * 70)

        logger.log("🧠 第1轮：策略归纳 (A部分，每窗口2候选)...")
        result = inducer.induce(df_a)
        policies_data = result.get('policies', [])
        policies = [SkillPolicy.from_dict(p) for p in policies_data]
        loop.load_policies(policies)
        collector.set_policies(policies)
        logger.log(f"📋 归纳完成: {len(policies)} 条策略")

        # ★★★ 将新策略分配到簇 ★★★
        if policy_graph:
            policy_dict = {p.policy_id: p for p in policies}
            mases = [p.avg_mase for p in policies if p.avg_mase != float('inf')]
            global_avg = np.mean(mases) if mases else 1.0
            worst_cluster_avg = float('inf')
            if policy_graph.clusters:
                worst_cluster_avg = max(c.avg_mase for c in policy_graph.clusters if c.is_active)
            
            context = {
                'current_round': 1,
                'policy_dict': policy_dict,
                'global_avg_mase': global_avg,
                'worst_cluster_avg_mase': worst_cluster_avg,
                'logger': logger
            }
            for policy in policies:
                if policy.cluster_id is None:
                    policy_graph.add_policy(policy, context=context)
            logger.log(f"📊 策略簇分配完成: {len(policy_graph.clusters)} 个簇")

        save_round_policies(llog_dir, 1, policies, "raw")
        save_round_policies(llog_dir, 1, policies, "optimized")
        policy_snapshots['round_1'] = policies.copy()
        round_results['round_1'] = {
            'policies': policies.copy(),
            'avg_mase': 0,
            'improvement': 0,
            'policy_count': len(policies)
        }

        checkpoint_manager.save(
            completed_rounds=1,
            current_policies=policies.copy(),
            dataset=name,
            horizon=horizon_val,
            round_results=round_results,
            current_b_subset_idx=b_subset_indices[0] if b_subsets else 0,
            policy_graph=policy_graph.to_dict() if policy_graph else None
        )
        logger.log(f"💾 检查点已保存 (第 1 轮)")
        start_round = 2

    # ★★★ ★★★ ★★★ 将分布模型传递给 evolver ★★★ ★★★ ★★★
    if hasattr(evolver, 'set_distribution_model'):
        evolver.set_distribution_model(loop.distribution_model)
        logger.log("   📦 已为 evolver 设置分布模型")

    # ★★★ 旧缓存文件路径（兼容旧版） ★★★
    old_cache_file = os.path.join(llog_dir, "rl_cache.pkl")

    # ★★★ 保存一份策略池的快照引用，用于检测 Re-Induction 是否已完成 ★★★
    last_round_policy_ids = {p.policy_id for p in policies}

    # 第2-n轮：B演化 + RL训练
    for round_num in range(start_round, num_rounds + 1):
        loop.set_current_round(round_num)

        # ★★★ ★★★ ★★★ 每轮开始前清理困难池中的已解决窗口 ★★★ ★★★ ★★★
        if hasattr(inducer, 'trouble_pool') and inducer.trouble_pool:
            pool_before = len(inducer.trouble_pool)
            cleaned = []
            for item in inducer.trouble_pool[:]:
                mase = item.get('mase', 1.0)
                if mase <= 1.0:
                    cleaned.append(item.get('window_id'))
            if cleaned:
                logger.log(f"   🧹 [轮前清理] 发现 {len(cleaned)} 个已解决窗口待清理: {cleaned[:10]}{'...' if len(cleaned) > 10 else ''}")

        # ★★★ 移除复活策略冻结期初始化（复活策略不冻结） ★★★
        # 复活策略已经在 tuner_core 中设置 metadata['revived']=True，并在 spls_loop._is_policy_frozen 中跳过冻结

        # ★★★ 动态学习率自适应 ★★★
        if round_num > start_round:
            prev_avg = round_results.get(f'round_{round_num-1}', {}).get('rl_avg_mase', float('inf'))
            if prev_avg != float('inf') and prev_avg > 0:
                if round_num >= 3:
                    prev_prev_avg = round_results.get(f'round_{round_num-2}', {}).get('rl_avg_mase', float('inf'))
                    if prev_prev_avg != float('inf'):
                        if prev_avg < prev_prev_avg * 0.98:
                            base_lr *= 1.05
                            logger.log(f"   📈 性能提升，学习率上调至 {base_lr:.6f}")
                        elif prev_avg > prev_prev_avg * 1.02:
                            base_lr *= 0.9
                            logger.log(f"   📉 性能下降，学习率下调至 {base_lr:.6f}")

        decay_factor = 0.99 ** (round_num - 1)
        current_lr = base_lr * decay_factor
        current_lr = max(1e-5, current_lr)
        loop.distribution_model.set_learning_rate(current_lr)
        logger.log(f"   📉 第 {round_num} 轮学习率: {current_lr:.6f} (衰减因子 {decay_factor:.4f})")

        if not b_subsets:
            logger.log("⚠️ 无B子集可用于演化，跳过")
            break

        evo_idx = (round_num - 2) % len(b_subsets)
        b_idx = b_subset_indices[evo_idx]
        df_b_current = b_subsets[evo_idx]
        total_windows = len(df_b_current)

        # ★★★ 确定子集名称 ★★★
        subset_name = f"b{b_idx + 1}"
        cache_file = os.path.join(llog_dir, f"rl_cache_{subset_name}.pkl")

        logger.log("\n" + "█" * 70)
        logger.log(f"█  📍 阶段 2/3: RL 演化训练（第 {round_num} 轮）")
        logger.log(f"█  ──────────────────────────────────────────────────────────────")
        logger.log(f"█  📊 B{b_idx + 1}子集: {total_windows} 个窗口")
        logger.log(f"█  📋 当前策略数: {len(policies)} 条")
        logger.log(f"█  📈 训练进度: {round_num}/{num_rounds} ({round_num/num_rounds*100:.0f}%)")
        logger.log("█" + "█" * 70)

        logger.log("")
        logger.log("   ┌─────────────────────────────────────────────────────────────┐")
        logger.log("   │  📌 第 {} 轮演化优化 (B{}子集, {} 个窗口)                  │".format(round_num, b_idx + 1, total_windows))
        logger.log("   └─────────────────────────────────────────────────────────────┘")

        round_dir = os.path.join(llog_dir, f"round_{round_num}")
        os.makedirs(round_dir, exist_ok=True)

        # ==================== ★★★ ★★★ ★★★ 检查子阶段状态 ★★★ ★★★ ★★★ ====================
        pending_state = checkpoint_manager.get_pending_round_state()
        
        skip_reinduction = False
        skip_to_rl = False
        rebuild_cache_needed = True
        
        if pending_state and pending_state.get('round') == round_num:
            if pending_state.get('reinduction_done', False):
                logger.log(f"   🔄 [子阶段恢复] 第 {round_num} 轮 Re-Induction 已完成")
                skip_reinduction = True
                
                if pending_state.get('cache_built', False):
                    logger.log(f"   🔄 [子阶段恢复] 第 {round_num} 轮缓存已构建，直接执行 RL 训练")
                    skip_to_rl = True
                    rebuild_cache_needed = False
                else:
                    logger.log(f"   🔄 [子阶段恢复] 第 {round_num} 轮缓存未构建，执行缓存构建（跳过 Re-Induction）")
                    skip_to_rl = False
                    rebuild_cache_needed = True
            else:
                logger.log(f"   🔄 [子阶段状态] 第 {round_num} 轮 Re-Induction 未完成，正常执行")
                skip_reinduction = False
                skip_to_rl = False
                rebuild_cache_needed = True
        else:
            skip_reinduction = False
            skip_to_rl = False
            rebuild_cache_needed = True

        # ==================== ★★★ 加载/构建当前子集的缓存 ★★★ ====================
        rl_cache = {}

        if not os.path.exists(cache_file) and os.path.exists(old_cache_file):
            logger.log(f"   📂 检测到旧缓存 {old_cache_file}，自动迁移到 {cache_file} ...")
            try:
                with open(old_cache_file, 'rb') as f:
                    old_cache = pickle.load(f)
                with open(cache_file, 'wb') as f:
                    pickle.dump(old_cache, f)
                rl_cache = old_cache
                logger.log(f"   ✅ 缓存迁移成功！已从 {old_cache_file} 迁移到 {cache_file}")
                loop.set_cache(rl_cache)
                if hasattr(evolver, 'set_cache'):
                    evolver.set_cache(rl_cache)
            except Exception as e:
                logger.log(f"   ⚠️ 缓存迁移失败: {e}，将自动构建新缓存")

        if os.path.exists(cache_file) and not rl_cache:
            logger.log(f"   📂 加载已有 RL 缓存: {cache_file}")
            with open(cache_file, 'rb') as f:
                rl_cache = pickle.load(f)
            loop.set_cache(rl_cache)
            if hasattr(evolver, 'set_cache'):
                evolver.set_cache(rl_cache)
        elif not rl_cache and rebuild_cache_needed:
            if auto_build_cache:
                if skip_incremental_cache:
                    logger.log(f"   ⚠️ [恢复模式] skip_incremental_cache_for_recovery=True，跳过增量缓存构建")
                    rl_cache = {}
                    logger.log(f"   ⚠️ [恢复模式] 缓存为空，RL 训练时将按需实时计算")
                    
                    checkpoint_manager.save(
                        completed_rounds=round_num - 1,
                        current_policies=policies.copy(),
                        dataset=name,
                        horizon=horizon_val,
                        round_results=round_results,
                        current_b_subset_idx=b_idx,
                        policy_graph=checkpoint_manager._checkpoint.get('policy_graph') if checkpoint_manager._checkpoint else None,
                        a_eval_completed=True,
                        pending_round_state={
                            'round': round_num,
                            'reinduction_done': True,
                            'cache_built': True,
                            'rl_training_pending': True,
                            'cache_skipped': True
                        }
                    )
                    logger.log(f"   💾 [恢复模式] 已保存检查点（跳过缓存构建，直接进入 RL 训练）")
                    
                    loop.set_cache(rl_cache)
                    if hasattr(evolver, 'set_cache'):
                        evolver.set_cache(rl_cache)
                    
                    skip_to_rl = True
                    rebuild_cache_needed = False
                else:
                    logger.log(f"   ⚠️ 缓存 {cache_file} 不存在，自动构建（{cache_workers} 线程）...")
                    rl_cache = build_rl_cache(policies, df_b_current, horizon_val, logger, 
                                              workers=cache_workers, cache_file=cache_file)
                    with open(cache_file, 'wb') as f:
                        pickle.dump(rl_cache, f)
                    loop.set_cache(rl_cache)
                    if hasattr(evolver, 'set_cache'):
                        evolver.set_cache(rl_cache)
            else:
                logger.log(f"   ❌ 缓存 {cache_file} 不存在且 auto_build_cache=False，请手动构建缓存")
                logger.log(f"   💡 运行: python -m experiments.autotune.build_cache --subset {subset_name.upper()} --resume {llog_dir}")
                sys.exit(1)

        # ==================== 子阶段 A: Re-Induction ====================
        if skip_to_rl:
            logger.log("   ⏭️ [子阶段恢复] 跳过 Re-Induction（缓存已构建或已跳过）")
            evo_result = {'new_policies_added': 0, 'changes': []}
        else:
            if skip_reinduction:
                logger.log("   ⏭️ [子阶段恢复] 跳过 Re-Induction（已执行），直接进入缓存构建")
                evo_result = {'new_policies_added': 0, 'changes': []}
            else:
                trial_policies = [p for p in policies if p.status == 'TRIAL' and f"reind_{round_num}" in p.name]
                if trial_policies:
                    logger.log(f"   ✅ 检测到本轮 {round_num} 已有 Re-Induction 生成的新策略（{len(trial_policies)} 条），跳过重复执行。")
                    skip_reinduction = True
                else:
                    current_ids = {p.policy_id for p in policies}
                    if len(current_ids) > len(last_round_policy_ids):
                        new_ids = current_ids - last_round_policy_ids
                        if new_ids:
                            has_reind = any(p.policy_id in new_ids and f"reind_{round_num}" in p.name for p in policies)
                            if has_reind:
                                logger.log(f"   ✅ 检测到本轮 {round_num} 已有新增策略（{len(new_ids)} 条），跳过 Re-Induction。")
                                skip_reinduction = True
                            else:
                                skip_reinduction = False
                        else:
                            skip_reinduction = False
                    else:
                        skip_reinduction = False

                if auto_optimize and len(policies) > 0 and not skip_reinduction:
                    logger.log("")
                    logger.log("   ┌─────────────────────────────────────────────────────────────┐")
                    logger.log("   │  📍 子阶段 A: Re-Induction（策略生成与评估）              │")
                    logger.log("   └─────────────────────────────────────────────────────────────┘")
                    logger.log("   ⚡ 执行 Re-Induction...")
                    logger.log("   ──────────────────────────────────────────────────────────")

                    old_policy_ids = {p.policy_id for p in policies}

                    evo_result = evolver.run_round(policies, df_b_current, round_num, force=True)
                    policies = evo_result.get('policies', policies)
                    loop.load_policies(policies)
                    collector.set_policies(policies)

                    new_added = evo_result.get('new_policies_added', 0)
                    
                    # ★★★ 注意：退休已禁用，evo_result 中不再有 retired_count ★★★
                    if new_added > 0:
                        # ★★★ 为新策略分配簇 ★★★
                        if policy_graph:
                            policy_dict = {p.policy_id: p for p in policies}
                            mases = [p.avg_mase for p in policies if p.avg_mase != float('inf')]
                            global_avg = np.mean(mases) if mases else 1.0
                            worst_cluster_avg = float('inf')
                            if policy_graph.clusters:
                                worst_cluster_avg = max(c.avg_mase for c in policy_graph.clusters if c.is_active)
                            
                            context = {
                                'current_round': round_num,
                                'policy_dict': policy_dict,
                                'global_avg_mase': global_avg,
                                'worst_cluster_avg_mase': worst_cluster_avg,
                                'logger': logger
                            }
                            # 只分配新策略
                            for policy in policies:
                                if policy.policy_id in old_policy_ids:
                                    continue
                                if policy.cluster_id is None:
                                    policy_graph.add_policy(policy, context=context)
                            logger.log(f"📊 新策略簇分配完成")

                        logger.log(f"   💾 ★★★ 保存新增策略状态（新增 {new_added} 条）...")
                        checkpoint_manager.save(
                            completed_rounds=round_num - 1,
                            current_policies=policies.copy(),
                            dataset=name,
                            horizon=horizon_val,
                            round_results=round_results,
                            current_b_subset_idx=b_idx,
                            policy_graph=policy_graph.to_dict() if policy_graph else None,
                            a_eval_completed=True,
                            pending_round_state={
                                'round': round_num,
                                'reinduction_done': True,
                                'cache_built': False,
                                'rl_training_pending': True
                            }
                        )
                        logger.log(f"   ✅ 检查点已保存（新增状态已持久化）")
                        last_round_policy_ids = {p.policy_id for p in policies}

                    if new_added > 0 and not skip_incremental_cache:
                        logger.log(f"   📦 检测到新策略，增量更新缓存（只算新策略）...")
                        new_policies = [p for p in policies if p.policy_id not in old_policy_ids]
                        if new_policies:
                            logger.log(f"   📦 新增策略: {len(new_policies)} 条，开始增量构建缓存（{cache_workers} 线程）...")
                            new_cache = build_rl_cache(new_policies, df_b_current, horizon_val, logger, 
                                                       workers=cache_workers, cache_file=cache_file)
                            
                            logger.log(f"   🔍 [诊断] new_cache 大小: {len(new_cache)} 项")
                            if new_cache:
                                sample_keys = list(new_cache.keys())[:3]
                                logger.log(f"   🔍 [诊断] new_cache 键示例: {sample_keys}")
                            
                            rl_cache.update(new_cache)
                            logger.log(f"   🔍 [诊断] rl_cache 更新后大小: {len(rl_cache)} 项")
                            
                            with open(cache_file, 'wb') as f:
                                pickle.dump(rl_cache, f)
                            
                            loop.set_cache(rl_cache)
                            if hasattr(evolver, 'set_cache'):
                                evolver.set_cache(rl_cache)
                            logger.log(f"   ✅ 增量更新完成，新增 {len(new_cache)} 项缓存，总缓存 {len(rl_cache)} 项")

                            checkpoint_manager.save(
                                completed_rounds=round_num - 1,
                                current_policies=policies.copy(),
                                dataset=name,
                                horizon=horizon_val,
                                round_results=round_results,
                                current_b_subset_idx=b_idx,
                                policy_graph=policy_graph.to_dict() if policy_graph else None,
                                a_eval_completed=True,
                                pending_round_state={
                                    'round': round_num,
                                    'reinduction_done': True,
                                    'cache_built': True,
                                    'rl_training_pending': True
                                }
                            )
                            logger.log(f"   💾 ★★★ [子阶段检查点] 增量缓存构建完成，已保存状态（轮次={round_num}，待执行 RL 训练）")
                        else:
                            logger.log("   ⚠️ 虽然 new_added>0 但未找到新增策略，跳过缓存更新")
                    elif new_added > 0 and skip_incremental_cache:
                        logger.log(f"   ⚠️ [恢复模式] 跳过新增策略的缓存构建")
                        checkpoint_manager.save(
                            completed_rounds=round_num - 1,
                            current_policies=policies.copy(),
                            dataset=name,
                            horizon=horizon_val,
                            round_results=round_results,
                            current_b_subset_idx=b_idx,
                            policy_graph=policy_graph.to_dict() if policy_graph else None,
                            a_eval_completed=True,
                            pending_round_state={
                                'round': round_num,
                                'reinduction_done': True,
                                'cache_built': True,
                                'rl_training_pending': True,
                                'cache_skipped': True
                            }
                        )
                        logger.log(f"   💾 [恢复模式] 已保存检查点（跳过新增策略缓存构建）")

                    logger.log("   ──────────────────────────────────────────────────────────")
                    logger.log(f"   ✅ Re-Induction 完成: {len(policies)} 条策略 (新增 {new_added} 条)")
                else:
                    if skip_reinduction:
                        logger.log("   ⏭️ 跳过 Re-Induction（本轮已有新策略生成）")
                    else:
                        logger.log("⏭️ 跳过 Re-Induction")
                    evo_result = {'new_policies_added': 0, 'changes': []}

        # ==================== ★★★ 缓存完整性检查（当前子集） ★★★ ====================
        if not skip_to_rl and not skip_incremental_cache:
            cached_policy_ids = {pid for (pid, _) in rl_cache.keys()}
            current_policy_ids = {p.policy_id for p in policies}
            missing_ids = current_policy_ids - cached_policy_ids

            logger.log(f"   🔍 [诊断] cached_policy_ids: {len(cached_policy_ids)} 条策略")
            logger.log(f"   🔍 [诊断] current_policy_ids: {len(current_policy_ids)} 条策略")
            if missing_ids:
                logger.log(f"   🔍 [诊断] missing_ids: {missing_ids}")

            if missing_ids:
                logger.log(f"   📦 检测到 {len(missing_ids)} 条策略缺少缓存，进行增量补充（{cache_workers} 线程）...")
                missing_policies = [p for p in policies if p.policy_id in missing_ids]
                new_cache = build_rl_cache(missing_policies, df_b_current, horizon_val, logger, 
                                           workers=cache_workers, cache_file=cache_file)
                rl_cache.update(new_cache)
                with open(cache_file, 'wb') as f:
                    pickle.dump(rl_cache, f)
                loop.set_cache(rl_cache)
                if hasattr(evolver, 'set_cache'):
                    evolver.set_cache(rl_cache)
                logger.log(f"   ✅ 缓存补充完成，新增 {len(new_cache)} 项缓存，总缓存 {len(rl_cache)} 项")
                
                checkpoint_manager.save(
                    completed_rounds=round_num - 1,
                    current_policies=policies.copy(),
                    dataset=name,
                    horizon=horizon_val,
                    round_results=round_results,
                    current_b_subset_idx=b_idx,
                    policy_graph=policy_graph.to_dict() if policy_graph else None,
                    a_eval_completed=True,
                    pending_round_state={
                        'round': round_num,
                        'reinduction_done': True,
                        'cache_built': True,
                        'rl_training_pending': True
                    }
                )
                logger.log(f"   💾 [子阶段检查点] 缓存完整性检查完成，已保存状态（轮次={round_num}，待执行 RL 训练）")
            else:
                logger.log(f"   ✅ 所有策略均已缓存，无需补充")
        elif skip_to_rl and skip_incremental_cache:
            logger.log(f"   ⏭️ [恢复模式] 跳过缓存完整性检查（缓存已跳过构建）")

        # ==================== 子阶段 B: 强制 RL 在线训练 ====================
        logger.log("")
        logger.log("   ┌─────────────────────────────────────────────────────────────┐")
        logger.log(f"   │  📍 子阶段 B: RL 在线训练（B{b_idx + 1}子集，{total_windows} 个窗口） │")
        logger.log("   │  ⚡ 铁定执行，无论是否触发 Re-Induction                   │")
        logger.log("   └─────────────────────────────────────────────────────────────┘")

        rl_success_count = 0
        rl_fail_count = 0
        rl_mases = []
        rl_start_time = time.time()

        pbar = tqdm(
            total=total_windows,
            desc=f"   🧠 RL 训练进度",
            unit="窗口",
            ncols=120,
            position=0,
            leave=True
        )

        for idx, row in df_b_current.iterrows():
            window_data_path = row.get('window_data_path')
            if not window_data_path or not os.path.exists(window_data_path):
                rl_fail_count += 1
                pbar.set_postfix({
                    '状态': '⚠️ 路径不存在',
                    '成功': rl_success_count,
                    '失败': rl_fail_count
                })
                pbar.update(1)
                continue

            try:
                wdata = load_window_data(window_data_path)
                train = wdata['train']
                test = wdata['test']
                horizon = wdata.get('horizon', horizon_val)

                result = loop.step_with_ground_truth(
                    observation=train,
                    horizon=horizon,
                    ground_truth=test,
                    split='B',
                    window_data_path=window_data_path
                )

                mase = result.get('mase', float('inf'))
                if mase != float('inf') and not np.isnan(mase):
                    rl_mases.append(mase)
                    rl_success_count += 1
                else:
                    rl_fail_count += 1

                if (idx + 1) % 5 == 0 or idx + 1 == total_windows:
                    avg_mase = np.mean(rl_mases) if rl_mases else 0.0
                    pbar.set_postfix({
                        '窗口': f'{idx + 1}/{total_windows}',
                        '成功': rl_success_count,
                        '失败': rl_fail_count,
                        '平均MASE': f'{avg_mase:.4f}'
                    })

            except Exception as e:
                rl_fail_count += 1
                logger.log(f"      ⚠️ RL 训练窗口 {idx} 异常: {e}")
                pbar.set_postfix({
                    '状态': f'❌ {str(e)[:20]}',
                    '成功': rl_success_count,
                    '失败': rl_fail_count
                })

            pbar.update(1)

        pbar.close()

        checkpoint_manager.save(
            completed_rounds=round_num,
            current_policies=policies.copy(),
            dataset=name,
            horizon=horizon_val,
            round_results=round_results,
            current_b_subset_idx=b_idx,
            policy_graph=policy_graph.to_dict() if policy_graph else None,
            a_eval_completed=True,
            pending_round_state=None
        )
        logger.log(f"💾 RL训练完成，检查点已保存 (第 {round_num} 轮，completed_rounds={round_num})")
        
        last_round_policy_ids = {p.policy_id for p in policies}

        rl_elapsed = time.time() - rl_start_time
        rl_avg_mase = np.mean(rl_mases) if rl_mases else float('inf')
        logger.log("")
        logger.log("   ──────────────────────────────────────────────────────────")
        logger.log(f"   ✅ RL 在线训练完成!")
        logger.log(f"      📊 成功: {rl_success_count}/{total_windows} 个窗口")
        logger.log(f"      ❌ 失败: {rl_fail_count}/{total_windows} 个窗口")
        logger.log(f"      📈 平均 MASE: {rl_avg_mase:.6f}" if rl_avg_mase != float('inf') else "      📈 平均 MASE: N/A")
        logger.log(f"      ⏱️  耗时: {rl_elapsed:.1f} 秒")

        if hasattr(loop, 'distribution_model') and loop.distribution_model:
            theta_dict = loop.distribution_model.theta
            if theta_dict:
                theta_str = ", ".join([f"{pid[:6]}:{theta:.4f}" for pid, theta in list(theta_dict.items())[:5]])
                logger.log(f"      🎯 θ 分布 (前5): {{{theta_str}{'...' if len(theta_dict) > 5 else ''}}}")

        if save_all_rounds:
            save_round_policies(llog_dir, round_num, policies, "optimized")
            raw_path = os.path.join(round_dir, "refined_policies_raw.json")
            if not os.path.exists(raw_path):
                save_round_policies(llog_dir, round_num, policies, "raw")

        policy_snapshots[f'round_{round_num}'] = policies.copy()
        round_results[f'round_{round_num}'] = {
            'policies': policies.copy(),
            'policy_count': len(policies),
            'rl_avg_mase': rl_avg_mase,
            'rl_success_count': rl_success_count,
            'rl_total_windows': total_windows
        }

        checkpoint_manager.save(
            completed_rounds=round_num,
            current_policies=policies.copy(),
            dataset=name,
            horizon=horizon_val,
            round_results=round_results,
            current_b_subset_idx=b_idx,
            policy_graph=policy_graph.to_dict() if policy_graph else None,
            pending_round_state=None
        )
        logger.log(f"💾 检查点已保存 (第 {round_num} 轮, B{b_idx + 1})")

        progress = round_num / num_rounds * 100
        logger.log(f"   📊 训练进度: {progress:.0f}% ({round_num}/{num_rounds})")

        logger.log("\n" + "─" * 70)
        logger.log(f"  ✅ 第 {round_num} 轮完成")
        logger.log(f"  📋 策略数: {len(policies)} 条")
        if evo_result.get('new_policies_added', 0) > 0:
            logger.log(f"  ★ 新增 {evo_result.get('new_policies_added', 0)} 条策略")
        # ★★★ 移除退休日志（退休已禁用） ★★★
        logger.log(f"  🧠 RL 平均 MASE: {rl_avg_mase:.6f}" if rl_avg_mase != float('inf') else "  🧠 RL 平均 MASE: N/A")
        logger.log("─" * 70)

    logger.log("\n" + "█" * 70)
    logger.log("█  🎉 训练完成！")
    logger.log("█  ──────────────────────────────────────────────────────────────")
    logger.log(f"█  📋 最终策略数: {len(policies)} 条")
    logger.log(f"█  🔄 总轮数: {num_rounds} 轮")
    logger.log("█" * 70)

    return policies