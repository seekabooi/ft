# experiments/autotune/iterative_refiner_utils.py
"""
Policy Evolution Engine - 独立运行和工具方法
包含：run() 独立运行模式、_identify_hard_windows_with_mases 缓存读取
★ ★ ★ 2026-07-XX 修复缓存写入路径：强制写入当前运行目录，而非扫描到的第一个目录
"""

import os
import sys
import json
import hashlib
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime
import time
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from experiments.autotune.utils import load_window_data, compute_mase, extract_features
from experiments.autotune.skill_policy import SkillPolicy


def identify_hard_windows_with_mases(policies: List[SkillPolicy], val_df: pd.DataFrame,
                                      hard_window_cache: Dict, policy_window_cache: Dict,
                                      cache_max_size: int, config: Dict, logger,
                                      hard_window_multiplier: float) -> tuple:
    """
    识别困难窗口并返回对应的MASE值，★ 优先从磁盘缓存读取 ★
    （独立函数，供 evolver 调用）

    优先顺序：
    1. 内存缓存（hard_window_cache）
    2. 磁盘缓存（window_results/*.json）—— 只读取，不写入
    3. 实时计算（若缓存不存在）
    
    ★★★ 关键修复：写入缓存时强制使用当前运行目录，而非扫描到的第一个目录 ★★★
    """
    import os
    import json
    from datetime import datetime
    from tqdm import tqdm

    policy_ids = sorted([p.policy_id for p in policies])
    current_hash = hashlib.md5(''.join(policy_ids).encode()).hexdigest()
    val_df_id = id(val_df)
    cache_key = f"{current_hash}_{val_df_id}"

    if cache_key in hard_window_cache:
        logger.log(f"   📦 使用内存缓存的困难窗口结果（策略池未变）")
        return hard_window_cache[cache_key]

    logger.log(f"   🔄 计算困难窗口（{len(policies)} 条策略 × {len(val_df)} 个窗口）...")

    # ★★★ ★★★ ★★★ 确定缓存目录 ★★★ ★★★ ★★★
    # 1. 读缓存目录：优先从现有目录读取（可复用旧缓存）
    llog_dir_from_config = config.get('llog_dir', 'llog')
    read_dir = None
    possible_paths = [
        os.path.join(llog_dir_from_config, 'window_results'),
        os.path.join('llog', 'window_results'),
        'window_results',
    ]
    # 尝试从运行目录的子目录中找到 window_results
    if os.path.exists(llog_dir_from_config):
        for subdir in os.listdir(llog_dir_from_config):
            if subdir.startswith('run_'):
                run_window_dir = os.path.join(llog_dir_from_config, subdir, 'window_results')
                if os.path.exists(run_window_dir):
                    possible_paths.insert(0, run_window_dir)

    for path in possible_paths:
        if os.path.exists(path):
            read_dir = path
            logger.log(f"   📁 [读缓存目录] 找到缓存目录: {read_dir}")
            break

    if read_dir is None:
        logger.log(f"   ⚠️ [读缓存目录] 未找到 window_results 目录，将实时计算")

    # ★★★ ★★★ ★★★ 2. 写缓存目录：强制使用当前运行目录 ★★★ ★★★ ★★★
    write_dir = os.path.join(llog_dir_from_config, 'window_results')
    os.makedirs(write_dir, exist_ok=True)  # 确保存在
    logger.log(f"   📁 [写缓存目录] 强制使用: {write_dir}")

    # ★★★ 加载磁盘缓存（仅从 read_dir 读取，不写入） ★★★
    disk_cache = {}
    if read_dir:
        try:
            json_files = [f for f in os.listdir(read_dir) if f.startswith('window_') and f.endswith('.json')]
            logger.log(f"   📁 [诊断] 找到 {len(json_files)} 个 JSON 缓存文件")
            for fname in json_files:
                fpath = os.path.join(read_dir, fname)
                try:
                    with open(fpath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    wid = data.get('window_id')
                    if wid is not None:
                        disk_cache[wid] = {
                            'best_mase': data.get('best_mase'),
                            'best_strategy': data.get('best_strategy'),
                            'is_trouble': data.get('is_trouble', False)
                        }
                except Exception as e:
                    logger.log(f"   ⚠️ [诊断] 读取 {fname} 失败: {e}")
            if disk_cache:
                logger.log(f"   📦 [诊断] 成功加载 {len(disk_cache)} 个窗口的缓存")
                sample_ids = list(disk_cache.keys())[:5]
                logger.log(f"   📋 [诊断] 缓存窗口ID示例: {sample_ids}")
        except Exception as e:
            logger.log(f"   ❌ [诊断] 扫描缓存目录失败: {e}")

    valid_windows = []
    for idx, row in val_df.iterrows():
        window_data_path = row.get('window_data_path')
        if window_data_path and os.path.exists(window_data_path):
            valid_windows.append((idx, row, window_data_path))

    if not valid_windows:
        logger.log(f"   ⚠️ 没有有效的窗口数据")
        return ([], [], [])

    logger.log(f"   📊 有效窗口: {len(valid_windows)}/{len(val_df)}")
    val_window_ids = [idx for idx, _, _ in valid_windows]
    cached_in_val = [wid for wid in val_window_ids if wid in disk_cache]
    not_cached = [wid for wid in val_window_ids if wid not in disk_cache]
    logger.log(f"   📊 [缓存命中] 命中: {len(cached_in_val)}/{len(val_window_ids)}")
    if not_cached:
        logger.log(f"   ⚠️ [缓存未命中] 未命中窗口: {not_cached[:10]}{'...' if len(not_cached) > 10 else ''}")
    if cached_in_val:
        logger.log(f"   ✅ [缓存命中] 命中窗口: {cached_in_val[:10]}{'...' if len(cached_in_val) > 10 else ''}")

    mases = []
    window_indices = []
    cache_hit_local = 0
    cache_miss_local = 0

    def _save_window_result_to_disk(window_id, origin, window_size, window_data_path,
                                    best_strategy, best_mase, features, is_trouble):
        """★ 强制写入 write_dir（当前运行目录） ★"""
        if write_dir is None:
            return False
        try:
            os.makedirs(write_dir, exist_ok=True)
            temp_file = os.path.join(write_dir, f'window_{window_id}.tmp')
            final_file = os.path.join(write_dir, f'window_{window_id}.json')
            result_data = {
                'window_id': window_id,
                'origin': origin,
                'window_size': window_size,
                'window_data_path': window_data_path,
                'best_strategy': best_strategy,
                'best_mase': best_mase,
                'features': features,
                'is_trouble': is_trouble,
                'saved_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, ensure_ascii=False, indent=2, default=str)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_file, final_file)
            return True
        except Exception as e:
            logger.log(f"   ❌ [缓存写入] 窗口 {window_id} 写入失败: {e}")
            return False

    pbar = tqdm(
        total=len(valid_windows),
        desc="   🔍 识别困难窗口",
        unit="窗口",
        ncols=110,
        position=0,
        leave=True
    )

    for idx, row, window_data_path in valid_windows:
        try:
            # ★★★ 优先使用磁盘缓存 ★★★
            if idx in disk_cache:
                cached = disk_cache[idx]
                mase = cached['best_mase']
                if mase is not None and mase != float('inf'):
                    mases.append(mase)
                    window_indices.append(idx)
                    cache_hit_local += 1
                    pbar.set_postfix({
                        '当前窗口': idx,
                        '已处理': len(mases),
                        '缓存命中': cache_hit_local
                    })
                    logger.log(f"   📂 [缓存命中] 窗口 {idx} 使用缓存 MASE={mase:.6f}")
                    pbar.update(1)
                    continue

            # ★★★ 缓存未命中：实时计算 ★★★
            try:
                wdata = load_window_data(window_data_path)
                train = wdata['train']
                test = wdata['test']
                period = wdata.get('period', 365)
                mase_scale = wdata.get('mase_scale', 1.0)
                horizon = wdata.get('horizon', 7)

                features = extract_features(train)

                scored = []
                for policy in policies:
                    if policy.status in ['ARCHIVE', 'DELETE']:
                        continue
                    score = policy.compute_applicability_score(features)
                    scored.append((policy, score))

                if scored:
                    best_policy, best_score = max(scored, key=lambda x: x[1])
                    if best_policy:
                        policy_version = getattr(best_policy, 'version', 1)
                        skill_hash = hashlib.md5(
                            json.dumps(best_policy.skill_strategy, sort_keys=True).encode()
                        ).hexdigest()[:8]
                        window_key = os.path.basename(window_data_path).replace('.pkl', '')
                        fine_cache_key = f"{best_policy.policy_id}_v{policy_version}_{skill_hash}_{window_key}"

                        if fine_cache_key in policy_window_cache:
                            mase = policy_window_cache[fine_cache_key]
                            cache_hit_local += 1
                        else:
                            pred = best_policy.execute(train, horizon, period)
                            if pred is not None and len(pred) == len(test):
                                mase = compute_mase(pred, test, mase_scale)
                                if len(policy_window_cache) < cache_max_size:
                                    policy_window_cache[fine_cache_key] = mase
                                else:
                                    keys = list(policy_window_cache.keys())[:cache_max_size // 2]
                                    for k in keys:
                                        del policy_window_cache[k]
                                    policy_window_cache[fine_cache_key] = mase
                                cache_miss_local += 1
                            else:
                                logger.log(f"   ⚠️ 窗口 {idx} 预测失败，跳过")
                                pbar.update(1)
                                continue

                        origin = row.get('origin', 0)
                        window_size = row.get('window_size', 600)
                        is_trouble = mase > 1.0
                        best_strategy_dict = {
                            'name': best_policy.name,
                            'stages': best_policy.skill_strategy.get('stages', []),
                            'description': best_policy.skill_strategy.get('description', '')
                        }
                        # ★★★ 写入当前运行目录（write_dir） ★★★
                        _save_window_result_to_disk(
                            idx,
                            origin,
                            window_size,
                            window_data_path,
                            best_strategy_dict,
                            mase,
                            features,
                            is_trouble
                        )

                        mases.append(mase)
                        window_indices.append(idx)
            except Exception as e:
                logger.log(f"   ⚠️ 窗口 {idx} 预测异常: {e}")
        finally:
            pbar.update(1)
            pbar.set_postfix({
                '当前窗口': idx,
                '已处理': len(mases),
                '缓存命中': cache_hit_local
            })

    pbar.close()

    logger.log(f"   ✅ 窗口处理完成，有效结果: {len(mases)}/{len(valid_windows)}")
    logger.log(f"   📊 [缓存统计] 磁盘缓存命中: {cache_hit_local} 个")
    logger.log(f"   📊 [缓存统计] 新计算并写入缓存: {cache_miss_local} 个")
    logger.log(f"   📁 [缓存写入位置] {write_dir}")

    if not mases:
        return ([], [], [])

    avg_mase = np.mean(mases)
    threshold = avg_mase * hard_window_multiplier

    hard_indices = []
    hard_mases = []
    for i, (mase, idx) in enumerate(zip(mases, window_indices)):
        if mase > threshold:
            hard_indices.append(idx)
            hard_mases.append(mase)

    result = (hard_indices, hard_mases, hard_indices)

    hard_window_cache[cache_key] = result

    if len(hard_window_cache) > 10:
        keys = list(hard_window_cache.keys())[:5]
        for k in keys:
            del hard_window_cache[k]

    total_cache = cache_hit_local + cache_miss_local
    if total_cache > 0:
        hit_rate = cache_hit_local / total_cache * 100
        logger.log(f"   📊 [缓存命中率] {hit_rate:.1f}% ({cache_hit_local}/{total_cache})")
        logger.log(f"   📦 [内存缓存大小] {len(policy_window_cache)} 条")

    return result