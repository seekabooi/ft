# experiments/autotune/iterative_refiner_patch.py
"""
Policy Evolution Engine - Patch + 困难窗口识别
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

from experiments.autotune.skill_policy import SkillPolicy
from experiments.autotune.iterative_refiner_base import PolicyEvolutionEngine
from experiments.autotune.iterative_refiner_utils import identify_hard_windows_with_mases


def patch_worst_windows_impl(self: PolicyEvolutionEngine, policies: List[SkillPolicy],
                             val_df: pd.DataFrame, round_num: int) -> int:
    """Patch 核心实现"""
    self.logger.log("   🔧 检查Patch条件（针对最差窗口打补丁）...")

    if len(val_df) < self.patch_min_windows:
        self.logger.log(f"   ℹ️ 验证集窗口数 ({len(val_df)}) < {self.patch_min_windows}，跳过补丁")
        return 0

    window_scores = []
    for idx, row in val_df.iterrows():
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

            features = extract_features(train)

            best_mase = float('inf')
            for policy in policies:
                if policy.status in ['ARCHIVE', 'DELETE']:
                    continue
                if policy.compute_applicability_score(features) > 0.3:
                    pred = policy.execute(train, horizon, period)
                    if pred is not None and len(pred) == len(test):
                        mase = compute_mase(pred, test, mase_scale)
                        if mase < best_mase:
                            best_mase = mase
            if best_mase < float('inf'):
                window_scores.append((idx, best_mase))
        except Exception:
            continue

    if not window_scores:
        self.logger.log("   ℹ️ 无有效窗口，跳过补丁")
        return 0

    window_scores.sort(key=lambda x: x[1], reverse=True)
    top_windows = window_scores[:self.patch_top_k]

    if len(top_windows) < self.patch_min_windows:
        self.logger.log(f"   ℹ️ 最差窗口数 ({len(top_windows)}) < {self.patch_min_windows}，不触发补丁")
        return 0

    self.logger.log(
        f"   📊 选中 {len(top_windows)} 个最差窗口 (MASE 范围: {top_windows[-1][1]:.4f} ~ {top_windows[0][1]:.4f})")

    for idx, mase in top_windows:
        if mase > self.trouble_mase_threshold:
            row = val_df.loc[idx]
            window_data_path = row.get('window_data_path', '')
            origin = row.get('origin', 0)
            window_size = row.get('window_size', 600)
            best_policy = min(policies, key=lambda p: p.avg_mase) if policies else None
            best_name = best_policy.name if best_policy else 'unknown'
            self._collect_trouble_window(
                idx, mase, window_data_path,
                origin, window_size, best_name
            )

    patch_df = val_df.loc[[idx for idx, _ in top_windows]].copy()

    best_policy = min(policies, key=lambda p: p.avg_mase) if policies else None
    best_mase = best_policy.avg_mase if best_policy else 1.0

    total_added = 0
    for retry in range(self.patch_max_retries):
        if retry > 0:
            self.logger.log(f"   🔄 补丁重试 {retry}/{self.patch_max_retries}...")
            time.sleep(self.patch_retry_delay)

        try:
            result = self.inducer.induce(patch_df)
            new_candidates_data = result.get('policies', [])
            new_candidates = [SkillPolicy.from_dict(p) for p in new_candidates_data]
        except Exception as e:
            self.logger.log(f"   ⚠️ 补丁生成失败 (重试 {retry + 1}/{self.patch_max_retries}): {e}")
            continue

        if not new_candidates:
            self.logger.log(f"   ℹ️ 未生成补丁策略 (重试 {retry + 1}/{self.patch_max_retries})")
            continue

        self.logger.log(f"   📋 生成 {len(new_candidates)} 个补丁候选 (重试 {retry + 1}/{self.patch_max_retries})")

        added_this_round = 0
        existing_embeddings = [np.array(p.embedding) for p in policies if p.embedding]

        pbar = tqdm(
            total=len(new_candidates),
            desc="   🔧 评估补丁候选",
            unit="个",
            ncols=100,
            position=0,
            leave=True
        )

        for candidate in new_candidates:
            pbar.update(1)

            if total_added >= self.max_new_policies_per_round:
                self.logger.log(f"   📌 已达到本轮补丁上限 ({self.max_new_policies_per_round})")
                pbar.close()
                break

            is_similar = False
            if candidate.embedding and existing_embeddings:
                cand_emb = np.array(candidate.embedding)
                for emb in existing_embeddings:
                    if len(emb) == len(cand_emb):
                        cos_sim = np.dot(cand_emb, emb) / (np.linalg.norm(cand_emb) * np.linalg.norm(emb) + 1e-8)
                        if cos_sim > 0.8:
                            is_similar = True
                            self.logger.log(f"      ⚠️ 补丁候选与现有策略相似度 {cos_sim:.3f} > 0.8，跳过")
                            pbar.set_postfix({'当前': candidate.name[:20], '状态': '跳过(相似)'})
                            break

            if is_similar:
                continue

            try:
                cand_mase = self._evaluate_policy_on_windows(candidate, patch_df)
            except Exception as e:
                self.logger.log(f"      ⚠️ 补丁候选评估失败: {e}")
                pbar.set_postfix({'当前': candidate.name[:20], '状态': '失败'})
                continue

            if cand_mase is None or np.isnan(cand_mase) or np.isinf(cand_mase):
                self.logger.log(f"      ⚠️ 补丁候选MASE无效，跳过")
                pbar.set_postfix({'当前': candidate.name[:20], '状态': 'MASE无效'})
                continue

            improvement = (best_mase - cand_mase) / best_mase if best_mase > 0 else 0
            self.logger.log(
                f"      补丁候选: MASE={cand_mase:.6f}, 提升={improvement:.2%} (阈值={self.patch_improvement_threshold:.2%})")

            if improvement >= self.patch_improvement_threshold:
                candidate.status = 'TRIAL'
                candidate.name = f"patch_{round_num}_{candidate.name[:8]}"
                candidate.created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                candidate.avg_mase = cand_mase
                candidate.error_mean = cand_mase

                policies.append(candidate)
                total_added += 1
                added_this_round += 1
                self.logger.log(
                    f"      ✅ 加入补丁策略: {candidate.name} (MASE={cand_mase:.6f}, 提升={improvement:.2%})")
                pbar.set_postfix({'当前': candidate.name[:20], '状态': '✅ 加入'})

                if cand_mase < best_mase:
                    best_mase = cand_mase
            else:
                self.logger.log(f"      ❌ 提升不足，跳过 (需 {self.patch_improvement_threshold:.2%})")
                pbar.set_postfix({'当前': candidate.name[:20], '状态': '❌ 跳过'})

        pbar.close()

        if added_this_round > 0:
            self.logger.log(f"   ✅ 本轮补丁新增 {added_this_round} 条策略")
            break
        else:
            self.logger.log(f"   ℹ️ 本次重试没有满足条件的补丁策略，继续尝试...")

    if total_added > 0:
        self.logger.log(f"   ✅ 补丁完成，共新增 {total_added} 条策略")
    else:
        self.logger.log(f"   ⚠️ 补丁失败，{self.patch_max_retries} 次重试均未加入任何策略")

    return total_added


# 将方法注入到类中
PolicyEvolutionEngine._patch_worst_windows = patch_worst_windows_impl