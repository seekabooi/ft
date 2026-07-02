# experiments/autotune/iterative_refiner_base.py
"""
Policy Evolution Engine - 基础定义
包含：类定义、__init__、基础属性、辅助方法
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

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from experiments.autotune.utils import ProgressLogger, load_window_data, compute_mase, extract_features
from experiments.autotune.skill_policy import SkillPolicy
from experiments.autotune.policy_health_monitor import PolicyHealthMonitor
from experiments.autotune.evolution_controller import EvolutionController
from experiments.autotune.retirement_mechanism import RetirementMechanism
from experiments.autotune.merge_simulator import MergeSimulator
from experiments.autotune.coverage_gap_analyzer import CoverageGapAnalyzer
from experiments.autotune.regime_drift_monitor import RegimeDriftMonitor
from experiments.autotune.rule_refresh import RuleRefresher
from experiments.autotune.freeze_guard import FreezeGuard
from experiments.autotune.confidence_calculator import ConfidenceCalculator


class PolicyEvolutionEngine:
    """
    Policy Evolution Engine - v5 多轮训练版（完整版）
    """

    def __init__(self, config: Dict, logger: ProgressLogger, inducer=None, branch_loader=None):
        self.config = config
        self.logger = logger
        self.inducer = inducer
        self.branch_loader = branch_loader

        self.health_monitor = PolicyHealthMonitor(config, logger)
        self.evolution_controller = EvolutionController(config, logger)
        self.retirement_mechanism = RetirementMechanism(config, logger)
        self.merge_simulator = MergeSimulator(config, logger)
        self.coverage_gap_analyzer = CoverageGapAnalyzer(config, logger)
        self.drift_monitor = RegimeDriftMonitor(config, logger)
        self.rule_refresher = RuleRefresher(config, logger)
        self.freeze_guard = FreezeGuard(config, logger)
        self.confidence_calculator = ConfidenceCalculator(config)

        reind_cfg = config.get('reinduction', {})
        self.reinduction_enabled = reind_cfg.get('enabled', True)
        self.hard_window_multiplier = reind_cfg.get('hard_window_multiplier', 1.2)
        self.hard_window_ratio_threshold = reind_cfg.get('hard_window_ratio_threshold', 0.10)
        self.mase_improvement_threshold = reind_cfg.get('mase_improvement_threshold', 0.03)
        self.max_new_policies_per_round = reind_cfg.get('max_new_policies_per_round', 2)
        self.min_hard_windows = reind_cfg.get('min_hard_windows', 3)
        self.use_reference_rules = reind_cfg.get('use_reference_rules', True)

        self.patch_enabled = reind_cfg.get('patch_enabled', True)
        self.patch_top_k = reind_cfg.get('patch_top_k', 3)
        self.patch_improvement_threshold = reind_cfg.get('patch_improvement_threshold', 0.05)
        self.patch_min_windows = reind_cfg.get('patch_min_windows', 2)
        self.patch_max_retries = reind_cfg.get('patch_max_retries', 3)
        self.patch_retry_delay = reind_cfg.get('patch_retry_delay', 2)

        trouble_cfg = config.get('trouble_patch', {})
        self.trouble_mase_threshold = trouble_cfg.get('mase_threshold', 1.0)

        self._round = 0

        self._hard_window_cache = {}
        self._last_policies_hash = None
        self._last_val_df_id = None
        self._policy_window_cache = {}
        self._cache_max_size = 500
        self._cache_hit_count = 0
        self._cache_miss_count = 0
        self._disk_cache_index = {}

    # ==================== 辅助方法 ====================

    def _collect_trouble_window(self, window_id: int, mase: float,
                                window_data_path: str, origin: int,
                                window_size: int, best_strategy_name: str):
        """收集困难窗口到全局池（委托给 inducer）"""
        if self.inducer is not None and hasattr(self.inducer, '_collect_trouble_window'):
            strategy = {'name': best_strategy_name}
            self.inducer._collect_trouble_window(
                window_id, mase, window_data_path,
                origin, window_size, strategy
            )

    # ★★★ 这是修改后的函数：直接读取缓存，不再暴力执行预测 ★★★
    def _collect_worst_window_scores(self, policies: List[SkillPolicy], val_df: pd.DataFrame,
                                     top_k: int = 3) -> List[Dict]:
        """
        ★ 优化版：不再暴力遍历所有策略执行预测，直接从 DataFrame 读取缓存的 best_mase。
        耗时从 O(m*n) 降为 O(1)，彻底解决 Patch 前置评估卡顿问题。
        """
        window_scores = []

        for idx, row in val_df.iterrows():
            # 1. 从采集缓存中读取已有 MASE（采集阶段已经算好）
            mase = row.get('best_mase', None)

            # 2. 若缓存缺失（极少情况），使用默认值 1.0，绝不在此处调用 policy.execute
            if mase is None or np.isnan(mase) or np.isinf(mase):
                mase = 1.0

            # 3. 简单收集特征（仅用于传给 LLM，不需要精确重算）
            features = {}
            for col in ['trend_strength', 'seasonal_strength', 'cv', 'window_size']:
                if col in row:
                    features[col] = row[col]

            window_scores.append({
                'window_id': idx,
                'mase': float(mase),
                'features': features
            })

        # 按 MASE 降序排列（最差的排最前）
        window_scores.sort(key=lambda x: x['mase'], reverse=True)

        # 返回 Top-K 最差窗口
        return window_scores[:top_k]

    def _compute_redundancy_score(self, policies: List[SkillPolicy]) -> float:
        if len(policies) < 2:
            return 0.0
        total_pairs = 0
        redundant_pairs = 0
        for i in range(len(policies)):
            for j in range(i + 1, len(policies)):
                total_pairs += 1
                groups_a = set(policies[i].feature_groups)
                groups_b = set(policies[j].feature_groups)
                if groups_a and groups_b:
                    overlap = len(groups_a & groups_b) / max(1, len(groups_a | groups_b))
                    if overlap > 0.7:
                        redundant_pairs += 1
        return redundant_pairs / max(1, total_pairs)

    def _validate_policies(self, policies: List[SkillPolicy], val_df: pd.DataFrame) -> float:
        if not policies or val_df.empty:
            return 1.0
        mases = []
        for _, row in val_df.iterrows():
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
                scored = []
                for policy in policies:
                    if policy.status in ['ARCHIVE', 'DELETE']:
                        continue
                    score = policy.compute_applicability_score(features)
                    scored.append((policy, score))
                if scored:
                    best_policy, _ = max(scored, key=lambda x: x[1])
                    if best_policy:
                        pred = best_policy.execute(train, horizon, period)
                        if pred is not None and len(pred) == len(test):
                            mases.append(compute_mase(pred, test, mase_scale))
            except:
                continue
        return np.mean(mases) if mases else 1.0

    def _evaluate_policy_on_windows(self, policy: SkillPolicy, df: pd.DataFrame) -> Optional[float]:
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

    def _identify_hard_windows(self, policies: List[SkillPolicy], val_df: pd.DataFrame) -> List[int]:
        hard_indices, _, _ = self._identify_hard_windows_with_mases(policies, val_df)
        return hard_indices

    def _load_policies(self) -> List[SkillPolicy]:
        llog_dir = self.config.get('llog_dir', 'llog')
        policies_path = os.path.join(llog_dir, "refined_policies.json")
        if os.path.exists(policies_path):
            with open(policies_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return [SkillPolicy.from_dict(p) for p in data.get('policies', [])]
        return []