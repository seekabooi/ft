# experiments/autotune/performance_auditor.py
"""
Policy Diagnostic Module - 简化版
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

from experiments.autotune.utils import load_window_data, compute_mase, extract_features
from experiments.autotune.skill_policy import SkillPolicy


class PolicyDiagnosticModule:
    def __init__(self, config: Dict, logger):
        self.config = config
        self.logger = logger
        self.threshold_multiplier = config.get('audit_threshold_multiplier', 1.2)

    def diagnose(self, collected_data: pd.DataFrame,
                 policies: List[SkillPolicy]) -> Tuple[List[int], Dict]:
        self.logger.log("=" * 60)
        self.logger.log("🔍 策略诊断开始...")
        self.logger.log("=" * 60)

        mase_list = []
        window_details = []

        for idx, row in collected_data.iterrows():
            window_id = row['window_id']
            window_split = row.get('split', 'unknown')
            window_data_path = row['window_data_path']

            if not window_data_path:
                continue

            try:
                wdata = load_window_data(window_data_path)
                train = wdata['train']
                test = wdata['test']
                period = wdata.get('period', 365)
                mase_scale = wdata.get('mase_scale', 1.0)
                horizon = wdata.get('horizon', 7)

                features = extract_features(train)
                matched_policy = None
                for policy in policies:
                    if policy.is_applicable(features):
                        matched_policy = policy
                        break

                if matched_policy is None:
                    continue

                pred = matched_policy.execute(train, horizon, period)
                if pred is None or len(pred) != len(test):
                    continue

                mase = compute_mase(pred, test, mase_scale)
                mase_list.append((window_id, mase, window_split, matched_policy.policy_id))
                window_details.append({
                    'window_id': window_id,
                    'split': window_split,
                    'mase': mase,
                    'policy_id': matched_policy.policy_id,
                    'policy_name': matched_policy.name,
                    'origin': row.get('origin', 0)
                })
            except Exception as e:
                self.logger.log(f"⚠️ 窗口 {window_id} 诊断失败: {e}")

        if not mase_list:
            return [], self._empty_report()

        mases = np.array([m for _, m, _, _ in mase_list])
        avg_mase = np.mean(mases)
        threshold = avg_mase * self.threshold_multiplier
        hard_window_ids = [wid for wid, m, _, _ in mase_list if m > threshold]

        sorted_list = sorted(mase_list, key=lambda x: x[1], reverse=True)
        worst_3_ids = [wid for wid, _, _, _ in sorted_list[:3]]

        report = {
            'avg_mase': avg_mase,
            'std_mase': np.std(mases),
            'total_windows': len(mase_list),
            'hard_window_ids': hard_window_ids,
            'hard_count': len(hard_window_ids),
            'worst_3_window_ids': worst_3_ids,
            'threshold': threshold,
            'window_details': window_details
        }

        self.logger.log(f"   📈 平均MASE: {avg_mase:.4f} ± {np.std(mases):.4f}")
        self.logger.log(f"   🎯 困难窗口数: {len(hard_window_ids)}/{len(mase_list)}")

        return hard_window_ids, report

    def _empty_report(self) -> Dict:
        return {
            'avg_mase': float('inf'),
            'std_mase': 0,
            'total_windows': 0,
            'hard_window_ids': [],
            'hard_count': 0,
            'worst_3_window_ids': [],
            'threshold': 0,
            'window_details': []
        }