# experiments/autotune/diagnostic_logger.py
"""
诊断日志模块 - v4 全功能版
★ 路径统一：所有输出写入 llog/ 根目录
★ 不再嵌套子文件夹
"""

import os
import json
import time
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import defaultdict


class DiagnosticLogger:
    """诊断日志器 - v4 全功能版"""

    def __init__(self, llog_dir: str = "llog"):
        # ★ 直接使用 llog_dir，不创建子文件夹
        self.llog_dir = llog_dir
        os.makedirs(llog_dir, exist_ok=True)

        self.data = {
            'timestamp': datetime.now().isoformat(),
            'retrieval_records': [],
            'mixture_records': [],
            'condition_evaluations': [],
            'policy_switches': [],
            'window_matches': [],
            'reward_records': [],
            'stability_records': [],
            'summary': {}
        }
        self._last_policy_id = None

    def log_retrieval(self, window_id: int, state: Dict,
                      matched_policy: Any, all_applicable: List,
                      fallback_used: bool):
        """记录策略检索"""
        record = {
            'window_id': window_id,
            'state_numeric': state.get('numeric', {}),
            'matched_policy_id': matched_policy.policy_id if matched_policy else None,
            'matched_policy_name': matched_policy.name if matched_policy else None,
            'matched_condition': matched_policy.state_condition if matched_policy else {},
            'matched_feature_groups': matched_policy.feature_groups if matched_policy else [],
            'matched_utility': matched_policy.utility_score if matched_policy else 0,
            'all_applicable_count': len(all_applicable),
            'all_applicable_ids': [p.policy_id for p in all_applicable],
            'fallback_used': fallback_used,
            'timestamp': time.time()
        }
        self.data['retrieval_records'].append(record)

        # 记录策略切换
        if self._last_policy_id and self._last_policy_id != matched_policy.policy_id:
            self.data['policy_switches'].append({
                'window_id': window_id,
                'from_policy': self._last_policy_id,
                'to_policy': matched_policy.policy_id if matched_policy else None,
                'timestamp': time.time()
            })
        self._last_policy_id = matched_policy.policy_id if matched_policy else None

    def log_mixture_weights(self, window_id: int, policies: List,
                            soft_probs: List, top_k_indices: List,
                            entropy: float = None):
        """记录混合权重"""
        record = {
            'window_id': window_id,
            'policy_weights': [
                {'policy_id': p.policy_id, 'policy_name': p.name,
                 'weight': float(soft_probs[i]), 'selected': i in top_k_indices}
                for i, p in enumerate(policies)
            ],
            'entropy': entropy,
            'timestamp': time.time()
        }
        self.data['mixture_records'].append(record)

    def log_condition_evaluation(self, policy_id: str, policy_name: str,
                                  state: Dict, results: Dict):
        """记录条件评估"""
        record = {
            'policy_id': policy_id,
            'policy_name': policy_name,
            'state': state.get('numeric', {}),
            'condition_results': results,
            'overall_applicable': all(results.values()) if results else False,
            'timestamp': time.time()
        }
        self.data['condition_evaluations'].append(record)

    def log_window_match(self, window_id: int, split: str,
                         policy_id: str, mase: float,
                         no_rule_mase: float, improvement: float):
        """记录窗口匹配"""
        record = {
            'window_id': window_id,
            'split': split,
            'policy_id': policy_id,
            'mase': mase,
            'no_rule_mase': no_rule_mase,
            'improvement': improvement,
            'timestamp': time.time()
        }
        self.data['window_matches'].append(record)

    def log_reward(self, window_id: int, reward: float, reward_detail: Dict):
        """记录 Reward"""
        record = {
            'window_id': window_id,
            'reward': reward,
            'reward_detail': reward_detail,
            'timestamp': time.time()
        }
        self.data['reward_records'].append(record)

    def log_stability(self, stability_status: Dict):
        """记录稳定性状态"""
        record = {
            'timestamp': time.time(),
            'status': stability_status
        }
        self.data['stability_records'].append(record)

    def compute_summary(self):
        """计算摘要统计"""
        records = self.data['retrieval_records']
        if not records:
            return

        policy_usage = defaultdict(int)
        for r in records:
            if r['matched_policy_id']:
                policy_usage[r['matched_policy_id']] += 1

        fallback_count = sum(1 for r in records if r['fallback_used'])
        switch_count = len(self.data['policy_switches'])
        utilities = [r['matched_utility'] for r in records if r['matched_policy_id']]
        avg_utility = np.mean(utilities) if utilities else 0.0

        # Reward 统计
        rewards = [r.get('reward', 0) for r in self.data['reward_records']]
        avg_reward = np.mean(rewards) if rewards else 0.0

        self.data['summary'] = {
            'total_windows': len(records),
            'policy_usage': dict(policy_usage),
            'fallback_usage_count': fallback_count,
            'fallback_usage_rate': fallback_count / len(records) if records else 0,
            'policy_switch_count': switch_count,
            'avg_utility': avg_utility,
            'avg_reward': avg_reward,
            'reward_records_count': len(self.data['reward_records'])
        }

    def save(self) -> Optional[str]:
        """保存诊断日志到 llog/ 根目录"""
        try:
            self.compute_summary()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # ★ 直接写入 llog/ 根目录
            file_path = os.path.join(self.llog_dir, f'diagnostic_{timestamp}.json')
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2, default=str)
            return file_path
        except Exception as e:
            print(f"⚠️ 保存诊断日志失败: {e}")
            return None

    def reset(self):
        """重置日志数据"""
        self.data = {
            'timestamp': datetime.now().isoformat(),
            'retrieval_records': [],
            'mixture_records': [],
            'condition_evaluations': [],
            'policy_switches': [],
            'window_matches': [],
            'reward_records': [],
            'stability_records': [],
            'summary': {}
        }
        self._last_policy_id = None