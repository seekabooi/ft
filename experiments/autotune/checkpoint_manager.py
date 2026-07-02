# experiments/autotune/checkpoint_manager.py
"""
检查点管理器
★ 支持断点续训
★ 保存/加载训练状态
★ 检测已完成轮次
★ 保存当前B子集索引
★ ★ 2026-06-25 增加 PolicyGraph 序列化
★ ★ 2026-06-26 增加 a_eval_completed 字段（跳过重复评估）
★ ★ ★ 2026-07-XX 增加 pending_round_state 字段（子阶段级别断点续传）
★ ★ ★ ★ 2026-08-XX 增加退休策略复活逻辑（一次性）- 已移至外部调用
"""

import os
import json
import time
import copy
from typing import Dict, List, Optional, Any
from datetime import datetime

from experiments.autotune.skill_policy import SkillPolicy


class CheckpointManager:
    """
    检查点管理器
    """

    def __init__(self, run_dir: str, logger):
        self.run_dir = run_dir
        self.logger = logger
        self.checkpoint_path = os.path.join(run_dir, "checkpoint.json")
        self._checkpoint = None

    def load(self) -> Dict:
        """
        加载检查点
        """
        if not os.path.exists(self.checkpoint_path):
            return {
                'completed_rounds': 0,
                'current_policies': [],
                'dataset': None,
                'horizon': None,
                'last_updated': None,
                'round_results': {},
                'best_round': 0,
                'best_mase': float('inf'),
                'current_b_subset_idx': 0,
                'policy_graph': None,
                'a_eval_completed': False,
                'pending_round_state': None
            }

        try:
            with open(self.checkpoint_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            policies_data = data.get('current_policies_data', [])
            current_policies = [SkillPolicy.from_dict(p) for p in policies_data]

            round_results = {}
            for round_key, round_data in data.get('round_results', {}).items():
                round_policies_data = round_data.get('policies_data', [])
                round_policies = [SkillPolicy.from_dict(p) for p in round_policies_data] if round_policies_data else []
                round_results[round_key] = {
                    'policies': round_policies,
                    'avg_mase': round_data.get('avg_mase', float('inf')),
                    'improvement': round_data.get('improvement', 0),
                    'policy_count': round_data.get('policy_count', 0)
                }

            self._checkpoint = {
                'completed_rounds': data.get('completed_rounds', 0),
                'current_policies': current_policies,
                'dataset': data.get('dataset'),
                'horizon': data.get('horizon'),
                'last_updated': data.get('last_updated'),
                'round_results': round_results,
                'best_round': data.get('best_round', 0),
                'best_mase': data.get('best_mase', float('inf')),
                'current_b_subset_idx': data.get('current_b_subset_idx', 0),
                'policy_graph': data.get('policy_graph', None),
                'a_eval_completed': data.get('a_eval_completed', False),
                'pending_round_state': data.get('pending_round_state', None)
            }

            self.logger.log(f"📂 加载检查点: 已完成 {self._checkpoint['completed_rounds']} 轮")
            if self._checkpoint['completed_rounds'] > 1:
                self.logger.log(f"   B子集索引: {self._checkpoint['current_b_subset_idx']}")
            if self._checkpoint.get('policy_graph'):
                clusters = self._checkpoint['policy_graph'].get('clusters', [])
                self.logger.log(f"   PolicyGraph: {len(clusters)} 个簇")
            if self._checkpoint.get('a_eval_completed', False):
                self.logger.log(f"   ✅ A部分评估已完成 (跳过重复评估)")
            pending = self._checkpoint.get('pending_round_state')
            if pending:
                self.logger.log(f"   🔄 [子阶段状态] 第 {pending.get('round')} 轮: "
                               f"reinduction_done={pending.get('reinduction_done', False)}, "
                               f"cache_built={pending.get('cache_built', False)}, "
                               f"rl_training_pending={pending.get('rl_training_pending', False)}")

            return self._checkpoint

        except Exception as e:
            self.logger.log(f"⚠️ 加载检查点失败: {e}")
            return {
                'completed_rounds': 0,
                'current_policies': [],
                'dataset': None,
                'horizon': None,
                'last_updated': None,
                'round_results': {},
                'best_round': 0,
                'best_mase': float('inf'),
                'current_b_subset_idx': 0,
                'policy_graph': None,
                'a_eval_completed': False,
                'pending_round_state': None
            }

    def save(self, completed_rounds: int, current_policies: List[SkillPolicy],
             dataset: str, horizon: int, round_results: Dict,
             best_round: int = 0, best_mase: float = float('inf'),
             current_b_subset_idx: int = 0,
             policy_graph: Optional[Dict] = None,
             a_eval_completed: bool = False,
             pending_round_state: Optional[Dict] = None) -> bool:
        """保存检查点"""
        try:
            checkpoint = {
                'completed_rounds': completed_rounds,
                'current_policies_data': [p.to_dict() for p in current_policies],
                'dataset': dataset,
                'horizon': horizon,
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'round_results': {},
                'best_round': best_round,
                'best_mase': best_mase,
                'current_b_subset_idx': current_b_subset_idx,
                'policy_graph': policy_graph,
                'a_eval_completed': a_eval_completed,
                'pending_round_state': pending_round_state
            }

            for round_key, round_data in round_results.items():
                policies = round_data.get('policies', [])
                checkpoint['round_results'][round_key] = {
                    'policies_data': [p.to_dict() for p in policies] if policies else [],
                    'avg_mase': round_data.get('avg_mase', float('inf')),
                    'improvement': round_data.get('improvement', 0),
                    'policy_count': round_data.get('policy_count', 0)
                }

            with open(self.checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(checkpoint, f, ensure_ascii=False, indent=2)

            self._checkpoint = checkpoint
            return True

        except Exception as e:
            self.logger.log(f"⚠️ 保存检查点失败: {e}")
            return False

    def get_next_round(self) -> int:
        if self._checkpoint is None:
            self.load()
        return self._checkpoint.get('completed_rounds', 0) + 1

    def is_completed(self, total_rounds: int) -> bool:
        if self._checkpoint is None:
            self.load()
        return self._checkpoint.get('completed_rounds', 0) >= total_rounds

    def get_completed_rounds(self) -> int:
        if self._checkpoint is None:
            self.load()
        return self._checkpoint.get('completed_rounds', 0)

    def is_a_eval_completed(self) -> bool:
        if self._checkpoint is None:
            self.load()
        return self._checkpoint.get('a_eval_completed', False)

    def get_pending_round_state(self) -> Optional[Dict]:
        if self._checkpoint is None:
            self.load()
        return self._checkpoint.get('pending_round_state', None)

    def clear_pending_round_state(self):
        if self._checkpoint is not None:
            self._checkpoint['pending_round_state'] = None

    def round_exists(self, round_num: int) -> bool:
        round_dir = os.path.join(self.run_dir, f"round_{round_num}")
        if not os.path.exists(round_dir):
            return False
        optimized_path = os.path.join(round_dir, "refined_policies_optimized.json")
        if os.path.exists(optimized_path):
            return True
        raw_path = os.path.join(round_dir, "refined_policies_raw.json")
        return os.path.exists(raw_path)

    def get_round_policies(self, round_num: int, version: str = "optimized") -> List[SkillPolicy]:
        round_dir = os.path.join(self.run_dir, f"round_{round_num}")
        if not os.path.exists(round_dir):
            return []

        if version == "optimized":
            file_name = "refined_policies_optimized.json"
        else:
            file_name = "refined_policies_raw.json"

        file_path = os.path.join(round_dir, file_name)
        if not os.path.exists(file_path):
            old_path = os.path.join(round_dir, "refined_policies.json")
            if os.path.exists(old_path):
                file_path = old_path

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                policies_data = data.get('policies', [])
                return [SkillPolicy.from_dict(p) for p in policies_data]
        except Exception as e:
            self.logger.log(f"⚠️ 加载轮次 {round_num} 策略失败: {e}")
            return []

    def detect_completed_rounds(self) -> int:
        completed = 0
        round_num = 1
        while True:
            if self.round_exists(round_num):
                completed = round_num
                round_num += 1
            else:
                break
        return completed