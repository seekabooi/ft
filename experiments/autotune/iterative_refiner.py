# experiments/autotune/iterative_refiner.py
"""
Policy Evolution Engine - 主入口
组装所有子模块，导出 PolicyEvolutionEngine 类
"""

from experiments.autotune.iterative_refiner_base import PolicyEvolutionEngine

# ★ 导入并注入所有方法（自动执行）
import experiments.autotune.iterative_refiner_core
import experiments.autotune.iterative_refiner_reind
import experiments.autotune.iterative_refiner_patch

# 还需要导入 run 和 _identify_hard_windows_with_mases
# 这些方法在原文件中是类方法，我们也注入它们
from experiments.autotune.iterative_refiner_utils import identify_hard_windows_with_mases


# 注入 _identify_hard_windows_with_mases 方法
def _identify_hard_windows_with_mases_method(self, policies, val_df):
    return identify_hard_windows_with_mases(
        policies, val_df, self._hard_window_cache, self._policy_window_cache,
        self._cache_max_size, self.config, self.logger, self.hard_window_multiplier
    )

PolicyEvolutionEngine._identify_hard_windows_with_mases = _identify_hard_windows_with_mases_method


# ★ 注入 run 方法（来自原文件）
def run_method(self, dataset_name: str, horizon: int, rounds: int = 3):
    import os
    import json
    self.logger.log("=" * 70)
    self.logger.log("🔄 Policy Evolution Engine v5 (独立运行)")
    self.logger.log("=" * 70)
    csv_path = os.path.join(self.config.get('output_dir', 'storage/autotune_results'), "collected_windows.csv")
    if not os.path.exists(csv_path):
        self.logger.log(f"❌ 未找到采集数据: {csv_path}")
        return
    collected_df = pd.read_csv(csv_path)
    val_df = collected_df[collected_df['split'] == 'val']
    if len(val_df) < 3:
        val_df = collected_df
    policies = self._load_policies()
    if not policies:
        self.logger.log("❌ 未找到策略文件")
        return
    self.logger.log(f"📋 初始策略数: {len(policies)}")
    for round_num in range(1, rounds + 1):
        result = self.run_round(policies, val_df, round_num)
        policies = result['policies']
        snapshot_path = os.path.join(self.config.get('llog_dir', 'llog'), f"policies_round_{round_num}.json")
        with open(snapshot_path, 'w', encoding='utf-8') as f:
            json.dump([p.to_dict() for p in policies], f, ensure_ascii=False, indent=2)
    final_path = os.path.join(self.config.get('llog_dir', 'llog'), "refined_policies.json")
    with open(final_path, 'w', encoding='utf-8') as f:
        json.dump({'policies': [p.to_dict() for p in policies]}, f, ensure_ascii=False, indent=2)
    self.logger.log(f"\n📁 最终策略: {final_path}")
    self.logger.log(f"📊 最终策略数: {len(policies)}")

PolicyEvolutionEngine.run = run_method


__all__ = ['PolicyEvolutionEngine']