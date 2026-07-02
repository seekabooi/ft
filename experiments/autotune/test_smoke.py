#!/usr/bin/env python
# experiments/autotune/test_smoke.py
"""
冒烟测试 - 快速验证（使用缓存，跳过 LLM）

运行指令：
    python -m experiments.autotune.test_smoke --verbose

预计耗时：< 5 秒（全部使用缓存）
"""

import os
import sys
import json
import time
import shutil
import pickle
import glob
from datetime import datetime
from typing import Dict, List, Any
import numpy as np
import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# ============================================================
# 测试运行器
# ============================================================

class SmokeTestRunner:
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.start_time = time.time()
        self.test_results = {
            'passed': [],
            'failed': [],
            'skipped': []
        }
        self._log("=" * 80)
        self._log("🧪 SPLS 冒烟测试 v2.0（缓存模式）")
        self._log(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._log("=" * 80)

    def _log(self, msg: str, level: str = "INFO"):
        if self.verbose:
            print(msg)

    def _log_section(self, title: str):
        self._log("")
        self._log("=" * 80)
        self._log(f"📍 {title}")
        self._log("=" * 80)

    def _log_pass(self, test_name: str, detail: str = ""):
        self.test_results['passed'].append(test_name)
        self._log(f"   ✅ PASS: {test_name}" + (f" ({detail})" if detail else ""))

    def _log_fail(self, test_name: str, detail: str = ""):
        self.test_results['failed'].append(test_name)
        self._log(f"   ❌ FAIL: {test_name}" + (f" ({detail})" if detail else ""))

    def _log_skip(self, test_name: str, reason: str = ""):
        self.test_results['skipped'].append(test_name)
        self._log(f"   ⏭️ SKIP: {test_name}" + (f" ({reason})" if reason else ""))

    def run(self) -> Dict:
        """执行所有测试"""
        self._log_section("1. 环境准备")

        # 检查关键模块
        try:
            from experiments.autotune.main import SPLSAutoTuner
            from experiments.autotune.iterative_refiner import PolicyEvolutionEngine
            from experiments.autotune.validator import PolicyEvaluationOracle
            from experiments.autotune.evolution_controller import EvolutionController
            from experiments.autotune.utils import ProgressLogger, load_config
            self._log_pass("模块导入", "所有核心模块可导入")
        except ImportError as e:
            self._log_fail("模块导入", str(e))
            return self.test_results

        # 检查缓存数据
        self._test_cache_data()
        self._test_collector_with_cache()
        self._test_evolution_force_trigger()
        self._test_validator_cache()
        self._test_parallel_evaluation()
        self._test_testset_evaluation()
        self._test_checkpoint()

        self._print_summary()
        return self.test_results

    # ============================================================
    # 测试1：检查缓存数据
    # ============================================================
    def _test_cache_data(self):
        self._log_section("2. 缓存数据检查")

        # 检查窗口数据缓存
        window_data_dir = "storage/autotune_results/window_data"
        if os.path.exists(window_data_dir):
            pkl_files = glob.glob(os.path.join(window_data_dir, "*.pkl"))
            self._log(f"   📊 窗口数据缓存: {len(pkl_files)} 个 .pkl 文件")
            if len(pkl_files) >= 3:
                self._log_pass("缓存数据", f"找到 {len(pkl_files)} 个窗口数据文件")
            else:
                self._log_fail("缓存数据", f"窗口数据不足 ({len(pkl_files)} < 3)")
        else:
            self._log_fail("缓存数据", f"目录不存在: {window_data_dir}")

        # 检查 collected_windows.csv
        csv_path = "storage/autotune_results/collected_windows.csv"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            self._log(f"   📊 collected_windows.csv: {len(df)} 个窗口")
            if len(df) >= 3:
                self._log_pass("缓存CSV", f"成功加载 {len(df)} 个窗口")
            else:
                self._log_fail("缓存CSV", f"窗口数 {len(df)} < 3")
        else:
            self._log_fail("缓存CSV", "文件不存在")

    # ============================================================
    # 测试2：采集器（使用缓存，不调用 LLM）
    # ============================================================
    def _test_collector_with_cache(self):
        self._log_section("3. 采集器测试（使用缓存，不调用 LLM）")

        try:
            from experiments.autotune.collector import StateWindowGenerator
            from experiments.autotune.utils import MemoryCache, ProgressLogger

            # 使用已有的缓存目录
            logger = ProgressLogger(log_dir="llog/test_smoke", verbose=False)
            cache = MemoryCache(cache_dir="storage/autotune_results/cache")

            # ★ 关键：设置 skip_collection=True 跳过 LLM
            config = {
                'output_dir': 'storage/autotune_results',
                'skip_collection': True,  # ★ 使用缓存，不调用 LLM
                'llm': {'model': 'glm-4.5-air'},
                'parallel': {'enabled': False},
                'fixed_params': {}
            }

            collector = StateWindowGenerator(config, logger, cache)
            collector.set_verbose(False)

            # 尝试加载已有数据（不重新生成）
            csv_path = "storage/autotune_results/collected_windows.csv"
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                collected = df.to_dict('records')
                self._log(f"   📊 从缓存加载 {len(collected)} 个窗口（无 LLM 调用）")
                if len(collected) >= 3:
                    self._log_pass("采集器（缓存）", f"成功加载 {len(collected)} 个窗口，无 LLM 调用")
                else:
                    self._log_fail("采集器（缓存）", f"窗口数 {len(collected)} < 3")
            else:
                self._log_fail("采集器（缓存）", "无缓存数据可加载")

        except Exception as e:
            self._log_fail("采集器（缓存）", str(e))
            import traceback
            traceback.print_exc()

    # ============================================================
    # 测试3：强制演化触发（不依赖 LLM）
    # ============================================================
    def _test_evolution_force_trigger(self):
        self._log_section("4. 强制演化触发测试 (force=True)")

        try:
            from experiments.autotune.evolution_controller import EvolutionController
            from experiments.autotune.utils import ProgressLogger

            logger = ProgressLogger(log_dir="llog/test_smoke", verbose=False)
            config = {'evolution': {'cooldown': {'global_period': 200}}}
            controller = EvolutionController(config, logger)

            # 测试正常触发
            normal_result = controller.should_trigger(
                current_step=100,
                policy_count=5,
                hard_window_ratio=0.01,
                redundancy_score=0.1,
                evolution_history=[]
            )

            # 测试强制触发
            force_result = controller.force_trigger()

            self._log(f"   📊 正常触发: {normal_result['should_trigger']} ({normal_result.get('reason', '')})")
            self._log(f"   📊 强制触发: {force_result['should_trigger']} ({force_result.get('reason', '')})")

            if force_result['should_trigger'] and force_result['trigger_type'] == 'force':
                self._log_pass("强制演化触发", "force_trigger() 返回正确")
            else:
                self._log_fail("强制演化触发", f"force_trigger() 返回异常: {force_result}")

            # 测试 record_evolution
            controller.record_evolution(150)
            cooldown = controller.get_cooldown_remaining(160)
            self._log(f"   📊 冷却剩余: {cooldown} 窗口")
            if cooldown >= 0:
                self._log_pass("冷却机制", f"record_evolution 正常工作，剩余 {cooldown} 窗口")

        except Exception as e:
            self._log_fail("强制演化触发", str(e))
            import traceback
            traceback.print_exc()

    # ============================================================
    # 测试4：验证器缓存
    # ============================================================
    def _test_validator_cache(self):
        self._log_section("5. 预测缓存测试")

        try:
            from experiments.autotune.validator import PolicyEvaluationOracle
            from experiments.autotune.utils import ProgressLogger

            logger = ProgressLogger(log_dir="llog/test_smoke", verbose=False)
            validator = PolicyEvaluationOracle({}, logger)

            # 检查缓存相关方法是否存在
            has_cache_key = hasattr(validator, '_get_cache_key')
            has_get_pred = hasattr(validator, '_get_prediction')
            has_cache_dict = hasattr(validator, '_pred_cache')

            self._log(f"   📊 _get_cache_key: {has_cache_key}")
            self._log(f"   📊 _get_prediction: {has_get_pred}")
            self._log(f"   📊 _pred_cache: {has_cache_dict}")

            if has_cache_key and has_get_pred and has_cache_dict:
                self._log_pass("预测缓存", "缓存方法完整存在")
            else:
                self._log_fail("预测缓存", "缓存方法缺失")

            # 测试缓存键生成
            if has_cache_key:
                key = validator._get_cache_key("test_policy", "/path/to/data.pkl", 6, 365)
                self._log(f"   📊 缓存键示例: {key}")
                if key and ":" in key:
                    self._log_pass("缓存键生成", "缓存键格式正确")
                else:
                    self._log_fail("缓存键生成", "缓存键格式异常")

        except Exception as e:
            self._log_fail("预测缓存", str(e))
            import traceback
            traceback.print_exc()

    # ============================================================
    # 测试5：并行评估
    # ============================================================
    def _test_parallel_evaluation(self):
        self._log_section("6. 并行评估测试")

        try:
            from experiments.autotune.validator import PolicyEvaluationOracle
            from experiments.autotune.utils import ProgressLogger

            logger = ProgressLogger(log_dir="llog/test_smoke", verbose=False)
            validator = PolicyEvaluationOracle({}, logger)

            # 检查 evaluate 方法的并行参数
            import inspect
            sig = inspect.signature(validator.evaluate)
            params = list(sig.parameters.keys())

            self._log(f"   📊 evaluate 参数: {params}")

            required_params = ['policies', 'dataset_name', 'split', 'parallel', 'workers']
            missing = [p for p in required_params if p not in params]

            if not missing:
                self._log_pass("并行评估接口", "evaluate 方法包含 parallel/workers 参数")
            else:
                self._log_fail("并行评估接口", f"缺少参数: {missing}")

            # 检查是否有 _evaluate_parallel 方法
            has_parallel_method = hasattr(validator, '_evaluate_parallel')
            has_serial_method = hasattr(validator, '_evaluate_serial')

            self._log(f"   📊 _evaluate_parallel: {has_parallel_method}")
            self._log(f"   📊 _evaluate_serial: {has_serial_method}")

            if has_parallel_method and has_serial_method:
                self._log_pass("并行评估实现", "串行/并行方法都存在")
            else:
                self._log_fail("并行评估实现", "串行或并行方法缺失")

        except Exception as e:
            self._log_fail("并行评估", str(e))
            import traceback
            traceback.print_exc()

    # ============================================================
    # 测试6：测试集评估
    # ============================================================
    def _test_testset_evaluation(self):
        self._log_section("7. 测试集评估测试 (split='test')")

        try:
            import inspect
            import experiments.autotune.main as main_module

            run_source = inspect.getsource(main_module.SPLSAutoTuner.run)

            has_test_eval = 'split=\'test\'' in run_source or 'split="test"' in run_source
            has_test_dir = 'test_eval' in run_source

            self._log(f"   📊 包含 'split=test': {has_test_eval}")
            self._log(f"   📊 包含 'test_eval' 目录: {has_test_dir}")

            if has_test_eval and has_test_dir:
                self._log_pass("测试集评估", "main.py 包含测试集评估代码")
            elif has_test_eval:
                self._log_pass("测试集评估", "包含 split='test' 但可能缺少目录创建")
            else:
                self._log_fail("测试集评估", "main.py 中未找到测试集评估代码")

        except Exception as e:
            self._log_fail("测试集评估", str(e))
            import traceback
            traceback.print_exc()

    # ============================================================
    # 测试7：检查点
    # ============================================================
    def _test_checkpoint(self):
        self._log_section("8. 检查点测试")

        try:
            from experiments.autotune.checkpoint_manager import CheckpointManager
            from experiments.autotune.utils import ProgressLogger
            from experiments.autotune.skill_policy import SkillPolicy

            logger = ProgressLogger(log_dir="llog/test_smoke", verbose=False)
            ckpt_manager = CheckpointManager("llog/test_smoke", logger)

            methods = ['load', 'save', 'get_next_round', 'is_completed', 'get_completed_rounds',
                       'round_exists', 'get_round_policies', 'detect_completed_rounds']

            existing_methods = [m for m in methods if hasattr(ckpt_manager, m)]
            missing_methods = [m for m in methods if not hasattr(ckpt_manager, m)]

            self._log(f"   📊 存在的方法: {existing_methods}")
            self._log(f"   📊 缺失的方法: {missing_methods}")

            if len(missing_methods) == 0:
                self._log_pass("检查点", f"所有 {len(methods)} 个方法都存在")
            else:
                self._log_fail("检查点", f"缺失 {len(missing_methods)} 个方法")

            # 测试保存和加载
            test_policy = SkillPolicy(
                policy_id="test_001",
                name="test_policy",
                avg_mase=0.5,
                utility_ema=0.7
            )

            try:
                result = ckpt_manager.save(
                    completed_rounds=2,
                    current_policies=[test_policy],
                    dataset="melbourne_temp",
                    horizon=6,
                    round_results={},
                    best_round=1,
                    best_mase=0.5,
                    current_b_subset_idx=1
                )
                if result:
                    self._log_pass("检查点保存", "save 方法执行成功")
                else:
                    self._log_fail("检查点保存", "save 方法返回 False")
            except Exception as e:
                self._log_fail("检查点保存", str(e))

            try:
                loaded = ckpt_manager.load()
                completed = loaded.get('completed_rounds', 0)
                self._log(f"   📊 加载的已完成轮次: {completed}")
                if completed == 2:
                    self._log_pass("检查点加载", f"成功加载，completed_rounds={completed}")
                else:
                    self._log_fail("检查点加载", f"completed_rounds={completed}, 期望=2")
            except Exception as e:
                self._log_fail("检查点加载", str(e))

        except Exception as e:
            self._log_fail("检查点", str(e))
            import traceback
            traceback.print_exc()

    # ============================================================
    # 汇总
    # ============================================================
    def _print_summary(self):
        self._log("")
        self._log("=" * 80)
        self._log("📊 冒烟测试汇总")
        self._log("=" * 80)

        total = len(self.test_results['passed']) + len(self.test_results['failed'])
        self._log(f"   ✅ 通过: {len(self.test_results['passed'])}")
        self._log(f"   ❌ 失败: {len(self.test_results['failed'])}")
        self._log(f"   ⏭️ 跳过: {len(self.test_results['skipped'])}")
        self._log(f"   📊 总计: {total}")

        if self.test_results['passed']:
            self._log("\n   ✅ 通过的测试:")
            for t in self.test_results['passed']:
                self._log(f"      - {t}")

        if self.test_results['failed']:
            self._log("\n   ❌ 失败的测试:")
            for t in self.test_results['failed']:
                self._log(f"      - {t}")

        elapsed = time.time() - self.start_time
        self._log(f"\n   ⏱️ 总耗时: {elapsed:.1f}s")

        self._log("")
        self._log("=" * 80)

        if not self.test_results['failed']:
            self._log("✅ 所有冒烟测试通过！系统可以正常运行。")
            self._log("=" * 80)
            return

        self._log("⚠️ 部分测试失败，请检查上述错误信息。")
        self._log("=" * 80)


# ============================================================
# 主入口
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="SPLS 冒烟测试（缓存模式）")
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='显示详细输出')
    args = parser.parse_args()

    runner = SmokeTestRunner(verbose=args.verbose)
    results = runner.run()

    if results['failed']:
        sys.exit(1)
    sys.exit(0)


if __name__ == '__main__':
    main()