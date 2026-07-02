# experiments/autotune/collector.py
"""
State Window Generator - 支持增量采集（自动补充缺失窗口）
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from tqdm import tqdm
import json
import traceback
import hashlib
import pickle
from datetime import datetime

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from experiments.autotune.utils import (
    ProgressLogger, MemoryCache,
    extract_features, detect_period,
    compute_mase, serialize_trajectory, save_window_data
)
from experiments.autotune.skill_policy import SkillPolicy


class StateWindowGenerator:
    """状态窗口生成器 - 支持增量采集（自动补充缺失窗口）"""

    def __init__(self, config: Dict, logger: ProgressLogger, cache: MemoryCache):
        self.config = config
        self.logger = logger
        self.cache = cache
        self.results = []
        self._verbose = False
        self.trigger = None
        self.policies = []
        self._cache_hits = 0
        self._cache_misses = 0
        self.skip_if_exists = config.get('skip_collection', True)

    def set_verbose(self, verbose: bool):
        self._verbose = verbose

    def set_trigger(self, trigger):
        self.trigger = trigger

    def set_policies(self, policies: List[SkillPolicy]):
        self.policies = policies

    def generate(self, dataset_name: str, window_sizes: List[int],
                 horizon: int, step: int,
                 max_train_size: Optional[int] = None) -> List[Dict]:
        """
        生成窗口，若已有缓存则检查完整性，缺失则增量补充
        """
        output_dir = self.config.get('output_dir', 'storage/autotune_results')
        csv_path = os.path.join(output_dir, "collected_windows.csv")

        # ★ 检查是否已有采集结果
        if self.skip_if_exists and os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                existing_count = len(df)
                self.logger.log(f"📁 加载已有数据: {csv_path} ({existing_count} 个窗口)")

                # ★ 计算预期窗口数
                series, freq = self._load_dataset(dataset_name)
                if series is None:
                    self.logger.log(f"❌ 无法加载数据集: {dataset_name}")
                    return []

                period = detect_period(series, freq)
                expected_count = 0
                for w in window_sizes:
                    if max_train_size is not None:
                        max_start = min(max_train_size - w, len(series) - w - horizon)
                    else:
                        max_start = len(series) - w - horizon
                    if max_start >= 0:
                        expected_count += len(list(range(0, max_start + 1, step)))

                # ★ 如果窗口数量不足，进行增量补充
                if existing_count < expected_count:
                    self.logger.log(f"⚠️ 缓存窗口不完整（现有 {existing_count}，预期 {expected_count}），增量补充...")
                    # 获取已有窗口ID集合
                    existing_ids = set(df['window_id'].tolist()) if 'window_id' in df.columns else set()

                    # 生成缺失的窗口（增量模式）
                    missing_results = self._generate_incremental(
                        dataset_name, series, freq, period,
                        window_sizes, horizon, step, max_train_size,
                        existing_ids, output_dir
                    )

                    if missing_results:
                        # 合并已有和新增
                        all_results = df.to_dict('records') + missing_results
                        # 重新保存完整CSV
                        new_df = pd.DataFrame(all_results)
                        new_df.to_csv(csv_path, index=False)
                        self.logger.log(
                            f"✅ 增量采集完成: 原有 {existing_count} 个，新增 {len(missing_results)} 个，总计 {len(all_results)} 个")
                        self.results = all_results
                        return all_results
                    else:
                        self.logger.log(f"✅ 窗口已完整，无需补充")
                        self.results = df.to_dict('records')
                        return self.results
                else:
                    self.logger.log(f"✅ 窗口完整 ({existing_count} 个)")
                    self.results = df.to_dict('records')
                    return self.results

            except Exception as e:
                self.logger.log(f"   ⚠️ 加载/校验失败: {e}，重新采集")

        # ★ 如果没有缓存或缓存无效，重新采集全部
        if self._verbose:
            self.logger.log(f"\n📊 生成窗口: {dataset_name}")

        series, freq = self._load_dataset(dataset_name)
        if series is None:
            self.logger.log(f"❌ 无法加载数据集: {dataset_name}")
            return []

        period = detect_period(series, freq)
        mase_scale = self._compute_mase_scale(series, period)

        if self._verbose:
            self.logger.log(f"   📈 数据长度: {len(series)}, 周期: {period}")

        fixed_params = self.config.get('fixed_params', {})
        for key, value in fixed_params.items():
            os.environ[f"TUNE_{key}"] = str(value)

        local_window_sizes = self.config.get('local_window_sizes', [7, 30])

        total_windows = 0
        for w in window_sizes:
            if max_train_size is not None:
                max_start = min(max_train_size - w, len(series) - w - horizon)
            else:
                max_start = len(series) - w - horizon
            if max_start >= 0:
                total_windows += len(list(range(0, max_start + 1, step)))

        all_results = []
        failed_total = 0
        success_total = 0
        window_counter = 0

        self._cache_hits = 0
        self._cache_misses = 0

        pbar = tqdm(total=total_windows, desc=f"生成窗口 {dataset_name}", unit="窗口", ncols=100)

        for window_size in window_sizes:
            if max_train_size is not None:
                max_start = min(max_train_size - window_size, len(series) - window_size - horizon)
            else:
                max_start = len(series) - window_size - horizon

            if max_start < 0:
                continue

            start_points = list(range(0, max_start + 1, step))

            for origin in start_points:
                window_counter += 1
                train = series[origin:origin + window_size]
                test = series[origin + window_size:origin + window_size + horizon]
                features = extract_features(train, local_window_sizes=local_window_sizes)
                features['window_size'] = window_size

                try:
                    mase, trajectory = self._run_prediction(train, test, horizon, period, mase_scale)
                except Exception as e:
                    self.logger.log(f"❌ 窗口 {window_counter} 预测异常: {e}")
                    failed_total += 1
                    pbar.update(1)
                    continue

                if mase == float('inf') or mase is None:
                    failed_total += 1
                    pbar.update(1)
                    continue

                success_total += 1

                window_data_path = save_window_data(
                    train, test, period, mase_scale, features,
                    window_counter, dataset_name, horizon
                )

                all_results.append({
                    'dataset': dataset_name,
                    'window_id': window_counter,
                    'origin': origin,
                    'window_size': window_size,
                    'train_size': len(train),
                    'test_size': len(test),
                    'horizon': horizon,
                    'period': period,
                    'mase_scale': mase_scale,
                    'window_data_path': window_data_path,
                    **features,
                    'best_trajectory': serialize_trajectory(trajectory),
                    'best_mase': mase,
                    'split': 'unknown'
                })

                pbar.set_postfix({
                    'size': window_size,
                    'origin': origin,
                    'best': f'{mase:.4f}',
                    'ok': success_total
                })
                pbar.update(1)

        pbar.close()

        for key in fixed_params.keys():
            os.environ.pop(f"TUNE_{key}", None)

        if self._verbose:
            self.logger.log(f"\n   ✅ 成功: {success_total} 个窗口, 失败: {failed_total} 个窗口")
            self.logger.log(f"   📊 缓存命中: {self._cache_hits}, 缓存未命中: {self._cache_misses}")

        self.results.extend(all_results)
        return all_results

    def _generate_incremental(self, dataset_name: str, series: np.ndarray,
                              freq: str, period: int,
                              window_sizes: List[int], horizon: int,
                              step: int, max_train_size: Optional[int],
                              existing_ids: set, output_dir: str) -> List[Dict]:
        """
        增量生成缺失的窗口（只生成不在 existing_ids 中的窗口）
        """
        local_window_sizes = self.config.get('local_window_sizes', [7, 30])
        mase_scale = self._compute_mase_scale(series, period)
        fixed_params = self.config.get('fixed_params', {})
        for key, value in fixed_params.items():
            os.environ[f"TUNE_{key}"] = str(value)

        missing_results = []
        window_counter = 0

        for window_size in window_sizes:
            if max_train_size is not None:
                max_start = min(max_train_size - window_size, len(series) - window_size - horizon)
            else:
                max_start = len(series) - window_size - horizon

            if max_start < 0:
                continue

            start_points = list(range(0, max_start + 1, step))

            for origin in start_points:
                window_counter += 1
                # ★ 检查是否已存在（使用 window_id 判断）
                if window_counter in existing_ids:
                    continue

                train = series[origin:origin + window_size]
                test = series[origin + window_size:origin + window_size + horizon]
                features = extract_features(train, local_window_sizes=local_window_sizes)
                features['window_size'] = window_size

                try:
                    mase, trajectory = self._run_prediction(train, test, horizon, period, mase_scale)
                except Exception as e:
                    self.logger.log(f"❌ 窗口 {window_counter} 预测异常: {e}")
                    continue

                if mase == float('inf') or mase is None:
                    continue

                window_data_path = save_window_data(
                    train, test, period, mase_scale, features,
                    window_counter, dataset_name, horizon
                )

                missing_results.append({
                    'dataset': dataset_name,
                    'window_id': window_counter,
                    'origin': origin,
                    'window_size': window_size,
                    'train_size': len(train),
                    'test_size': len(test),
                    'horizon': horizon,
                    'period': period,
                    'mase_scale': mase_scale,
                    'window_data_path': window_data_path,
                    **features,
                    'best_trajectory': serialize_trajectory(trajectory),
                    'best_mase': mase,
                    'split': 'unknown'
                })

                # 每生成一个就追加保存（防止中断丢失）
                if len(missing_results) % 10 == 0:
                    self.logger.log(f"   📝 已补充 {len(missing_results)} 个窗口...")

        for key in fixed_params.keys():
            os.environ.pop(f"TUNE_{key}", None)

        self.logger.log(f"   ✅ 增量生成完成: {len(missing_results)} 个新窗口")
        return missing_results

    def _load_dataset(self, dataset_name: str) -> Tuple[Optional[np.ndarray], Optional[str]]:
        try:
            from src.dataset.registry import DatasetRegistry
            from src.dataset.loader import load_dataset
            registry = DatasetRegistry()
            ds_config = registry.get(dataset_name)
            if not ds_config:
                return None, None
            df = load_dataset(ds_config)
            target_col = ds_config['target_column']
            series = df[target_col].values
            freq = ds_config.get('frequency', 'daily')
            return series, freq
        except Exception as e:
            self.logger.log(f"⚠️ 加载数据集失败: {e}")
            return None, None

    def _compute_mase_scale(self, series: np.ndarray, period: int) -> float:
        n = len(series)
        if n >= 2 * period:
            seasonal_errors = np.abs(series[period:] - series[:-period])
            return np.mean(seasonal_errors) if len(seasonal_errors) > 0 else 1.0
        else:
            naive_errors = np.abs(np.diff(series))
            return np.mean(naive_errors) if len(naive_errors) > 0 else 1.0

    def _run_prediction(self, train: np.ndarray, test: np.ndarray,
                        horizon: int, period: int, mase_scale: float) -> Tuple[float, List]:
        try:
            train_bytes = pickle.dumps(train)
            train_hash = hashlib.md5(train_bytes).hexdigest()
            cache_key = f"pred_{train_hash}_{horizon}"

            if self.cache.exists(cache_key):
                cached = self.cache.get(cache_key)
                if cached and 'pred' in cached and 'trajectory' in cached:
                    pred = np.array(cached['pred'])
                    if len(pred) == len(test):
                        self._cache_hits += 1
                        return compute_mase(pred, test, mase_scale), cached['trajectory']

            self._cache_misses += 1

            from src.agents.llm_planner import LLMPlannerAgent
            from src.tasks.instance import TaskInstance
            from run_benchmark import build_full_registry

            full_registry, _ = build_full_registry()
            agent = LLMPlannerAgent(
                model="glm-4.5-air",
                skill_registry=full_registry,
                log_file=None,
                use_skills=True,
                logger=self.logger  # ★ 新增：传入 logger
            )

            task = TaskInstance(
                id=f"collect_{len(train)}_{len(test)}",
                dataset_id="autotune",
                template_id="fixed_origin",
                question=f"预测未来{len(test)}步",
                question_type="numerical",
                history=train.tolist(),
                horizon=len(test),
                frequency="daily",
                prediction_target={},
                resolution_date=datetime.now(),
                difficulty_level=1,
                ground_truth_extractor="",
                dates=None,
                target_date=""
            )

            pred, trajectory = agent.predict_with_trajectory(task)
            pred_array = np.array(pred)

            self.cache.set(cache_key, {
                'pred': pred_array.tolist(),
                'train_size': len(train),
                'test_size': len(test),
                'trajectory': trajectory
            })

            if len(pred_array) != len(test):
                return float('inf'), trajectory

            mase = compute_mase(pred_array, test, mase_scale)
            return mase, trajectory

        except Exception as e:
            self.logger.log(f"❌ _run_prediction 异常: {e}")
            self.logger.log(traceback.format_exc())
            return float('inf'), []

    def save_results(self, output_path: str):
        if not self.results:
            return None
        df = pd.DataFrame(self.results)
        output_file = os.path.join(output_path, "collected_windows.csv")
        os.makedirs(output_path, exist_ok=True)
        df.to_csv(output_file, index=False)
        self.logger.log(f"📁 采集结果已保存: {output_file}")
        return output_file