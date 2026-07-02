#!/usr/bin/env python
"""
独立缓存补全脚本 - 为策略池中缺失缓存的策略构建缓存
可在主训练运行的同时，在另一个窗口执行

用法：
    # 使用默认文件名（r.pkl, rr.pkl）
    python -m experiments.autotune.build_missing_cache --resume llog/cs2 --workers 16

    # 使用自定义文件名
    python -m experiments.autotune.build_missing_cache --resume llog/cs2 --workers 16 --cache-b1 my_b1.pkl --cache-b2 my_b2.pkl

功能：
    1. 加载 cs2/checkpoint.json 获取当前策略池
    2. 加载指定的缓存文件（默认 r.pkl / rr.pkl）
    3. 找出每个子集中缺失缓存的策略
    4. 并行计算缺失的缓存项
    5. 合并写入对应的缓存文件
"""

import os
import sys
import argparse
import pickle
import time
import json
import hashlib
from typing import Dict, List, Optional, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
import threading
from tqdm import tqdm
import numpy as np
import pandas as pd

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from experiments.autotune.utils import load_window_data, compute_mase, load_config, ProgressLogger
from experiments.autotune.skill_policy import SkillPolicy
from experiments.autotune.checkpoint_manager import CheckpointManager


# ★★★ 顶层函数：供线程池调用 ★★★
def _process_cache_task(task):
    """
    处理单个缓存任务
    task: (policy, wpath, horizon_val)
    """
    policy, wpath, horizon_val = task
    policy_name = policy.name[:20] if policy.name else "unknown"
    try:
        wdata = load_window_data(wpath)
        train = wdata['train']
        test = wdata['test']
        period = wdata.get('period', 365)
        horizon = wdata.get('horizon', horizon_val)
        mase_scale = wdata.get('mase_scale', 1.0)

        pred = policy.execute(train, horizon, period)
        if pred is not None and len(pred) == len(test):
            mase = compute_mase(pred, test, mase_scale)
            return (policy.policy_id, wpath), {'pred': pred, 'mase': mase}
        else:
            return None
    except Exception as e:
        print(f"      ❌ [任务异常] {policy_name} @ {os.path.basename(wpath)}: {type(e).__name__}: {e}")
        return None


class MissingCacheBuilder:
    """缺失缓存补全构建器"""

    def __init__(self, run_dir: str, config_path: str = None,
                 workers: int = 16,
                 cache_b1_name: str = "r.pkl",
                 cache_b2_name: str = "rr.pkl"):
        self.run_dir = run_dir
        self.workers = workers
        self.cache_b1_name = cache_b1_name
        self.cache_b2_name = cache_b2_name
        self.config = load_config(config_path)
        self.logger = ProgressLogger(log_dir=run_dir, verbose=True, run_folder=False)
        self.logger.start_log("build_missing_cache")

        # 确保路径正确
        self.llog_dir = run_dir

        # 加载检查点
        self.checkpoint_manager = CheckpointManager(run_dir, self.logger)
        self.checkpoint = self.checkpoint_manager.load()

        # 获取策略列表
        self.policies = self.checkpoint.get('current_policies', [])
        if not self.policies:
            self.logger.log("❌ 检查点中没有策略，请先运行训练")
            sys.exit(1)

        self.logger.log(f"📋 加载策略: {len(self.policies)} 条")

        # 加载窗口数据
        self.output_dir = self.config.get('output_dir', 'storage/autotune_results')
        self._load_window_data()

        # 确定 B 子集划分
        self._load_b_subsets()

        # ★★★ 缓存文件路径（默认 r.pkl / rr.pkl） ★★★
        self.cache_file_b1 = os.path.join(run_dir, self.cache_b1_name)
        self.cache_file_b2 = os.path.join(run_dir, self.cache_b2_name)

        self.logger.log(f"📁 B1 缓存文件: {self.cache_file_b1}")
        self.logger.log(f"📁 B2 缓存文件: {self.cache_file_b2}")

        # 统计
        self.stats = {
            'b1': {'total_tasks': 0, 'cached': 0, 'missing': 0, 'success': 0, 'failed': 0},
            'b2': {'total_tasks': 0, 'cached': 0, 'missing': 0, 'success': 0, 'failed': 0}
        }

        # ★★★ 打印并行配置 ★★★
        self.logger.log(f"⚡ 并行线程数: {self.workers}")

    def _load_window_data(self):
        """加载采集的窗口数据"""
        csv_path = os.path.join(self.output_dir, "collected_windows.csv")
        if not os.path.exists(csv_path):
            self.logger.log(f"❌ 采集数据不存在: {csv_path}")
            sys.exit(1)

        self.df = pd.read_csv(csv_path)
        self.logger.log(f"📊 加载窗口数据: {len(self.df)} 个窗口")

    def _load_b_subsets(self):
        """确定 B1 和 B2 子集的窗口"""
        split_cfg = self.config.get('data_split', {})
        first_round_ratio = split_cfg.get('first_round_ratio', 0.50)
        B_ratio = split_cfg.get('B_ratio', 0.50)
        b_subset_count = self.config.get('evolution', {}).get('b_subset_count', 2)

        n_windows = len(self.df)
        a_end = int(n_windows * first_round_ratio)
        b_start = a_end
        b_end = int(n_windows * (first_round_ratio + B_ratio))
        if b_end > n_windows:
            b_end = n_windows

        df_b = self.df.iloc[b_start:b_end].copy()

        # 划分为 B1 和 B2
        b_subset_size = len(df_b) // b_subset_count if b_subset_count > 0 else len(df_b)
        self.b1_windows = []
        self.b2_windows = []

        for i in range(b_subset_count):
            start_idx = i * b_subset_size
            end_idx = (i + 1) * b_subset_size if i < b_subset_count - 1 else len(df_b)
            subset = df_b.iloc[start_idx:end_idx].copy()

            # 排除测试集（每个子集的前 1/3）
            test_size = len(subset) // 3
            train_subset = subset.iloc[test_size:].copy()

            if i == 0:
                self.b1_windows = train_subset.to_dict('records')
                self.logger.log(f"📊 B1 子集: {len(self.b1_windows)} 个窗口")
            elif i == 1:
                self.b2_windows = train_subset.to_dict('records')
                self.logger.log(f"📊 B2 子集: {len(self.b2_windows)} 个窗口")

    def _load_existing_cache(self, cache_file: str) -> Dict:
        """加载已有缓存"""
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cache = pickle.load(f)
                self.logger.log(f"   📂 加载已有缓存: {cache_file} ({len(cache)} 项)")
                return cache
            except Exception as e:
                self.logger.log(f"   ⚠️ 加载缓存失败: {e}，将重新创建")
                return {}
        else:
            self.logger.log(f"   📂 缓存文件不存在: {cache_file}，将创建新缓存")
            return {}

    def _save_cache(self, cache: Dict, cache_file: str):
        """保存缓存"""
        try:
            # 写入临时文件再重命名
            temp_file = cache_file + ".tmp"
            with open(temp_file, 'wb') as f:
                pickle.dump(cache, f)
            os.replace(temp_file, cache_file)
            self.logger.log(f"   💾 缓存已保存: {cache_file} ({len(cache)} 项)")
            return True
        except Exception as e:
            self.logger.log(f"   ❌ 保存缓存失败: {e}")
            return False

    def _get_window_data_paths(self, windows: List[Dict]) -> List[str]:
        """获取窗口数据路径列表"""
        paths = []
        for w in windows:
            wpath = w.get('window_data_path')
            if wpath and os.path.exists(wpath):
                paths.append(wpath)
        return paths

    def _find_missing_tasks(self, policies: List[SkillPolicy],
                            window_paths: List[str],
                            existing_cache: Dict) -> List[Tuple]:
        """
        找出缺失的缓存任务
        返回: [(policy, wpath, horizon_val), ...]
        """
        horizon_val = 12  # 从配置读取
        tasks = []
        existing_keys = set(existing_cache.keys())

        for wpath in window_paths:
            for policy in policies:
                if policy.status in ['ARCHIVE', 'DELETE']:
                    continue
                key = (policy.policy_id, wpath)
                if key not in existing_keys:
                    tasks.append((policy, wpath, horizon_val))

        return tasks

    def build_subset_cache(self, subset_name: str, windows: List[Dict],
                           cache_file: str, policies: List[SkillPolicy]) -> Dict:
        """
        为指定子集构建/补全缓存
        """
        self.logger.log(f"\n{'=' * 70}")
        self.logger.log(f"📦 构建 {subset_name} 缓存")
        self.logger.log(f"📁 缓存文件: {cache_file}")
        self.logger.log(f"{'=' * 70}")

        # 1. 加载已有缓存
        existing_cache = self._load_existing_cache(cache_file)

        # 2. 获取窗口路径
        window_paths = self._get_window_data_paths(windows)
        self.logger.log(f"   📊 {subset_name} 窗口数: {len(window_paths)}")

        # 3. 找出缺失任务
        tasks = self._find_missing_tasks(policies, window_paths, existing_cache)

        total_tasks = len(policies) * len(window_paths)
        cached_count = len(existing_cache)
        missing_count = len(tasks)

        self.logger.log(f"   📊 总任务: {total_tasks}, 已有缓存: {cached_count}, 缺失: {missing_count}")

        # 更新统计
        self.stats[subset_name.lower()]['total_tasks'] = total_tasks
        self.stats[subset_name.lower()]['cached'] = cached_count
        self.stats[subset_name.lower()]['missing'] = missing_count

        if missing_count == 0:
            self.logger.log(f"   ✅ {subset_name} 缓存完整，无需补全")
            return existing_cache

        # 4. 并行计算缺失任务
        self.logger.log(f"   ⚡ 开始并行计算 {missing_count} 个缺失任务（{self.workers} 线程）...")

        cache = existing_cache.copy()
        success_count = 0
        failed_count = 0
        cache_lock = threading.Lock()

        # 每个任务超时 120 秒
        TASK_TIMEOUT = 120

        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = {executor.submit(_process_cache_task, task): task for task in tasks}

            pbar = tqdm(total=len(futures), desc=f"   {subset_name} 缓存进度", unit="项", ncols=100)

            for future in as_completed(futures):
                try:
                    result = future.result(timeout=TASK_TIMEOUT)
                    if result is not None:
                        key, value = result
                        with cache_lock:
                            cache[key] = value
                        success_count += 1
                    else:
                        failed_count += 1
                except TimeoutError:
                    # 超时：取消该任务
                    future.cancel()
                    failed_count += 1
                    self.logger.log(f"      ⚠️ 任务超时 ({TASK_TIMEOUT}s)，跳过")
                except Exception as e:
                    failed_count += 1
                    self.logger.log(f"      ❌ 任务异常: {type(e).__name__}: {e}，跳过")

                pbar.update(1)
                pbar.set_postfix({
                    '成功': success_count,
                    '失败': failed_count,
                    '总缓存': len(cache)
                })

            pbar.close()

        # 5. 保存缓存
        self.stats[subset_name.lower()]['success'] = success_count
        self.stats[subset_name.lower()]['failed'] = failed_count

        self.logger.log(f"\n   ✅ {subset_name} 缓存补全完成:")
        self.logger.log(f"      - 新增成功: {success_count} 项")
        self.logger.log(f"      - 新增失败: {failed_count} 项")
        self.logger.log(f"      - 总缓存: {len(cache)} 项")

        if success_count > 0:
            self._save_cache(cache, cache_file)

        return cache

    def run(self):
        """执行缓存补全"""
        self.logger.log("=" * 70)
        self.logger.log("🚀 独立缓存补全工具")
        self.logger.log(f"📁 运行目录: {self.run_dir}")
        self.logger.log(f"📋 策略数: {len(self.policies)}")
        self.logger.log(f"⚡ 并行线程: {self.workers}")
        self.logger.log("=" * 70)

        start_time = time.time()

        # 构建 B1 缓存
        cache_b1 = self.build_subset_cache(
            "B1",
            self.b1_windows,
            self.cache_file_b1,
            self.policies
        )

        # 构建 B2 缓存
        cache_b2 = self.build_subset_cache(
            "B2",
            self.b2_windows,
            self.cache_file_b2,
            self.policies
        )

        elapsed = time.time() - start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)

        # 打印统计
        self.logger.log("\n" + "=" * 70)
        self.logger.log("📊 缓存补全完成统计")
        self.logger.log("=" * 70)

        total_cache = len(cache_b1) + len(cache_b2)
        total_missing = self.stats['b1']['missing'] + self.stats['b2']['missing']
        total_success = self.stats['b1']['success'] + self.stats['b2']['success']

        self.logger.log(f"   B1 缓存: {len(cache_b1)} 项 (新增 {self.stats['b1']['success']} 项)")
        self.logger.log(f"   B2 缓存: {len(cache_b2)} 项 (新增 {self.stats['b2']['success']} 项)")
        self.logger.log(f"   总缓存: {total_cache} 项")
        self.logger.log(f"   新增成功: {total_success} 项")
        self.logger.log(f"   总耗时: {minutes}分{seconds}秒")

        # 验证缓存文件
        self.logger.log("\n📁 缓存文件位置:")
        self.logger.log(f"   {self.cache_file_b1}")
        self.logger.log(f"   {self.cache_file_b2}")

        self.logger.log("\n✅ 缓存补全完成！")


def main():
    parser = argparse.ArgumentParser(description="独立缓存补全工具")
    parser.add_argument('--resume', type=str, required=True,
                        help='运行目录（如 llog/cs2）')
    parser.add_argument('--config', type=str, default=None,
                        help='配置文件路径')
    parser.add_argument('--workers', type=int, default=16,
                        help='并行线程数（默认 16）')
    parser.add_argument('--cache-b1', type=str, default='r.pkl',
                        help='B1 缓存文件名（默认 r.pkl）')
    parser.add_argument('--cache-b2', type=str, default='rr.pkl',
                        help='B2 缓存文件名（默认 rr.pkl）')

    args = parser.parse_args()

    # 处理运行目录
    run_dir = args.resume
    if not os.path.exists(run_dir):
        test_path = os.path.join("llog", args.resume)
        if os.path.exists(test_path):
            run_dir = test_path
        else:
            # 尝试查找 run_ 前缀
            if not args.resume.startswith("run_"):
                run_path = os.path.join("llog", f"run_{args.resume}")
                if os.path.exists(run_path):
                    run_dir = run_path

    if not os.path.exists(run_dir):
        print(f"❌ 运行目录不存在: {args.resume}")
        sys.exit(1)

    print(f"📁 使用运行目录: {run_dir}")
    print(f"⚡ 并行线程数: {args.workers}")
    print(f"📁 B1 缓存文件: {args.cache_b1}")
    print(f"📁 B2 缓存文件: {args.cache_b2}")

    builder = MissingCacheBuilder(
        run_dir=run_dir,
        config_path=args.config,
        workers=args.workers,
        cache_b1_name=args.cache_b1,
        cache_b2_name=args.cache_b2
    )

    builder.run()


if __name__ == '__main__':
    main()