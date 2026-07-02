#!/usr/bin/env python
"""
策略执行器（阶段2）：加载已生成的策略（均为 LLM 生成的 stages），在本地执行并计算 MASE
★ 所有模式统一处理：从策略文件读取 stages，构建临时 SkillPolicy 执行预测
★ 保存每个模式每个窗口的 12 步预测值到 .npy 文件
★ 支持断点续跑
★ 双目录输出：全部模式 / 剔除 no_rule
★ ★ ★ 自动迁移旧版单目录数据，自动从 predictions 恢复缺失模式
★ ★ ★ 如果预测文件已存在（包括旧目录迁移），则跳过执行，避免重复计算
★ ★ ★ 报告中 MASE 列自动标注相对于 no_rule 的百分比变化
★ ★ ★ 支持 --half-mode all，一次性处理全部测试窗口（不截取），并强制刷新所有模式数据
"""

import os
import sys
import json
import time
import math
import threading
import concurrent.futures
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from experiments.autotune.utils import (
    load_config, load_window_data, compute_all_metrics
)
from experiments.autotune.skill_policy import SkillPolicy
from run_benchmark import build_full_registry

WINDOW_TIMEOUT = 120


class StrategyExecutor:
    """策略执行器 - 阶段2，双目录输出"""

    def __init__(self, run_dir: str, round_num: int, half_mode: str = 'first',
                 config_path: str = None, workers: int = 8, force: bool = False):

        self.run_dir = run_dir
        self.round_num = round_num
        self.half_mode = half_mode
        self.force = force
        self.config = load_config(config_path)
        self.output_dir = self.config.get('output_dir', 'storage/autotune_results')
        self.test_workers = workers
        self._timeout_counter = 0
        self._lock = threading.Lock()

        # 输出目录
        self.run_dir_out = run_dir + "_half"
        self.strategies_dir = os.path.join(self.run_dir_out, "generated_strategies")
        self.results_dir_all = os.path.join(self.run_dir_out, "semantic_vs_rl_results_all")
        self.results_dir_no_no_rule = os.path.join(self.run_dir_out, "semantic_vs_rl_results_no_no_rule")
        self.predictions_dir = os.path.join(self.results_dir_all, "predictions")  # 预测值只存一份

        os.makedirs(self.results_dir_all, exist_ok=True)
        os.makedirs(self.results_dir_no_no_rule, exist_ok=True)
        os.makedirs(self.predictions_dir, exist_ok=True)

        print("   🔧 构建技能注册表...")
        self.full_registry, self.all_skills = build_full_registry()

        self.test_df = self._load_test_df()
        self.policies = self._load_round_policies(round_num)

        # ★★★ 所有模式定义（与 generate 阶段一致） ★★★
        self.modes = [
            'no_rule',
            'semantic_top1',
            'semantic_top30_theta_max',
            'semantic_top40_theta_max',
            'semantic_top50_theta_max',
            'semantic_top60_theta_max',
            'semantic_top70_theta_max',
            'rl_top10_semantic_best',
            'rl_top20_semantic_best',
            'rl_top30_semantic_best',
            'rl_top40_semantic_best',
            'rl_top50_semantic_best',
            'rl_top60_semantic_best',
            'rl_top3_semantic_best',
        ]

        # 为每个模式创建预测值子目录（但迁移时会自动创建）
        for mode in self.modes:
            mode_pred_dir = os.path.join(self.predictions_dir, mode)
            os.makedirs(mode_pred_dir, exist_ok=True)

        # ★★★ 迁移旧版预测文件（从旧目录迁移到新目录） ★★★
        self._migrate_predictions()

        # 加载已有执行状态
        self._status = self._load_status()
        # 加载生成的策略
        self._strategies = self._load_strategies()

        # ★★★ 自动迁移旧版单目录数据（results.json） ★★★
        self._migrate_legacy_data()

        # 加载已有结果（两个目录分别加载）
        self._results_all = self._load_existing_results(self.results_dir_all)
        self._results_no_no_rule = self._load_existing_results(self.results_dir_no_no_rule)

        print(f"\n📊 策略执行器初始化完成")
        print(f"   📁 策略目录: {self.strategies_dir}")
        print(f"   📁 结果目录(全部): {self.results_dir_all}")
        print(f"   📁 结果目录(剔除no_rule): {self.results_dir_no_no_rule}")
        print(f"   📁 预测值目录: {self.predictions_dir}")
        print(f"   📋 模式: {self.modes}")
        print(f"   📊 窗口数: {len(self.test_df)}")
        print(f"   ⚡ 并发数: {self.test_workers}")
        print(f"   🔄 已执行: {self._count_executed()} 个")

        if self.force:
            print("   ⚠️ 强制模式：将重新执行所有窗口")

    def _log(self, msg: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {msg}")

    def _load_test_df(self) -> pd.DataFrame:
        csv_path = os.path.join(self.output_dir, "collected_windows.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"❌ 未找到采集数据: {csv_path}")

        df = pd.read_csv(csv_path)

        # 获取测试集
        if 'split' in df.columns:
            test_df = df[df['split'] == 'test'].copy()
        else:
            n = len(df)
            a_end = int(n * 0.5)
            b_mask = pd.Series([False] * n)
            b_mask.iloc[a_end:] = True
            df['split'] = ['A'] * a_end + ['B'] * (n - a_end)
            b_df = df[b_mask].copy().sort_values('window_id').reset_index(drop=True)
            n_b = len(b_df)
            test_size = int(n_b * 0.5)
            test_df = b_df.iloc[:test_size].copy()

        test_df = test_df.sort_values('window_id').reset_index(drop=True)

        if self.half_mode == 'first':
            test_df = test_df.head(25).copy()
        elif self.half_mode == 'second':
            test_df = test_df.tail(25).copy()
        elif self.half_mode == 'all':
            # 全部窗口，不做截取
            pass
        else:
            raise ValueError(f"half_mode 必须为 'first'、'second' 或 'all'，得到 {self.half_mode}")

        # 构建窗口ID到数据的映射（用于恢复时快速查找）
        self._window_id_to_data = {}
        for _, row in test_df.iterrows():
            wid = row.get('window_id')
            wpath = row.get('window_data_path')
            if wid is not None and wpath and os.path.exists(wpath):
                try:
                    wdata = load_window_data(wpath)
                    self._window_id_to_data[wid] = {
                        'test': wdata['test'],
                        'mase_scale': wdata.get('mase_scale', 1.0),
                        'horizon': wdata.get('horizon', 7),
                        'train_mean': np.mean(wdata['train']),
                        'period': wdata.get('period', 365),
                    }
                except:
                    pass

        print(f"📊 使用测试集 ({self.half_mode})，共 {len(test_df)} 个窗口")
        return test_df

    def _load_round_policies(self, round_num: int) -> List[SkillPolicy]:
        path = os.path.join(self.run_dir, f"round_{round_num}", "refined_policies_optimized.json")
        if not os.path.exists(path):
            path = os.path.join(self.run_dir, f"round_{round_num}", "refined_policies_raw.json")
        if not os.path.exists(path):
            print(f"❌ 未找到第 {round_num} 轮策略文件")
            return []

        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            policies = [SkillPolicy.from_dict(p) for p in data.get('policies', [])]
            print(f"📋 加载第 {round_num} 轮策略: {len(policies)} 条")
            return policies
        except Exception as e:
            print(f"⚠️ 加载失败: {e}")
            return []

    def _load_strategies(self) -> Dict[str, Dict[str, Dict]]:
        strategies = {}
        for mode in self.modes:
            mode_file = os.path.join(self.strategies_dir, f"{mode}_strategies.json")
            if os.path.exists(mode_file):
                try:
                    with open(mode_file, 'r', encoding='utf-8') as f:
                        strategies[mode] = json.load(f)
                    self._log(f"   📂 加载策略: {mode} ({len(strategies[mode])} 个)")
                except Exception as e:
                    self._log(f"   ⚠️ 加载 {mode} 策略失败: {e}")
                    strategies[mode] = {}
            else:
                self._log(f"   ⚠️ 未找到 {mode} 策略文件，该模式将使用均值回退")
                strategies[mode] = {}
        return strategies

    def _migrate_predictions(self):
        """
        将旧目录（semantic_vs_rl_results/predictions）中的预测文件迁移到新目录。
        如果新目录已存在同名文件，则跳过。
        """
        old_pred_dir = os.path.join(self.run_dir_out, "semantic_vs_rl_results", "predictions")
        if not os.path.exists(old_pred_dir):
            return

        self._log("   📂 检测到旧版预测目录，正在迁移预测文件...")
        migrated_count = 0
        for mode in self.modes:
            src_mode_dir = os.path.join(old_pred_dir, mode)
            if not os.path.exists(src_mode_dir):
                continue
            dst_mode_dir = os.path.join(self.predictions_dir, mode)
            os.makedirs(dst_mode_dir, exist_ok=True)
            for fname in os.listdir(src_mode_dir):
                if fname.startswith('window_') and fname.endswith('.npy'):
                    src_file = os.path.join(src_mode_dir, fname)
                    dst_file = os.path.join(dst_mode_dir, fname)
                    if not os.path.exists(dst_file):
                        shutil.copy2(src_file, dst_file)
                        migrated_count += 1
        if migrated_count > 0:
            self._log(f"   ✅ 迁移了 {migrated_count} 个预测文件到新目录")
        else:
            self._log("   ℹ️ 没有需要迁移的新预测文件（可能已存在）")

    def _load_status(self) -> Dict:
        status_file = os.path.join(self.results_dir_all, "execution_status.json")
        if os.path.exists(status_file):
            try:
                with open(status_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return {"executed": {}}

    def _save_status(self):
        status_file = os.path.join(self.results_dir_all, "execution_status.json")
        with open(status_file, 'w', encoding='utf-8') as f:
            json.dump(self._status, f, ensure_ascii=False, indent=2)

    def _load_existing_results(self, results_dir: str) -> Dict:
        """加载指定目录的 results.json，自动兼容扁平/嵌套格式"""
        json_path = os.path.join(results_dir, "results.json")
        if not os.path.exists(json_path):
            return {'results': {}}

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if 'results' in data and isinstance(data['results'], dict):
                return data
            else:
                nested = {'results': data}
                try:
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(nested, f, ensure_ascii=False, indent=2)
                    self._log(f"   🔄 已自动将 {results_dir}/results.json 转换为嵌套格式")
                except:
                    pass
                return nested
        except Exception as e:
            self._log(f"   ⚠️ 读取 {results_dir}/results.json 失败: {e}")
            return {'results': {}}

    def _save_results_to_dir(self, results_dir: str, mode_data: Dict):
        """保存模式数据到指定目录（合并模式）"""
        json_path = os.path.join(results_dir, "results.json")
        existing = self._load_existing_results(results_dir)
        existing_data = existing.get('results', {})
        for mode, data in mode_data.items():
            existing_data[mode] = data
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({'results': existing_data}, f, ensure_ascii=False, indent=2)

    def _save_results(self, mode_data: Dict):
        """
        保存数据到两个目录：
        - _all：全部模式
        - _no_no_rule：剔除 no_rule
        """
        # 保存全部
        self._save_results_to_dir(self.results_dir_all, mode_data)

        # 保存剔除 no_rule
        filtered = {k: v for k, v in mode_data.items() if k != 'no_rule'}
        if filtered:
            self._save_results_to_dir(self.results_dir_no_no_rule, filtered)

    def _migrate_legacy_data(self):
        """
        如果存在旧版单目录 semantic_vs_rl_results，将其数据迁移到两个新目录。
        迁移后旧目录保留不动。
        """
        legacy_dir = os.path.join(self.run_dir_out, "semantic_vs_rl_results")
        if not os.path.exists(legacy_dir):
            return

        legacy_json = os.path.join(legacy_dir, "results.json")
        if not os.path.exists(legacy_json):
            return

        self._log("   📂 检测到旧版单目录，正在迁移数据...")
        legacy_data = self._load_existing_results(legacy_dir)
        if 'results' not in legacy_data or not legacy_data['results']:
            return

        # 合并到新目录
        for target_dir in [self.results_dir_all, self.results_dir_no_no_rule]:
            existing = self._load_existing_results(target_dir)
            existing_data = existing.get('results', {})
            # 合并所有模式
            for mode, data in legacy_data['results'].items():
                if target_dir == self.results_dir_no_no_rule and mode == 'no_rule':
                    continue
                existing_data[mode] = data
            # 保存
            with open(os.path.join(target_dir, "results.json"), 'w', encoding='utf-8') as f:
                json.dump({'results': existing_data}, f, ensure_ascii=False, indent=2)
        self._log("   ✅ 旧数据迁移完成")

    def _prediction_exists(self, mode: str, window_id: int) -> bool:
        """检查预测文件是否已存在"""
        pred_file = os.path.join(self.predictions_dir, mode, f"window_{window_id}.npy")
        return os.path.exists(pred_file)

    def _restore_missing_modes_from_predictions(self, force_refresh_all: bool = False):
        """
        从 predictions 目录恢复模式数据。
        如果 force_refresh_all 为 True，则强制对所有模式重新从 predictions 读取并覆盖现有数据。
        否则只补充缺失的模式。
        """
        self._log("\n🔍 检查是否有缺失模式需要从 predictions 恢复...")

        # 先确定当前两个目录已有的模式集合
        all_existing = self._load_existing_results(self.results_dir_all).get('results', {})
        no_no_rule_existing = self._load_existing_results(self.results_dir_no_no_rule).get('results', {})

        restored_all = {}
        restored_no_no_rule = {}

        for mode in self.modes:
            # 如果强制刷新，或者模式缺失，则重新/从 predictions 恢复
            if force_refresh_all or mode not in all_existing or not all_existing[mode].get('mases'):
                mode_data = self._restore_single_mode_from_predictions(mode)
                if mode_data:
                    restored_all[mode] = mode_data
                    if mode != 'no_rule':
                        restored_no_no_rule[mode] = mode_data
                    self._log(f"      {'🔄 刷新' if force_refresh_all else '✅ 恢复'}模式 {mode}: {len(mode_data['mases'])} 个窗口")
            else:
                # 如果 _all 有，但 _no_no_rule 缺失（no_rule 除外）
                if mode != 'no_rule' and mode not in no_no_rule_existing:
                    mode_data = self._load_existing_results(self.results_dir_all).get('results', {}).get(mode, {})
                    if mode_data:
                        restored_no_no_rule[mode] = mode_data

        if restored_all:
            self._save_results(restored_all)
            self._log(f"   ✅ 共{'刷新' if force_refresh_all else '恢复'} {len(restored_all)} 个模式的数据到全部目录")
        if restored_no_no_rule:
            self._save_results_to_dir(self.results_dir_no_no_rule, restored_no_no_rule)
            self._log(f"   ✅ 共{'刷新' if force_refresh_all else '恢复'} {len(restored_no_no_rule)} 个模式的数据到剔除目录")

    def _restore_single_mode_from_predictions(self, mode: str) -> Optional[Dict]:
        """从 predictions 恢复单个模式的数据"""
        mode_pred_dir = os.path.join(self.predictions_dir, mode)
        if not os.path.exists(mode_pred_dir):
            return None

        npy_files = [f for f in os.listdir(mode_pred_dir) if f.startswith('window_') and f.endswith('.npy')]
        if not npy_files:
            return None

        mases, maes, rmses, smapes, owas = [], [], [], [], []
        window_mases = {}

        for npy_file in npy_files:
            try:
                wid_str = npy_file.replace('window_', '').replace('.npy', '')
                wid = int(wid_str)
                if wid not in self._window_id_to_data:
                    continue
                win_data = self._window_id_to_data[wid]
                pred = np.load(os.path.join(mode_pred_dir, npy_file))
                if len(pred) != len(win_data['test']):
                    continue
                metrics = compute_all_metrics(pred, win_data['test'], win_data['mase_scale'])
                mase = metrics.get('mase', float('inf'))
                if mase != float('inf') and not np.isnan(mase):
                    mases.append(mase)
                    maes.append(metrics.get('mae', 0))
                    rmses.append(metrics.get('rmse', 0))
                    smapes.append(metrics.get('smape', 0))
                    owas.append(metrics.get('owa', 0))
                    window_mases[wid] = mase
            except Exception as e:
                continue

        if mases:
            return {
                'mases': mases,
                'maes': maes,
                'rmses': rmses,
                'smapes': smapes,
                'owas': owas,
                'window_mases': window_mases,
            }
        return None

    def _count_executed(self) -> int:
        count = 0
        for mode in self.modes:
            mode_key = f"{mode}_windows"
            if mode_key in self._status.get('executed', {}):
                count += len(self._status['executed'][mode_key])
        return count

    def _get_executed_windows(self, mode: str) -> set:
        """
        返回已执行的窗口ID集合。
        优先从 status 读取，同时检查预测文件是否存在（如果存在也视为已执行）。
        """
        mode_key = f"{mode}_windows"
        if self.force:
            return set()
        executed = set()
        if 'executed' in self._status and mode_key in self._status['executed']:
            executed.update(self._status['executed'][mode_key])
        # 补充检查 predictions 文件是否存在
        for wid in self._window_id_to_data.keys():
            if self._prediction_exists(mode, wid):
                executed.add(wid)
        return executed

    def _mark_executed(self, mode: str, window_id: int):
        mode_key = f"{mode}_windows"
        if 'executed' not in self._status:
            self._status['executed'] = {}
        if mode_key not in self._status['executed']:
            self._status['executed'][mode_key] = []
        if window_id not in self._status['executed'][mode_key]:
            self._status['executed'][mode_key].append(window_id)
        self._save_status()

    def _save_prediction(self, mode: str, window_id: int, pred: np.ndarray):
        mode_pred_dir = os.path.join(self.predictions_dir, mode)
        os.makedirs(mode_pred_dir, exist_ok=True)
        pred_file = os.path.join(mode_pred_dir, f"window_{window_id}.npy")
        np.save(pred_file, pred)

    def _execute_strategy(self, strategy: Dict, train: np.ndarray,
                          horizon: int, period: int) -> Optional[np.ndarray]:
        try:
            from experiments.autotune.skill_policy import SkillPolicy
            import hashlib, time
            temp_policy = SkillPolicy(
                policy_id=hashlib.md5(f"temp_{time.time()}".encode()).hexdigest()[:8],
                name="temp_policy",
                skill_strategy=strategy,
                avg_mase=1.0
            )
            pred = temp_policy.execute(train, horizon, period)
            if pred is not None and len(pred) == horizon:
                return pred
            return None
        except Exception:
            return None

    def process_window(self, mode: str, row: pd.Series) -> Optional[Dict]:
        window_id = row.get('window_id', 'unknown')
        window_data_path = row.get('window_data_path')

        if window_id in self._get_executed_windows(mode):
            return None

        if not window_data_path or not os.path.exists(window_data_path):
            self._log(f"   ⚠️ 窗口 {window_id} 数据路径不存在")
            return None

        try:
            wdata = load_window_data(window_data_path)
            train = wdata['train']
            test = wdata['test']
            period = wdata.get('period', 365)
            mase_scale = wdata.get('mase_scale', 1.0)
            horizon = wdata.get('horizon', 7)

            strategy = self._strategies.get(mode, {}).get(str(window_id))

            if strategy is None:
                self._log(f"   ⚠️ 窗口 {window_id} 模式 {mode} 无策略，使用均值回退")
                pred = np.full(horizon, np.mean(train))
            else:
                pred = self._execute_strategy(strategy, train, horizon, period)
                if pred is None or len(pred) != horizon:
                    pred = np.full(horizon, np.mean(train))

            self._save_prediction(mode, window_id, pred)
            metrics = compute_all_metrics(pred, test, mase_scale)
            self._mark_executed(mode, window_id)

            return {
                'window_id': window_id,
                'mode': mode,
                'mase': metrics.get('mase', float('inf')),
                'mae': metrics.get('mae', float('inf')),
                'rmse': metrics.get('rmse', float('inf')),
                'smape': metrics.get('smape', float('inf')),
                'owa': metrics.get('owa', float('inf')),
            }

        except Exception as e:
            self._log(f"   ❌ 窗口 {window_id} 处理异常: {e}")
            return None

    def execute_all(self):
        self._log("\n" + "=" * 80)
        self._log("🚀 阶段2：策略执行（本地执行 + 计算 MASE）")
        self._log(f"📁 策略目录: {self.strategies_dir}")
        self._log(f"📁 结果目录(全部): {self.results_dir_all}")
        self._log(f"📁 结果目录(剔除no_rule): {self.results_dir_no_no_rule}")
        self._log(f"📁 预测值目录: {self.predictions_dir}")
        self._log("=" * 80)

        tasks = []
        for _, row in self.test_df.iterrows():
            window_id = row.get('window_id')
            if window_id is None:
                continue
            for mode in self.modes:
                if not self.force and window_id in self._get_executed_windows(mode):
                    continue
                strategy = self._strategies.get(mode, {}).get(str(window_id))
                if strategy is None:
                    self._log(f"   ⚠️ 窗口 {window_id} 模式 {mode} 无策略，将使用均值回退")
                tasks.append({'mode': mode, 'row': row})

        # 初始化两个目录的数据结构
        all_results = self._load_existing_results(self.results_dir_all)
        if 'results' not in all_results:
            all_results['results'] = {}
        for mode in self.modes:
            if mode not in all_results['results']:
                all_results['results'][mode] = {
                    'mases': [], 'maes': [], 'rmses': [], 'smapes': [], 'owas': [], 'window_mases': {}
                }

        if tasks:
            self._log(f"\n📊 待执行任务: {len(tasks)} 个")

            with concurrent.futures.ThreadPoolExecutor(max_workers=self.test_workers) as executor:
                futures = {}
                for task in tasks:
                    future = executor.submit(self.process_window, task['mode'], task['row'])
                    futures[future] = task

                pbar = tqdm(total=len(futures), desc="执行策略", unit="个", ncols=100)

                for future in concurrent.futures.as_completed(futures):
                    task = futures[future]
                    mode = task['mode']
                    window_id = task['row'].get('window_id', 'unknown')

                    try:
                        result = future.result(timeout=WINDOW_TIMEOUT)
                        if result is not None:
                            # 更新内存数据
                            results = all_results['results'][mode]
                            results['mases'].append(result['mase'])
                            results['maes'].append(result['mae'])
                            results['rmses'].append(result['rmse'])
                            results['smapes'].append(result['smape'])
                            results['owas'].append(result['owa'])
                            results['window_mases'][result['window_id']] = result['mase']

                            # 增量保存到两个目录（只保存当前模式，但保留已有全部）
                            self._save_results({mode: results})
                        else:
                            pbar.set_postfix({'状态': f'跳过窗口{window_id}'})
                    except concurrent.futures.TimeoutError:
                        pbar.set_postfix({'超时': f'窗口{window_id}'})
                    except Exception as e:
                        pbar.set_postfix({'异常': f'{type(e).__name__}'})

                    pbar.update(1)
                    total_executed = self._count_executed()
                    pbar.set_postfix({'已完成': total_executed})

                pbar.close()

            self._log(f"\n✅ 执行完成！共执行 {self._count_executed()} 个窗口")
        else:
            self._log("✅ 所有窗口已执行完成（无新任务）")

        # ★★★ 从 predictions 恢复/刷新数据 ★★★
        # 如果 half_mode 是 'all'，强制刷新所有模式的数据（覆盖旧数据）
        force_refresh = (self.half_mode == 'all')
        self._restore_missing_modes_from_predictions(force_refresh_all=force_refresh)

        # 生成两份报告
        self._generate_final_reports(self.results_dir_all, "全部模式" if force_refresh else "部分模式")
        self._generate_final_reports(self.results_dir_no_no_rule, "剔除 no_rule" if force_refresh else "部分模式")

        self._log(f"📁 报告已保存至: {self.results_dir_all} 和 {self.results_dir_no_no_rule}")

    def _generate_final_reports(self, results_dir: str, title_suffix: str):
        """生成单份报告"""
        self._log(f"\n📊 生成报告: {title_suffix}")

        all_results = self._load_existing_results(results_dir)
        if not all_results or 'results' not in all_results:
            self._log(f"⚠️ 无数据生成报告（{title_suffix}）")
            return

        results_data = all_results['results']

        # 提取汇总数据（只包含有数据的模式）
        rows = []
        for mode in self.modes:
            # 如果目录是剔除 no_rule，且模式为 no_rule，跳过
            if title_suffix == "剔除 no_rule" and mode == 'no_rule':
                continue
            data = results_data.get(mode, {})
            mases = data.get('mases', [])
            if mases:
                rows.append({
                    'mode': mode,
                    'avg_mase': np.mean(mases),
                    'avg_mae': np.mean(data.get('maes', [0])),
                    'avg_rmse': np.mean(data.get('rmses', [0])),
                    'avg_smape': np.mean(data.get('smapes', [0])),
                    'avg_owa': np.mean(data.get('owas', [0])),
                    'window_count': len(mases),
                    'window_mases': data.get('window_mases', {})
                })

        if not rows:
            self._log(f"⚠️ 无有效数据（{title_suffix}）")
            return

        # 生成文本报告
        self._generate_report_text(rows, results_dir, title_suffix)
        self._plot_bar_chart(rows, results_dir, title_suffix)
        self._plot_line_chart(rows, results_dir, title_suffix)

    def _generate_report_text(self, rows: List[Dict], output_dir: str, title_suffix: str):
        # 先找出 no_rule 的 MASE 作为基准
        no_rule_mase = None
        for r in rows:
            if r['mode'] == 'no_rule':
                no_rule_mase = r['avg_mase']
                break

        report_lines = []
        report_lines.append("=" * 120)
        report_lines.append(f"📊 语义匹配 vs RL 参数消融实验报告（半窗口版）")
        if title_suffix != "全部模式":
            report_lines.append(f"   ({title_suffix})")
        report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"运行目录: {self.run_dir_out}")
        # 根据实际窗口数显示
        total_windows = len(set().union(*[set(r.get('window_mases', {}).keys()) for r in rows if r.get('window_mases')]))
        if total_windows > 0:
            report_lines.append(f"测试集窗口数: {total_windows} 个")
        else:
            report_lines.append(f"测试集: 未知")
        report_lines.append("=" * 120)
        report_lines.append("")
        report_lines.append("模式说明:")
        for r in rows:
            report_lines.append(f"  - {r['mode']}")
        report_lines.append("")
        report_lines.append("★ 预测值已保存到 predictions/ 目录，每个窗口的预测值为 .npy 格式")
        report_lines.append("")
        # 表头
        report_lines.append(f"{'模式':<28} | {'窗口数':<8} | {'MASE':<16} | {'MAE':<12} | {'RMSE':<12} | {'SMAPE':<12} | {'OWA':<12}")
        report_lines.append("-" * 160)

        for r in rows:
            mase = r['avg_mase']
            if mase == float('inf') or math.isnan(mase):
                continue
            # 计算变化百分比
            if no_rule_mase is not None and no_rule_mase > 0 and r['mode'] != 'no_rule':
                pct = (mase - no_rule_mase) / no_rule_mase * 100
                if pct >= 0:
                    change_str = f"(+{pct:.1f}%)"
                else:
                    change_str = f"({pct:.1f}%)"
            elif r['mode'] == 'no_rule':
                change_str = "(基准)"
            else:
                change_str = "(N/A)"

            # 拼接 MASE 列：数值 + 变化标注
            mase_display = f"{mase:.6f} {change_str}"

            report_lines.append(
                f"{r['mode']:<28} | {r['window_count']:<8} | {mase_display:<16} | {r['avg_mae']:<12.6f} | "
                f"{r['avg_rmse']:<12.6f} | {r['avg_smape']:<12.6f} | {r['avg_owa']:<12.6f}"
            )

        report_lines.append("-" * 160)

        report_path = os.path.join(output_dir, "comparison_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        self._log(f"📄 对比报告已保存: {report_path}")

    def _plot_bar_chart(self, rows: List[Dict], output_dir: str, title_suffix: str):
        modes = [r['mode'] for r in rows]
        mases = [r['avg_mase'] for r in rows if r['avg_mase'] != float('inf')]

        if not modes or not mases:
            return

        # 自动扩展颜色
        base_colors = ['#808080', '#2E86AB', '#F5A623', '#E68A2E']
        extra_colors = ['#A23B72', '#3F7E5C', '#D4693A', '#1B998B', '#E84A5F', '#6A4C93', '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        colors = base_colors + extra_colors
        if len(colors) < len(modes):
            import matplotlib.cm as cm
            colors = [cm.tab20(i % 20) for i in range(len(modes))]
            colors = [f'#{int(c[0]*255):02x}{int(c[1]*255):02x}{int(c[2]*255):02x}' for c in colors]

        fig, ax = plt.subplots(figsize=(14, 6))
        bars = ax.bar(modes, mases, color=colors[:len(modes)], alpha=0.7, edgecolor='black', linewidth=1.5)

        for bar, mase in zip(bars, mases):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{mase:.4f}', ha='center', va='bottom', fontsize=9)

        ax.set_ylabel('MASE', fontsize=12)
        ax.set_title(f'各模式平均 MASE 对比（半窗口版） - {title_suffix}', fontsize=14)
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=15, ha='right')
        plt.tight_layout()

        bar_path = os.path.join(output_dir, "comparison_bar.png")
        plt.savefig(bar_path, dpi=150, bbox_inches='tight')
        plt.close()
        self._log(f"📊 柱状图已保存: {bar_path}")

    def _plot_line_chart(self, rows: List[Dict], output_dir: str, title_suffix: str):
        mode_window_data = {}
        for r in rows:
            window_mases = r.get('window_mases', {})
            if window_mases:
                mode_window_data[r['mode']] = {int(wid): mase for wid, mase in window_mases.items()}

        if not mode_window_data:
            return

        all_wids = set()
        for wdata in mode_window_data.values():
            all_wids.update(wdata.keys())
        all_wids = sorted(all_wids)

        color_map = {
            'no_rule': '#808080',
            'semantic_top1': '#2E86AB',
            'semantic_top30_theta_max': '#F5A623',
            'semantic_top40_theta_max': '#E68A2E',
            'semantic_top50_theta_max': '#D4693A',
            'semantic_top60_theta_max': '#A23B72',
            'semantic_top70_theta_max': '#6A4C93',
            'rl_top10_semantic_best': '#3F7E5C',
            'rl_top20_semantic_best': '#1B998B',
            'rl_top30_semantic_best': '#E84A5F',
            'rl_top40_semantic_best': '#FF6B6B',
            'rl_top50_semantic_best': '#4ECDC4',
            'rl_top60_semantic_best': '#45B7D1',
            'rl_top3_semantic_best': '#96CEB4',
        }
        display_names = {
            'no_rule': 'no_rule',
            'semantic_top1': 'semantic_top1',
            'semantic_top30_theta_max': 'top30% θ_max',
            'semantic_top40_theta_max': 'top40% θ_max',
            'semantic_top50_theta_max': 'top50% θ_max',
            'semantic_top60_theta_max': 'top60% θ_max',
            'semantic_top70_theta_max': 'top70% θ_max',
            'rl_top10_semantic_best': 'θ Top10% + sem',
            'rl_top20_semantic_best': 'θ Top20% + sem',
            'rl_top30_semantic_best': 'θ Top30% + sem',
            'rl_top40_semantic_best': 'θ Top40% + sem',
            'rl_top50_semantic_best': 'θ Top50% + sem',
            'rl_top60_semantic_best': 'θ Top60% + sem',
            'rl_top3_semantic_best': 'θ Top3 + sem (test)',
        }

        fig, ax = plt.subplots(figsize=(14, 7))
        for mode, wdata in mode_window_data.items():
            sorted_items = sorted(wdata.items())
            wids = [w[0] for w in sorted_items]
            mases = [w[1] for w in sorted_items]
            color = color_map.get(mode, '#000000')
            label = display_names.get(mode, mode)
            ax.plot(wids, mases, marker='o', color=color, linewidth=2, markersize=4, label=label)

        ax.set_xlabel('窗口ID', fontsize=12)
        ax.set_ylabel('MASE', fontsize=12)
        ax.set_title(f'各模式窗口 MASE 对比折线图（半窗口版） - {title_suffix}', fontsize=14)
        ax.legend(loc='upper right', fontsize=10, ncol=2)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        line_path = os.path.join(output_dir, "window_comparison.png")
        plt.savefig(line_path, dpi=150, bbox_inches='tight')
        plt.close()
        self._log(f"📊 折线图已保存: {line_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="策略执行器（阶段2）")
    parser.add_argument('--resume', type=str, required=True,
                        help='原始运行目录（如 llog/cs2）')
    parser.add_argument('--round', type=int, required=True,
                        help='指定轮次（如 57）')
    parser.add_argument('--half-mode', type=str, default='first', choices=['first', 'second', 'all'],
                        help='选择测试集位置：first(前25), second(后25), all(全部)')
    parser.add_argument('--config', type=str, default=None,
                        help='配置文件路径')
    parser.add_argument('--workers', type=int, default=8,
                        help='并行线程数（默认 8）')
    parser.add_argument('--force', action='store_true',
                        help='强制重新执行所有窗口（忽略缓存）')

    args = parser.parse_args()

    if os.path.exists(args.resume):
        run_dir = args.resume
    else:
        run_dir = os.path.join("llog", args.resume)

    if not os.path.exists(run_dir):
        print(f"❌ 目录不存在: {run_dir}")
        return

    executor = StrategyExecutor(
        run_dir=run_dir,
        round_num=args.round,
        half_mode=args.half_mode,
        config_path=args.config,
        workers=args.workers,
        force=args.force
    )
    executor.execute_all()


if __name__ == '__main__':
    main()