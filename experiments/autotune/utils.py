# experiments/autotune/utils.py
"""
工具函数 - v5 多轮训练版
★ 多指标计算（MASE, RMSE, MAE, SMAPE, OWA）
★ 对比报告生成
★ 运行时文件夹
"""

import os
import sys
import json
import yaml
import time
import shutil
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def load_config(config_path: str = None) -> Dict:
    """加载YAML配置"""
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def format_weight_for_display(value: float, precision: int = 6) -> str:
    """显示权重，去除尾随零"""
    if isinstance(value, float):
        s = f"{value:.{precision}f}".rstrip('0').rstrip('.')
        if s == '' or s == '-0':
            return "0.0"
        return s
    return str(value)


def create_run_folder(base_dir: str = "llog") -> str:
    """创建本次运行的专属文件夹"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def compute_mase(pred: np.ndarray, actual: np.ndarray, scale: float = 1.0) -> float:
    if len(pred) == 0:
        return float('nan')
    errors = np.abs(pred - actual)
    mae = np.mean(errors)
    result = mae / scale if scale > 0 else float('nan')
    return round(float(result), 6)


def compute_mae(pred: np.ndarray, actual: np.ndarray) -> float:
    if len(pred) == 0:
        return float('nan')
    return round(float(np.mean(np.abs(pred - actual))), 6)


def compute_rmse(pred: np.ndarray, actual: np.ndarray) -> float:
    if len(pred) == 0:
        return float('nan')
    errors = pred - actual
    return round(float(np.sqrt(np.mean(errors ** 2))), 6)


def compute_smape(pred: np.ndarray, actual: np.ndarray) -> float:
    if len(pred) == 0:
        return float('nan')
    errors = pred - actual
    denominator = np.abs(pred) + np.abs(actual) + 1e-8
    return round(float(np.mean(2.0 * np.abs(errors) / denominator) * 100), 6)


def compute_owa(pred: np.ndarray, actual: np.ndarray) -> float:
    if len(pred) == 0:
        return float('nan')
    rmse = compute_rmse(pred, actual)
    mae = np.mean(np.abs(pred - actual))
    return round(float((rmse + mae) / 2.0), 6)


def compute_all_metrics(pred: np.ndarray, actual: np.ndarray, mase_scale: float = 1.0) -> Dict:
    """计算所有评估指标"""
    if len(pred) == 0 or len(actual) == 0:
        return {}
    return {
        'mase': compute_mase(pred, actual, mase_scale),
        'mae': compute_mae(pred, actual),
        'rmse': compute_rmse(pred, actual),
        'smape': compute_smape(pred, actual),
        'owa': compute_owa(pred, actual),
    }


def compute_improvement(baseline: Dict, target: Dict) -> Dict:
    """计算改进百分比"""
    improvement = {}
    for key in baseline:
        if key in target and baseline[key] != 0:
            improvement[key] = (baseline[key] - target[key]) / abs(baseline[key]) * 100
        else:
            improvement[key] = 0.0
    return improvement


def generate_comparison_report(round_results: Dict, output_dir: str) -> str:
    """
    生成多轮对比报告

    Args:
        round_results: {
            'no_rule': {'mase': 0.9074, 'mae': ..., 'rmse': ..., 'smape': ..., 'owa': ...},
            'round_1': {'mase': 0.7737, ...},
            'round_2': {'mase': 0.7500, ...},
            ...
        }
        output_dir: 输出目录

    Returns:
        report_path: 报告文件路径
    """
    if not round_results:
        return ""

    # 提取所有模式
    modes = list(round_results.keys())
    metrics = ['mase', 'mae', 'rmse', 'smape', 'owa']
    metric_names = {
        'mase': 'MASE',
        'mae': 'MAE',
        'rmse': 'RMSE',
        'smape': 'SMAPE (%)',
        'owa': 'OWA'
    }

    # 构建表格
    lines = []
    lines.append("=" * 100)
    lines.append("📊 多轮策略对比报告")
    lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 100)
    lines.append("")

    # 表头
    header = f"│ {'模式':<15} │"
    for m in metrics:
        header += f" {metric_names[m]:<12} │"
    header += f" {'改善(相对无策略)':<18} │"
    lines.append("─" * (20 + 15 * len(metrics) + 22))
    lines.append(header)
    lines.append("─" * (20 + 15 * len(metrics) + 22))

    # 数据行
    baseline = round_results.get('no_rule', {})
    for mode in modes:
        if mode == 'no_rule':
            continue
        data = round_results.get(mode, {})
        row = f"│ {mode:<15} │"
        for m in metrics:
            val = data.get(m, float('nan'))
            row += f" {val:<12.6f} │"
        # 计算平均改善
        if baseline:
            improvements = []
            for m in metrics:
                if baseline.get(m, 0) != 0:
                    imp = (baseline[m] - data.get(m, baseline[m])) / abs(baseline[m]) * 100
                    improvements.append(imp)
            avg_imp = np.mean(improvements) if improvements else 0
            row += f" {avg_imp:>+8.2f}%      │"
        else:
            row += f" {'N/A':<18} │"
        lines.append(row)

    # 基线行
    if baseline:
        row = f"│ {'no_rule':<15} │"
        for m in metrics:
            val = baseline.get(m, float('nan'))
            row += f" {val:<12.6f} │"
        row += f" {'—':<18} │"
        lines.append(row)

    lines.append("─" * (20 + 15 * len(metrics) + 22))
    lines.append("")

    # 最佳轮次
    best_mode = None
    best_mase = float('inf')
    for mode in modes:
        if mode == 'no_rule':
            continue
        data = round_results.get(mode, {})
        mase = data.get('mase', float('inf'))
        if mase < best_mase:
            best_mase = mase
            best_mode = mode

    if best_mode and baseline:
        improvement = (baseline.get('mase', 1) - best_mase) / abs(baseline.get('mase', 1)) * 100
        lines.append(f"🏆 最佳策略: {best_mode}")
        lines.append(f"   MASE: {best_mase:.6f} (改善 {improvement:+.2f}%)")
        lines.append("")

    lines.append("=" * 100)
    lines.append("✅ 对比报告生成完成")

    report_content = "\n".join(lines)

    # 保存报告
    report_path = os.path.join(output_dir, "comparison_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)

    return report_path


class ProgressLogger:
    """进度日志器 - 支持tqdm集成"""

    def __init__(self, log_dir: str = None, verbose: bool = True, run_folder: bool = True):
        self.verbose = verbose
        self._start_time = None
        self._log_file = None
        self._step_count = 0
        self._total_steps = 0

        if log_dir:
            self.log_dir = log_dir
        else:
            base_dir = "llog"
            if run_folder:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.log_dir = os.path.join(base_dir, f"run_{timestamp}")
            else:
                self.log_dir = base_dir
            os.makedirs(self.log_dir, exist_ok=True)

    def start_log(self, name: str = "autotune"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._log_file = os.path.join(self.log_dir, f"{name}_{timestamp}.log")

    def log(self, message: str, level: str = "INFO"):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full_msg = f"[{timestamp}] [{level}] {message}"
        if self.verbose:
            print(full_msg)
        if self._log_file:
            with open(self._log_file, "a", encoding="utf-8") as f:
                f.write(full_msg + "\n")

    def start(self, total_steps: int, message: str = ""):
        self._start_time = time.time()
        self._total_steps = total_steps
        self._step_count = 0
        self.log("=" * 70)
        if message:
            self.log(f"🚀 {message}")
        self.log(f"📊 总步数: {total_steps}")
        self.log("=" * 70)

    def step(self, message: str = "", sub_message: str = ""):
        self._step_count += 1
        elapsed = time.time() - self._start_time if self._start_time else 0
        remaining = (elapsed / self._step_count) * (
                    self._total_steps - self._step_count) if self._step_count > 0 and self._total_steps > 0 else 0
        progress = f"[{self._step_count}/{self._total_steps}]"
        time_info = f"⏱️ 已用: {elapsed:.1f}s | 预计剩余: {remaining:.1f}s" if self._total_steps > 0 else ""
        log_msg = f"{progress} {message}"
        if sub_message:
            log_msg += f" → {sub_message}"
        if time_info:
            log_msg += f"  {time_info}"
        self.log(log_msg)

    def finish(self, message: str = ""):
        if self._start_time:
            elapsed = time.time() - self._start_time
            self.log(f"✅ 完成! 总耗时: {elapsed:.1f}s")
        if message:
            self.log(message)
        self.log("=" * 70)

    def get_log_file(self) -> str:
        return self._log_file

    def get_log_dir(self) -> str:
        return self.log_dir


class MemoryCache:
    def __init__(self, cache_dir: str = "storage/autotune_results/cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def get(self, key: str) -> Optional[Any]:
        cache_file = os.path.join(self.cache_dir, f"{key}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return None
        return None

    def set(self, key: str, value: Any):
        cache_file = os.path.join(self.cache_dir, f"{key}.json")
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(value, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ 缓存写入失败: {e}")

    def exists(self, key: str) -> bool:
        return os.path.exists(os.path.join(self.cache_dir, f"{key}.json"))


def detect_period(series: np.ndarray, freq: str = None) -> int:
    from src.skills.data_profiler import DataProfiler
    return DataProfiler._auto_period(series, freq=freq)


def extract_features(series: np.ndarray, local_window_sizes: List[int] = None) -> Dict[str, float]:
    from src.skills.data_profiler import DataProfiler

    n = len(series)
    profile = DataProfiler.profile_selected(series, [
        'trend_strength', 'seasonal_strength', 'adf_pvalue',
        'period', 'data_length', 'skewness', 'cv'
    ])

    features = {
        'trend_strength': round(profile.get('trend_strength', 0.0), 6),
        'seasonal_strength': round(profile.get('seasonal_strength', 0.0), 6),
        'adf_pvalue': round(profile.get('adf_pvalue', 0.5), 6),
        'period': profile.get('period', 12),
        'data_length': n,
        'skewness': round(profile.get('skewness', 0.0), 6),
        'cv': round(profile.get('cv', 0.0), 6),
    }

    if local_window_sizes is None:
        local_window_sizes = [7, 30]

    for local_window in local_window_sizes:
        local_n = min(local_window, n)
        if local_n <= 0:
            continue
        local_series = series[-local_n:]

        if len(local_series) >= 3:
            from scipy import stats
            x = np.arange(len(local_series))
            slope, _, _, _, _ = stats.linregress(x, local_series)
            slope = round(float(slope), 6)
        else:
            slope = 0.0

        local_std = np.std(local_series) if len(local_series) > 1 else 0.0
        global_std = np.std(series) if len(series) > 1 else 1.0
        local_std_ratio = round(local_std / (global_std + 1e-8), 6)

        if len(local_series) >= 2 and local_series[0] != 0:
            local_change_rate = round((local_series[-1] - local_series[0]) / abs(local_series[0] + 1e-8), 6)
        else:
            local_change_rate = 0.0

        local_mean = np.mean(local_series) if len(local_series) > 0 else 0.0
        global_mean = np.mean(series) if len(series) > 0 else 1.0
        local_mean_ratio = round(local_mean / (global_mean + 1e-8), 6)

        features[f'local_slope_{local_window}'] = slope
        features[f'local_std_ratio_{local_window}'] = local_std_ratio
        features[f'local_change_rate_{local_window}'] = local_change_rate
        features[f'local_mean_ratio_{local_window}'] = local_mean_ratio

    return features


def serialize_trajectory(trajectory):
    import json
    return json.dumps(trajectory, ensure_ascii=False)


def deserialize_trajectory(traj_str):
    import json
    return json.loads(traj_str)


def save_window_data(train: np.ndarray, test: np.ndarray, period: int,
                     mase_scale: float, features: Dict, window_id: int,
                     dataset_name: str, horizon: int) -> str:
    import pickle
    data_dir = "storage/autotune_results/window_data"
    os.makedirs(data_dir, exist_ok=True)

    file_path = os.path.join(data_dir, f"{dataset_name}_window_{window_id}.pkl")
    data = {
        'train': train,
        'test': test,
        'period': period,
        'mase_scale': mase_scale,
        'features': features,
        'window_id': window_id,
        'horizon': horizon
    }
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
    return file_path


def load_window_data(file_path: str) -> Dict:
    import pickle
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def compute_strategy_similarity(weights_a: dict, weights_b: dict) -> float:
    if not weights_a or not weights_b:
        return 0.0

    all_skills = set(weights_a.keys()) | set(weights_b.keys())
    if not all_skills:
        return 0.0

    total_a = sum(weights_a.values())
    total_b = sum(weights_b.values())
    if total_a == 0 or total_b == 0:
        return 0.0

    vec_a = np.array([weights_a.get(s, 0) / total_a for s in all_skills])
    vec_b = np.array([weights_b.get(s, 0) / total_b for s in all_skills])

    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0

    cos_sim = np.dot(vec_a, vec_b) / (norm_a * norm_b)
    return max(0.0, min(1.0, round(float(cos_sim), 4)))