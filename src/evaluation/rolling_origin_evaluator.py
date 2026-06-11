import numpy as np
import pandas as pd
import time
from typing import Dict
from tqdm import tqdm
from src.tasks.instance import TaskInstance
from src.dataset.registry import DatasetRegistry
from src.dataset.loader import load_dataset
from src.agents.base import BaseAgent
import uuid
from datetime import datetime, timedelta

class RollingOriginEvaluator:
    def __init__(self, agent: BaseAgent, min_train_size: int = 50, horizon: int = 1,
                 step_size: int = 1, data_ratio: float = 1.0, max_prototypes: int = 50):
        self.agent = agent
        self.min_train_size = min_train_size
        self.horizon = horizon
        self.step_size = step_size
        self.data_ratio = max(0.1, min(1.0, data_ratio))
        self.max_prototypes = max_prototypes

    def evaluate(self, dataset_id: str) -> Dict:
        registry = DatasetRegistry()
        ds_config = registry.get(dataset_id)
        if not ds_config:
            raise ValueError(f"Dataset {dataset_id} not found")

        df = load_dataset(ds_config)
        target_col = ds_config['target_column']
        series = df[target_col].values
        dates = df.index                           # DatetimeIndex

        freq = ds_config.get('frequency', 'daily')

        if self.data_ratio < 1.0:
            n_use = int(len(series) * self.data_ratio)
            series = series[:n_use]
            dates = dates[:n_use]

        n_total = len(series)
        if n_total <= self.min_train_size:
            raise ValueError("数据集太小，请减小 min_train_size 或增大 data_ratio")

        total_steps = (n_total - self.min_train_size - self.horizon + 1) // self.step_size
        if total_steps <= 0:
            total_steps = 1

        first_test_idx = self.min_train_size
        train_for_scale = series[:first_test_idx]
        naive_errors = np.abs(np.diff(train_for_scale))
        mase_scale = np.mean(naive_errors) if len(naive_errors) > 0 else 1.0
        rmsse_scale = np.sqrt(np.mean(naive_errors ** 2)) if len(naive_errors) > 0 else 1.0

        predictions = []
        actuals = []
        start_time = time.time()

        for idx in tqdm(range(0, total_steps * self.step_size, self.step_size),
                        desc=f"评估 {dataset_id}", unit="step", total=total_steps):
            i = self.min_train_size + idx
            history = series[:i].tolist()
            target_idx = i + self.horizon - 1
            actual = series[target_idx]
            target_date = dates[target_idx]
            # 历史对应的日期列表（转为字符串）
            history_dates = [d.strftime('%Y-%m-%d') for d in dates[:i]]

            task = TaskInstance(
                id=f"ro_{dataset_id}_{uuid.uuid4().hex[:8]}",
                dataset_id=dataset_id,
                template_id="rolling_origin",
                question=f"基于历史数据预测 {target_date.strftime('%Y-%m-%d')} 的 {target_col}",
                question_type="numerical",
                history=history,
                horizon=self.horizon,
                frequency=freq,
                prediction_target={'date': target_date.strftime('%Y-%m-%d')},
                resolution_date=datetime.now() + timedelta(days=1),
                difficulty_level=1,
                ground_truth=None,                    # 尚未知道真实值，可后续填充
                ground_truth_extractor="",
                dates=history_dates,                  # 传递历史日期
                target_date=target_date.strftime('%Y-%m-%d')  # 目标日期
            )

            try:
                pred = self.agent.predict(task)
                if pred is not None:
                    predictions.append(pred)
                    actuals.append(actual)
                    self._record_prototype(task, pred, actual)
            except Exception as e:
                print(f"⚠️ 预测失败 {target_date}: {e}")

        elapsed = time.time() - start_time
        pred = np.array(predictions)
        true = np.array(actuals)

        metrics = self._compute_all_metrics(pred, true, mase_scale, rmsse_scale)

        return {
            'dataset_id': dataset_id,
            'min_train_size': self.min_train_size,
            'n_predictions': len(predictions),
            'predictions': predictions,
            'actuals': actuals,
            **metrics,
            'elapsed_seconds': round(elapsed, 1)
        }

    def _record_prototype(self, task, prediction, actual):
        try:
            skill_name = getattr(self.agent, '_last_used_skill', None)
            if not skill_name:
                return
            skill = self.agent.skills.get(skill_name)
            if not skill:
                return
            from src.skills.data_profiler import DataProfiler
            profile = DataProfiler.profile(np.array(task.history), dates=task.dates)  # 传入日期

            snap = profile.get('snapshot_vector', [])
            if not snap:
                return
            skill.prototypes.append({
                'vector': snap,
                'meta': {'error': abs(prediction - actual), 'date': str(task.prediction_target.get('date', ''))}
            })
            if len(skill.prototypes) > self.max_prototypes:
                skill.prototypes = skill.prototypes[-self.max_prototypes:]
        except Exception as e:
            pass

    def _compute_all_metrics(self, pred: np.ndarray, true: np.ndarray,
                             mase_scale: float, rmsse_scale: float) -> Dict:
        if len(pred) == 0:
            return {'rmse': np.nan, 'mae': np.nan, 'smape': np.nan,
                    'mase': np.nan, 'rmsse': np.nan, 'owa': np.nan, 'mdape': np.nan}

        error = pred - true
        rmse = np.sqrt(np.mean(error ** 2))
        mae = np.mean(np.abs(error))
        denominator = np.abs(pred) + np.abs(true) + 1e-8
        smape = np.mean(2.0 * np.abs(error) / denominator) * 100
        mase = mae / mase_scale if mase_scale != 0 else np.nan
        rmsse = rmse / rmsse_scale if rmsse_scale != 0 else np.nan
        owa = (mase + rmsse) / 2.0
        mask = np.abs(true) > 1e-8
        mdape = np.median(np.abs((pred[mask] - true[mask]) / true[mask]) * 100) if np.sum(mask) > 0 else np.nan

        return {
            'rmse': round(rmse, 4),
            'mae': round(mae, 4),
            'smape': round(smape, 4),
            'mase': round(mase, 4),
            'rmsse': round(rmsse, 4),
            'owa': round(owa, 4),
            'mdape': round(mdape, 4)
        }

    @staticmethod
    def print_report(result: Dict):
        print(f"\n{'='*50}")
        print(f"数据集: {result['dataset_id']}")
        print(f"最小训练窗口: {result['min_train_size']}")
        print(f"有效预测次数: {result['n_predictions']}")
        print(f"{'指标':<10} {'数值'}")
        print(f"{'RMSE':<10} {result['rmse']}")
        print(f"{'MAE':<10} {result['mae']}")
        print(f"{'sMAPE':<10} {result['smape']}%")
        print(f"{'MASE':<10} {result['mase']}")
        print(f"{'RMSSE':<10} {result['rmsse']}")
        print(f"{'OWA':<10} {result['owa']}")
        print(f"{'MdAPE':<10} {result['mdape']}%")
        print(f"{'耗时':<10} {result['elapsed_seconds']} 秒")
        print(f"{'='*50}\n")