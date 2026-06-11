import numpy as np
import pandas as pd
import time
from typing import Dict
from src.tasks.instance import TaskInstance
from src.dataset.registry import DatasetRegistry
from src.dataset.loader import load_dataset
import uuid
from datetime import datetime, timedelta

class FixedOriginEvaluator:
    def __init__(self, agent, min_train_size: int = 132, horizon: int = 12,
                 data_ratio: float = 1.0):
        self.agent = agent
        self.min_train_size = min_train_size
        self.horizon = horizon
        self.data_ratio = max(0.1, min(1.0, data_ratio))

    def _detect_period(self, series, freq):
        n = len(series)
        if n < 24:
            return 1
        if freq and freq.lower() in ('d', 'daily'):
            if n >= 365:
                return 365
            return 7
        elif freq and freq.lower() in ('m', 'monthly'):
            return 12
        try:
            from statsmodels.tsa.stattools import acf
            acf_vals = acf(series, nlags=min(n // 2, 50))
            peaks = [i for i in range(2, len(acf_vals) - 1) if
                     acf_vals[i] > acf_vals[i - 1] and acf_vals[i] > acf_vals[i + 1] and acf_vals[i] > 0.2]
            if peaks:
                return peaks[0]
        except:
            pass
        return 1

    def evaluate(self, dataset_id: str) -> Dict:
        registry = DatasetRegistry()
        ds_config = registry.get(dataset_id)
        if not ds_config:
            raise ValueError(f"Dataset {dataset_id} not found")

        df = load_dataset(ds_config)
        target_col = ds_config['target_column']
        series = df[target_col].values
        dates = df.index

        freq = ds_config.get('frequency', 'daily')

        if self.data_ratio < 1.0:
            n_use = int(len(series) * self.data_ratio)
            series = series[:n_use]
            dates = dates[:n_use]

        n_total = len(series)
        if n_total < self.min_train_size + self.horizon:
            raise ValueError(f"数据集太小（总长度{n_total}），需要至少{self.min_train_size + self.horizon}")

        train_series = series[:self.min_train_size]
        train_dates = dates[:self.min_train_size]
        test_series = series[self.min_train_size:self.min_train_size + self.horizon]
        test_dates = dates[self.min_train_size:self.min_train_size + self.horizon]

        period = self._detect_period(train_series, freq)
        if period > 1 and len(train_series) >= 2 * period:
            seasonal_errors = np.abs(train_series[period:] - train_series[:-period])
            mase_scale = np.mean(seasonal_errors) if len(seasonal_errors) > 0 else 1.0
            rmsse_scale = np.sqrt(np.mean(seasonal_errors ** 2)) if len(seasonal_errors) > 0 else 1.0
        else:
            naive_errors = np.abs(np.diff(train_series))
            mase_scale = np.mean(naive_errors) if len(naive_errors) > 0 else 1.0
            rmsse_scale = np.sqrt(np.mean(naive_errors ** 2)) if len(naive_errors) > 0 else 1.0

        task = TaskInstance(
            id=f"fo_{dataset_id}_{uuid.uuid4().hex[:8]}",
            dataset_id=dataset_id,
            template_id="fixed_origin",
            question=f"基于历史数据预测未来{self.horizon}期的{target_col}",
            question_type="numerical",
            history=train_series.tolist(),
            horizon=self.horizon,
            frequency=freq,
            prediction_target={'start_date': test_dates[0].strftime('%Y-%m-%d'),
                               'end_date': test_dates[-1].strftime('%Y-%m-%d')},
            resolution_date=datetime.now() + timedelta(days=1),
            difficulty_level=1,
            ground_truth_extractor="",
            dates=[d.strftime('%Y-%m-%d') for d in train_dates],
            target_date=test_dates[0].strftime('%Y-%m-%d')
        )

        start_time = time.time()
        try:
            predictions = self.agent.predict(task)
            if predictions is None or len(predictions) != self.horizon:
                raise ValueError("Agent 未返回正确数量的预测值")
        except Exception as e:
            print(f"❌ 预测失败: {e}")
            return {}

        elapsed = time.time() - start_time

        pred = np.array(predictions)
        true = test_series

        metrics = self._compute_all_metrics(pred, true, mase_scale, rmsse_scale)

        return {
            'dataset_id': dataset_id,
            'min_train_size': self.min_train_size,
            'horizon': self.horizon,
            'n_predictions': len(predictions),
            'predictions': predictions,
            'actuals': true.tolist(),
            **metrics,
            'elapsed_seconds': round(elapsed, 1)
        }

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
            'rmse': round(rmse, 6),
            'mae': round(mae, 6),
            'smape': round(smape, 6),
            'mase': round(mase, 6),
            'rmsse': round(rmsse, 6),
            'owa': round(owa, 6),
            'mdape': round(mdape, 6)
        }

    @staticmethod
    def print_report(result: Dict):
        # 防御性检查
        if not result or 'dataset_id' not in result:
            print("❌ 无法打印报告：结果数据不完整")
            return
        print(f"\n{'='*50}")
        print(f"数据集: {result['dataset_id']}")
        print(f"训练窗口: {result['min_train_size']}，预测步数: {result['horizon']}")
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