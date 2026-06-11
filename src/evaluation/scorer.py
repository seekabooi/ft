import os
import json
import numpy as np
import pandas as pd
import glob
from datetime import datetime
from src.config import STORAGE_DIR, SCORES_DIR
from src.evaluation.metrics import normalized_score
from src.tasks.storage import load_tasks
from src.evaluation.collector import AnswerCollector
from src.dataset.loader import load_dataset

class Scorer:
    def __init__(self):
        self.collector = AnswerCollector()

    def run(self, date_str):
        tasks = load_tasks(date_str)
        predictions_dir = os.path.join(STORAGE_DIR, "predictions", date_str)
        if not os.path.exists(predictions_dir):
            return pd.DataFrame()
        results = []
        for agent_file in os.listdir(predictions_dir):
            agent_name = agent_file.replace('.jsonl', '')
            pred_path = os.path.join(predictions_dir, agent_file)
            preds = {}
            with open(pred_path, 'r', encoding='utf-8') as f:
                for line in f:
                    record = json.loads(line)
                    preds[record['task_id']] = record['prediction']
            for task in tasks:
                if task.id not in preds:
                    continue

                # ===== 直接从数据源获取真实值 =====
                ds = self.collector.registry.get(task.dataset_id)
                df = load_dataset(ds)  # 此时 df.index 是日期
                target_col = ds['target_column']
                date_str_target = task.prediction_target.get('date')
                truth = None
                if date_str_target:
                    try:
                        target_date = pd.Timestamp(date_str_target)
                        if target_date in df.index:
                            truth = float(df.loc[target_date, target_col])
                    except:
                        pass
                # ==================================

                if truth is None:
                    continue

                # 计算历史标准差
                history = task.history
                if isinstance(history, dict):
                    history = list(history.values())
                elif not isinstance(history, list):
                    history = list(history)
                if len(history) > 1:
                    hist_std = float(np.std(history))
                    if hist_std == 0:
                        hist_std = 1.0
                else:
                    hist_std = 1.0

                score = normalized_score(preds[task.id], truth, hist_std)
                results.append({
                    'date': date_str,
                    'task_id': task.id,
                    'agent': agent_name,
                    'dataset': task.dataset_id,
                    'difficulty': task.difficulty_level,
                    'prediction': preds[task.id],
                    'ground_truth': truth,
                    'score': score
                })

        if results:
            df = pd.DataFrame(results)
            os.makedirs(SCORES_DIR, exist_ok=True)
            df.to_parquet(os.path.join(SCORES_DIR, f"{date_str}_scores.parquet"))
            self.update_leaderboard()
        return pd.DataFrame(results)

    def update_leaderboard(self):
        files = glob.glob(os.path.join(SCORES_DIR, "*.parquet"))
        if not files:
            return
        df_all = pd.concat([pd.read_parquet(f) for f in files])
        leaderboard = df_all.groupby('agent')['score'].mean().reset_index()
        leaderboard.to_csv(os.path.join(STORAGE_DIR, "leaderboard.csv"), index=False)