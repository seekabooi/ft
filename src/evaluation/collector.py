import pandas as pd
from datetime import datetime
from src.dataset.loader import load_dataset
from src.dataset.registry import DatasetRegistry


class AnswerCollector:
    def __init__(self):
        self.registry = DatasetRegistry()

    def collect_for_task(self, task):
        resolution = task.resolution_date
        if isinstance(resolution, str):
            resolution = datetime.fromisoformat(resolution)
        if datetime.now() < resolution:
            return None

        ds = self.registry.get(task.dataset_id)
        df = load_dataset(ds)
        target_col = ds['target_column']
        date_str = task.prediction_target.get('date')

        if date_str:
            if 'date' in df.columns:
                match = df[df['date'].astype(str) == date_str]
                if not match.empty:
                    return float(match[target_col].iloc[0])
            try:
                target_date = pd.Timestamp(date_str)
                if target_date in df.index:
                    return float(df.loc[target_date, target_col])
            except:
                pass
        return None