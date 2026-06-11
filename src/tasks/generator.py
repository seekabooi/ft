import random
import uuid
import pandas as pd
from datetime import datetime, timedelta
from src.tasks.instance import TaskInstance
from src.dataset.registry import DatasetRegistry
from src.dataset.loader import load_dataset
from src.dataset.splitter import split_history_future
from src.config import TASKS_DIR
import os

class TaskGenerator:
    def __init__(self):
        self.registry = DatasetRegistry()
        self.datasets = self.registry.datasets

    def generate_daily_tasks(self, date_str=None, num_tasks=10,
                             start_override=None, end_override=None,
                             failed_collector=None,
                             dataset_filter=None):
        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")
        gen_date = datetime.strptime(date_str, "%Y-%m-%d")
        tasks = []
        dataset_list = list(self.datasets.values())
        random.shuffle(dataset_list)

        attempts = 0
        max_attempts = num_tasks * 3
        failed_sources = {}

        while len(tasks) < num_tasks and attempts < max_attempts:
            attempts += 1

            if dataset_filter:
                ds = self.datasets.get(dataset_filter)
                if not ds:
                    print(f"  ⚠️ 数据集 {dataset_filter} 不存在")
                    break
            else:
                ds = random.choice(dataset_list)

            template = random.choice(ds['templates'])
            frequency = ds.get('frequency', 'daily')
            variables = self._fill_variables(template, gen_date, frequency)

            ds_config = ds.copy()
            if start_override or end_override:
                ds_config['source']['params'] = ds_config['source']['params'].copy()
                if start_override:
                    ds_config['source']['params']['start'] = start_override
                if end_override:
                    ds_config['source']['params']['end'] = end_override

            try:
                df = load_dataset(ds_config)
                history = split_history_future(df, ds['target_column'], variables, template,
                                               frequency=frequency)
            except Exception as e:
                ds_id = ds['id']
                if ds_id not in failed_sources:
                    failed_sources[ds_id] = {
                        'id': ds_id,
                        'name': ds.get('name', ds_id),
                        'error': str(e)
                    }
                continue

            horizon = variables.get('horizon', template.get('horizon', 1))
            resolution_date = self._calc_resolution_date(gen_date, horizon, frequency,
                                                         ds.get('resolution_delay_days', 1))

            task = TaskInstance(
                id=f"{ds['id']}_{date_str}_{uuid.uuid4().hex[:8]}",
                dataset_id=ds['id'],
                template_id=template['id'],
                question=template['question_template'].format(**variables),
                question_type=template['type'],
                history=history,
                horizon=horizon,
                frequency=frequency,
                prediction_target=variables,
                resolution_date=resolution_date,
                difficulty_level=template.get('difficulty', 1),
                ground_truth_extractor=template['answer_extraction']
            )
            tasks.append(task)

        if failed_collector is not None:
            failed_collector.extend(failed_sources.values())

        os.makedirs(TASKS_DIR, exist_ok=True)
        task_path = os.path.join(TASKS_DIR, f"{date_str}.jsonl")
        with open(task_path, 'w', encoding='utf-8') as f:
            for task in tasks:
                f.write(task.model_dump_json() + "\n")

        if len(tasks) < num_tasks:
            print(f"⚠️ 只成功生成 {len(tasks)}/{num_tasks} 个任务")
        return tasks

    def _fill_variables(self, template, gen_date, frequency='daily'):
        vars = {}
        if 'variables' in template:
            for v in template['variables']:
                if v == 'date':
                    if frequency == 'monthly':
                        # 下个月的第一天
                        next_month = gen_date.replace(day=1) + pd.DateOffset(months=1)
                        vars['date'] = next_month.strftime("%Y-%m-%d")
                    else:
                        # 每天：随机 1~7 天后
                        vars['date'] = (gen_date + timedelta(days=random.randint(1, 7))).strftime("%Y-%m-%d")
                elif v == 'horizon':
                    opts = template.get('horizon_options', [1, 3, 7])
                    vars['horizon'] = random.choice(opts)
        return vars

    def _calc_resolution_date(self, gen_date, horizon, frequency, delay_days):
        if frequency == 'hourly':
            delta = timedelta(hours=horizon) + timedelta(days=delay_days)
        elif frequency == 'minutely':
            delta = timedelta(minutes=horizon) + timedelta(days=delay_days)
        elif frequency == 'monthly':
            # 月频：下个月第一天 + delay_days 天
            next_month = gen_date.replace(day=1) + pd.DateOffset(months=1)
            return next_month + timedelta(days=delay_days)
        else:
            delta = timedelta(days=horizon) + timedelta(days=delay_days)
        return gen_date + delta