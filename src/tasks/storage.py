import os
from src.tasks.instance import TaskInstance
from src.config import TASKS_DIR

def load_tasks(date_str):
    path = os.path.join(TASKS_DIR, f"{date_str}.jsonl")
    tasks = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            tasks.append(TaskInstance.model_validate_json(line))
    return tasks