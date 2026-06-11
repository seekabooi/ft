import os
from dotenv import load_dotenv

load_dotenv()

ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY", "6a4a1ccfac924e95a8d7ab903325a5c1.teAGpy4lhpWvofKF")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://open.bigmodel.cn/api/paas/v4")
DATA_DIR = "data"
STORAGE_DIR = "storage"
TASKS_DIR = os.path.join(STORAGE_DIR, "tasks")
PREDICTIONS_DIR = os.path.join(STORAGE_DIR, "predictions")
GROUND_TRUTH_DIR = os.path.join(STORAGE_DIR, "ground_truth")
SCORES_DIR = os.path.join(STORAGE_DIR, "scores")
LOGS_DIR = os.path.join(STORAGE_DIR, "logs")
REGISTRY_PATH = os.path.join(DATA_DIR, "dataset_registry.yaml")