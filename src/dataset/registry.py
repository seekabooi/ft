import yaml
from src.config import REGISTRY_PATH

class DatasetRegistry:
    def __init__(self, path=REGISTRY_PATH):
        with open(path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        self.datasets = {ds['id']: ds for ds in config['datasets']}

    def get(self, dataset_id):
        return self.datasets.get(dataset_id)

    def list_ids(self):
        return list(self.datasets.keys())