# experiments/autotune/cache_manager.py
import os
import json
import hashlib
import pickle
from typing import Dict, List, Optional, Any
from datetime import datetime


class StrategyCache:
    """策略缓存管理器"""

    def __init__(self, cache_dir: str = "storage/autotune_results/strategy_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _get_cache_key(self, dataset_name: str, window_size: int, step: int,
                       horizon: int, fixed_params: Dict) -> str:
        """生成缓存键"""
        param_str = f"{dataset_name}_{window_size}_{step}_{horizon}_{json.dumps(fixed_params, sort_keys=True)}"
        return hashlib.md5(param_str.encode()).hexdigest()

    def get(self, dataset_name: str, window_size: int, step: int,
            horizon: int, fixed_params: Dict) -> Optional[Dict]:
        """获取缓存"""
        key = self._get_cache_key(dataset_name, window_size, step, horizon, fixed_params)
        cache_file = os.path.join(self.cache_dir, f"{key}.pkl")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except:
                return None
        return None

    def save(self, dataset_name: str, window_size: int, step: int,
             horizon: int, fixed_params: Dict, data: Dict):
        """保存缓存"""
        key = self._get_cache_key(dataset_name, window_size, step, horizon, fixed_params)
        cache_file = os.path.join(self.cache_dir, f"{key}.pkl")
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"⚠️ 缓存保存失败: {e}")

    def clear(self, dataset_name: str = None):
        """清除缓存"""
        if dataset_name:
            # 清除特定数据集的缓存（简单实现）
            for f in os.listdir(self.cache_dir):
                if dataset_name in f:
                    os.remove(os.path.join(self.cache_dir, f))
        else:
            import shutil
            shutil.rmtree(self.cache_dir)
            os.makedirs(self.cache_dir, exist_ok=True)