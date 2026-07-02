# experiments/autotune/trouble_pool_manager.py
"""
困难窗口池管理器
封装池子的 CRUD 操作，支持动态管理和已解决窗口清理

职责：
- 加载/保存 trouble_windows.json
- 添加/移除/更新窗口
- 检查窗口是否在池中
- 清理已解决窗口（MASE <= 1.0）
- 获取未解决窗口列表
"""

import os
import json
import time
from typing import Dict, List, Optional, Any, Set
import numpy as np


class TroublePoolManager:
    """
    困难窗口池管理器

    核心原则：
    1. 池子是动态的——已解决的窗口会被自动移除
    2. 已解决判定：MASE <= 1.0
    3. 池中窗口只更新 MASE，不重复添加
    """

    def __init__(self, llog_dir: str, logger=None, mase_threshold: float = 1.0):
        """
        Args:
            llog_dir: 运行目录（如 llog/run_xxx/）
            logger: 日志器（可选）
            mase_threshold: 已解决阈值（默认 1.0）
        """
        self.llog_dir = llog_dir
        self.logger = logger
        self.mase_threshold = mase_threshold
        self.pool_file = os.path.join(llog_dir, 'trouble_windows.json')
        self._pool: List[Dict] = []
        self._loaded = False

    def _log(self, msg: str, level: str = "INFO"):
        if self.logger is not None:
            self.logger.log(msg)
        else:
            print(f"[TroublePoolManager] {msg}")

    def load(self) -> List[Dict]:
        """加载困难窗口池，自动清理已解决窗口"""
        self._pool = []
        if os.path.exists(self.pool_file):
            try:
                with open(self.pool_file, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                # ★★★ 自动清理已解决窗口（MASE <= threshold） ★★★
                for item in loaded:
                    mase = item.get('mase', 1.0)
                    if mase > self.mase_threshold:
                        self._pool.append(item)
                if len(loaded) != len(self._pool):
                    self._log(f"   🧹 加载池时自动清理 {len(loaded) - len(self._pool)} 个已解决窗口")
            except Exception as e:
                self._log(f"   ⚠️ 加载困难池失败: {e}")
                self._pool = []
        self._loaded = True
        return self._pool

    def save(self):
        """保存池子到文件"""
        try:
            seen = set()
            unique_pool = []
            for item in self._pool:
                wid = item.get('window_id')
                if wid not in seen:
                    seen.add(wid)
                    cleaned_item = {}
                    for k, v in item.items():
                        if isinstance(v, np.integer):
                            cleaned_item[k] = int(v)
                        elif isinstance(v, np.floating):
                            cleaned_item[k] = float(v)
                        elif isinstance(v, np.ndarray):
                            cleaned_item[k] = v.tolist()
                        else:
                            cleaned_item[k] = v
                    unique_pool.append(cleaned_item)
            with open(self.pool_file, 'w', encoding='utf-8') as f:
                json.dump(unique_pool, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self._log(f"   ⚠️ 保存困难池失败: {e}")

    def get_pool(self) -> List[Dict]:
        """获取池中所有窗口"""
        if not self._loaded:
            self.load()
        return self._pool

    def get_pool_ids(self) -> Set[int]:
        """获取池中所有窗口 ID"""
        return {w.get('window_id') for w in self.get_pool() if w.get('window_id') is not None}

    def is_in_pool(self, window_id: int) -> bool:
        """检查窗口是否在池中"""
        return window_id in self.get_pool_ids()

    def add_window(self, window_id: int, mase: float, origin: int = 0,
                   window_size: int = 600, window_data_path: str = '',
                   best_strategy_name: str = 'unknown') -> bool:
        """
        添加窗口到池中（若 MASE > threshold 且不在池中）

        Returns:
            bool: 是否成功添加
        """
        if mase <= self.mase_threshold:
            # 已解决，不加入
            self._log(f"      ℹ️ 窗口 {window_id} 已解决 (MASE={mase:.4f} <= {self.mase_threshold})，不加入池")
            return False

        if self.is_in_pool(window_id):
            # 已在池中，更新 MASE
            self._log(f"      ℹ️ 窗口 {window_id} 已在池中，更新 MASE={mase:.4f}")
            self.update_mase(window_id, mase)
            return False

        # 新窗口：加入池
        if not self._loaded:
            self.load()
        self._pool.append({
            'window_id': window_id,
            'origin': origin,
            'window_size': window_size,
            'mase': mase,
            'window_data_path': window_data_path,
            'best_strategy_name': best_strategy_name,
            'collected_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'last_updated': time.strftime('%Y-%m-%d %H:%M:%S')
        })
        self.save()
        self._log(f"      ✅ 新困难窗口 {window_id} 已加入困难池 (MASE={mase:.4f} > {self.mase_threshold})")
        return True

    def remove_window(self, window_id: int) -> bool:
        """从池中移除窗口"""
        initial_len = len(self._pool)
        self._pool = [w for w in self._pool if w.get('window_id') != window_id]
        if len(self._pool) < initial_len:
            self.save()
            self._log(f"      🗑️ 窗口 {window_id} 已从困难池移除")
            return True
        return False

    def update_mase(self, window_id: int, mase: float) -> bool:
        """更新池中窗口的 MASE"""
        for item in self._pool:
            if item.get('window_id') == window_id:
                item['mase'] = mase
                item['last_updated'] = time.strftime('%Y-%m-%d %H:%M:%S')
                self.save()
                return True
        return False

    def clean_solved(self) -> int:
        """
        清理池中已解决窗口（MASE <= threshold）
        返回清理数量
        """
        if not self._loaded:
            self.load()
        initial_len = len(self._pool)
        self._pool = [w for w in self._pool if w.get('mase', 1.0) > self.mase_threshold]
        cleaned = initial_len - len(self._pool)
        if cleaned > 0:
            self.save()
            self._log(f"   🧹 清理了 {cleaned} 个已解决窗口（MASE <= {self.mase_threshold}）")
        return cleaned

    def get_unsolved_windows(self) -> List[Dict]:
        """获取所有未解决窗口（MASE > threshold）"""
        return [w for w in self.get_pool() if w.get('mase', 1.0) > self.mase_threshold]

    def get_unsolved_ids(self) -> Set[int]:
        """获取所有未解决窗口 ID"""
        return {w.get('window_id') for w in self.get_unsolved_windows() if w.get('window_id') is not None}

    def clear(self):
        """清空池子（谨慎使用）"""
        self._pool = []
        if os.path.exists(self.pool_file):
            os.remove(self.pool_file)
        self._log("   🧹 困难池已清空")

    def get_stats(self) -> Dict:
        """获取池子统计信息"""
        pool = self.get_pool()
        unsolved = self.get_unsolved_windows()
        if not pool:
            return {
                'total': 0,
                'unsolved': 0,
                'solved': 0,
                'avg_mase': 0.0,
                'min_mase': 0.0,
                'max_mase': 0.0
            }
        mases = [w.get('mase', 1.0) for w in pool]
        return {
            'total': len(pool),
            'unsolved': len(unsolved),
            'solved': len(pool) - len(unsolved),
            'avg_mase': sum(mases) / len(mases),
            'min_mase': min(mases),
            'max_mase': max(mases)
        }