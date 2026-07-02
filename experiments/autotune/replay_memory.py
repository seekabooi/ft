# experiments/autotune/replay_memory.py
"""
Replay Memory - 经验回放缓存
"""

import random
from typing import Dict, List, Optional, Any
from collections import deque
import time


class ReplayMemory:
    """经验回放缓存"""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._memory: deque = deque(maxlen=max_size)
        self._count = 0

    def store(self, experience: Dict):
        """存储一条经验"""
        # 添加时间戳
        if 'timestamp' not in experience:
            experience['timestamp'] = time.time()
        self._memory.append(experience)
        self._count += 1

    def sample(self, batch_size: int) -> List[Dict]:
        """随机采样一批经验"""
        if len(self._memory) < batch_size:
            return list(self._memory)
        return random.sample(list(self._memory), batch_size)

    def get_all(self) -> List[Dict]:
        """获取所有经验"""
        return list(self._memory)

    def clear(self):
        """清空缓存"""
        self._memory.clear()

    def __len__(self) -> int:
        return len(self._memory)

    def to_dict(self) -> Dict:
        return {
            'max_size': self.max_size,
            'count': self._count,
            'memory': list(self._memory)
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'ReplayMemory':
        memory = cls(max_size=data.get('max_size', 1000))
        memory._count = data.get('count', 0)
        memory._memory = deque(data.get('memory', []), maxlen=memory.max_size)
        return memory

    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            'size': len(self._memory),
            'max_size': self.max_size,
            'usage_ratio': len(self._memory) / self.max_size if self.max_size > 0 else 0
        }