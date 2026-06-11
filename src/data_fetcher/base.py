from abc import ABC, abstractmethod

class DataSource(ABC):
    @abstractmethod
    def fetch(self, *args, **kwargs):
        """返回包含 date 和目标列的 DataFrame"""
        pass