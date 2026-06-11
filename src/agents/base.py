from abc import ABC, abstractmethod

class BaseAgent(ABC):
    @abstractmethod
    def predict(self, task):
        pass