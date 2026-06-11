import numpy as np
from .base import BaseAgent

class FixedAgent(BaseAgent):
    def __init__(self, skill):
        self.skill = skill

    def predict(self, task):
        pred = self.skill.execute(np.array(task.history), task.horizon)
        return float(np.mean(pred))