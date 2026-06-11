import numpy as np

def normalized_score(pred: float, truth: float, history_std: float) -> float:
    error = abs(pred - truth)
    if history_std == 0:
        return 1.0 if error == 0 else 0.0
    if error <= history_std:
        return 1 - (error / history_std) * 0.5
    else:
        return max(0.0, 1 - (error - history_std) / history_std)