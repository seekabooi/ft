import pandas as pd
import os
from src.config import STORAGE_DIR

def get_leaderboard():
    path = os.path.join(STORAGE_DIR, "leaderboard.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()