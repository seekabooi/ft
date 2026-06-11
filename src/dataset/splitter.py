import pandas as pd
from datetime import timedelta

def split_history_future(df, target_col, prediction_target, template, frequency='daily'):
    date_str = prediction_target.get('date')
    if date_str:
        ref_date = pd.Timestamp(date_str)
        if frequency == 'hourly':
            history = df.loc[:ref_date - timedelta(hours=1), target_col].values
        elif frequency == 'minutely':
            history = df.loc[:ref_date - timedelta(minutes=1), target_col].values
        else:
            history = df.loc[:ref_date - timedelta(days=1), target_col].values
    else:
        horizon = prediction_target.get('horizon', template.get('horizon', 1))
        if frequency == 'hourly':
            cutoff = df.index[-1] - timedelta(hours=horizon)
        elif frequency == 'minutely':
            cutoff = df.index[-1] - timedelta(minutes=horizon)
        else:
            cutoff = df.index[-1] - timedelta(days=horizon)
        history = df.loc[:cutoff, target_col].values
    return history.tolist()