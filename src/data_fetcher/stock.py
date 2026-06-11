import pandas as pd
from .base import DataSource

class YFinanceDataSource(DataSource):
    def fetch(self, symbol, start="2020-01-01", end=None):
        import yfinance as yf
        if end is None:
            end = pd.Timestamp.today().strftime("%Y-%m-%d")
        df = yf.download(symbol, start=start, end=end, progress=False)
        df = df.reset_index()
        df = df.rename(columns={"Date": "date", "Close": "close"})[["date", "close"]]
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date")