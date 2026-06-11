import pandas as pd
from .base import DataSource

class WeatherDataSource(DataSource):
    def fetch(self, city, start, end, api_key):
        # 示例：实际可接入免费天气 API，如 Open-Meteo (无需 key)
        # https://open-meteo.com/
        import requests
        url = f"https://archive-api.open-meteo.com/v1/archive?latitude=39.9042&longitude=116.4074&start_date={start}&end_date={end}&daily=temperature_2m_max&timezone=Asia/Shanghai"
        resp = requests.get(url).json()
        dates = resp['daily']['time']
        temps = resp['daily']['temperature_2m_max']
        df = pd.DataFrame({'date': pd.to_datetime(dates), 'temp_max': temps})
        return df