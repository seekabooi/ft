import pandas as pd
import requests
from datetime import datetime, timedelta
from .base import DataSource


class ApiDataSource(DataSource):
    def fetch(self, **params):
        fetch_type = params.pop('fetch_type', 'rest')
        if fetch_type == 'openmeteo':
            return self._fetch_openmeteo(params)
        elif fetch_type == 'openmeteo_air':
            return self._fetch_openmeteo_air(params)
        elif fetch_type == 'frankfurter':
            return self._fetch_frankfurter(params)
        elif fetch_type == 'tencent':
            return self._fetch_tencent(params)
        elif fetch_type == 'rest':
            return self._fetch_rest(params)
        else:
            raise ValueError(f"Unknown fetch_type: {fetch_type}")

    def _fetch_openmeteo(self, params):
        latitude = params['latitude']
        longitude = params['longitude']
        start = params.get('start', '2020-01-01')
        end = params.get('end', '2025-12-31')
        target_col = params.get('target_column', 'temp_max')
        daily_metric = params.get('daily_metric', 'temperature_2m_max')
        url = 'https://archive-api.open-meteo.com/v1/archive'
        payload = {
            'latitude': latitude,
            'longitude': longitude,
            'start_date': start,
            'end_date': end,
            'daily': daily_metric,
            'timezone': 'Asia/Shanghai'
        }
        resp = requests.get(url, params=payload, timeout=30).json()
        dates = resp['daily']['time']
        values = resp['daily'][daily_metric]
        df = pd.DataFrame({'date': pd.to_datetime(dates), target_col: values})
        return df[['date', target_col]].sort_values('date')

    def _fetch_openmeteo_air(self, params):
        latitude = params['latitude']
        longitude = params['longitude']
        start = params.get('start', '2024-01-01')
        target_col = params.get('target_column', 'pm25')
        end = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        url = 'https://air-quality-api.open-meteo.com/v1/air-quality'
        payload = {
            'latitude': latitude,
            'longitude': longitude,
            'hourly': 'pm2_5',
            'start_date': start,
            'end_date': end,
            'timezone': 'Asia/Shanghai'
        }
        resp = requests.get(url, params=payload, timeout=30).json()
        times = resp['hourly']['time']
        values = resp['hourly']['pm2_5']
        df = pd.DataFrame({'time': pd.to_datetime(times), 'pm2_5': values})
        df['date'] = df['time'].dt.date
        daily = df.groupby('date')['pm2_5'].mean().reset_index()
        daily['date'] = pd.to_datetime(daily['date'])
        daily = daily.rename(columns={'pm2_5': target_col})
        return daily[['date', target_col]].sort_values('date')

    def _fetch_frankfurter(self, params):
        from_cur = params['from_currency']
        to_cur = params['to_currency']
        start = params.get('start', '2020-01-01')
        target_col = params.get('target_column', 'rate')
        url = f'https://api.frankfurter.app/{start}..?from={from_cur}&to={to_cur}'
        resp = requests.get(url, timeout=30)
        if not resp.text.strip():
            raise Exception(f"Frankfurter 返回空: {from_cur}/{to_cur}")
        data = resp.json()
        if 'rates' not in data:
            raise Exception(f"Frankfurter 格式错误: {str(data)[:80]}")
        dates, values = [], []
        for date, rate_dict in sorted(data['rates'].items()):
            dates.append(date)
            values.append(rate_dict[to_cur])
        df = pd.DataFrame({'date': pd.to_datetime(dates), target_col: values})
        return df[['date', target_col]].sort_values('date')

    def _fetch_tencent(self, params):
        """腾讯财经K线接口，兼容股票和期货"""
        symbol = params['symbol']
        target_col = params.get('target_column', 'close')
        url = 'http://web.ifzq.gtimg.cn/appstock/app/fqkline/get'
        payload = {'param': f'{symbol},day,,,320,qfq'}
        resp = requests.get(url, params=payload, timeout=30).json()

        if resp.get('code') != 0:
            raise Exception(f"腾讯财经错误: {resp.get('msg', 'unknown')}")

        stock_data = resp.get('data', {}).get(symbol, {})

        # 尝试获取 qfqday / day / klines
        klines = stock_data.get('qfqday') or stock_data.get('day')

        # 期货返回格式可能是 list
        if klines is None and isinstance(stock_data, list):
            klines = stock_data

        if not klines:
            raise Exception(f"腾讯财经无K线数据: {symbol}")

        dates, closes = [], []
        for line in klines:
            if isinstance(line, str):
                parts = line.split(',')
            elif isinstance(line, list):
                parts = line
            else:
                continue
            if len(parts) < 3:
                continue
            dates.append(parts[0])
            closes.append(float(parts[2]))

        df = pd.DataFrame({'date': pd.to_datetime(dates), target_col: closes})
        return df[['date', target_col]].sort_values('date')

    def _fetch_rest(self, params):
        pass