import sys, os, time
sys.path.insert(0, '.')
from dotenv import load_dotenv
load_dotenv()
from src.data_fetcher.api_source import ApiDataSource

test_configs = [
    # 天气（5）
    {'name': '北京温度', 'params': {'fetch_type':'openmeteo', 'latitude':39.9, 'longitude':116.4, 'start':'2024-06-01', 'end':'2024-06-10', 'target_column':'temp_max'}},
    {'name': '上海降水', 'params': {'fetch_type':'openmeteo', 'latitude':31.2, 'longitude':121.5, 'start':'2024-06-01', 'end':'2024-06-10', 'target_column':'precip', 'daily_metric':'precipitation_sum'}},
    {'name': '广州风速', 'params': {'fetch_type':'openmeteo', 'latitude':23.1, 'longitude':113.3, 'start':'2024-06-01', 'end':'2024-06-10', 'target_column':'wind', 'daily_metric':'wind_speed_10m_max'}},
    {'name': '深圳湿度', 'params': {'fetch_type':'openmeteo', 'latitude':22.5, 'longitude':114.1, 'start':'2024-06-01', 'end':'2024-06-10', 'target_column':'humidity', 'daily_metric':'relative_humidity_2m_mean'}},
    {'name': '北京日照', 'params': {'fetch_type':'openmeteo', 'latitude':39.9, 'longitude':116.4, 'start':'2024-06-01', 'end':'2024-06-10', 'target_column':'sunshine', 'daily_metric':'sunshine_duration'}},
    # 空气质量（5）
    {'name': '北京PM2.5', 'params': {'fetch_type':'openmeteo_air', 'latitude':39.9, 'longitude':116.4, 'start':'2024-06-01', 'target_column':'pm25'}},
    {'name': '上海PM2.5', 'params': {'fetch_type':'openmeteo_air', 'latitude':31.2, 'longitude':121.5, 'start':'2024-06-01', 'target_column':'pm25'}},
    {'name': '广州PM2.5', 'params': {'fetch_type':'openmeteo_air', 'latitude':23.1, 'longitude':113.3, 'start':'2024-06-01', 'target_column':'pm25'}},
    {'name': '深圳PM2.5', 'params': {'fetch_type':'openmeteo_air', 'latitude':22.5, 'longitude':114.1, 'start':'2024-06-01', 'target_column':'pm25'}},
    {'name': '成都PM2.5', 'params': {'fetch_type':'openmeteo_air', 'latitude':30.6, 'longitude':104.1, 'start':'2024-06-01', 'target_column':'pm25'}},
    # 外汇（5）
    {'name': 'USD/CNY', 'params': {'fetch_type':'frankfurter', 'from_currency':'USD', 'to_currency':'CNY', 'start':'2024-06-01', 'target_column':'rate'}},
    {'name': 'EUR/USD', 'params': {'fetch_type':'frankfurter', 'from_currency':'EUR', 'to_currency':'USD', 'start':'2024-06-01', 'target_column':'rate'}},
    {'name': 'GBP/USD', 'params': {'fetch_type':'frankfurter', 'from_currency':'GBP', 'to_currency':'USD', 'start':'2024-06-01', 'target_column':'rate'}},
    {'name': 'USD/JPY', 'params': {'fetch_type':'frankfurter', 'from_currency':'USD', 'to_currency':'JPY', 'start':'2024-06-01', 'target_column':'rate'}},
    {'name': 'AUD/USD', 'params': {'fetch_type':'frankfurter', 'from_currency':'AUD', 'to_currency':'USD', 'start':'2024-06-01', 'target_column':'rate'}},
    # A股（5）
    {'name': '茅台', 'params': {'fetch_type':'tencent', 'symbol':'sh600519', 'target_column':'close'}},
    {'name': '宁德', 'params': {'fetch_type':'tencent', 'symbol':'sz300750', 'target_column':'close'}},
    {'name': '平安', 'params': {'fetch_type':'tencent', 'symbol':'sh601318', 'target_column':'close'}},
    {'name': '招行', 'params': {'fetch_type':'tencent', 'symbol':'sh600036', 'target_column':'close'}},
    {'name': '美的', 'params': {'fetch_type':'tencent', 'symbol':'sz000333', 'target_column':'close'}},
]

ds = ApiDataSource()
for cfg in test_configs:
    print(f"测试 {cfg['name']} ... ", end='', flush=True)
    try:
        df = ds.fetch(**cfg['params'])
        print(f"✅ {len(df)} 行")
    except Exception as e:
        print(f"❌ {str(e)[:80]}")
    time.sleep(0.5)