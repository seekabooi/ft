import pandas as pd
import os

# ========== 1. 墨尔本每日最低气温 ==========
os.makedirs('data/weather', exist_ok=True)
# 1981-1990 年的每日最低温度
dates = pd.date_range('1981-01-01', '1990-12-31', freq='D')
import numpy as np
np.random.seed(42)
base = 10 + 5 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)  # 年周期
noise = np.random.normal(0, 2, len(dates))
temps = base + noise
df = pd.DataFrame({'date': dates, 'Temp': temps})
df.to_parquet('data/weather/daily-min-temperatures.parquet', index=False)

# ========== 2. 每月航空乘客数 ==========
os.makedirs('data/transport', exist_ok=True)
dates = pd.date_range('1949-01-01', '1960-12-01', freq='MS')
# 经典航空乘客数据（近似）
base = np.array([112,118,132,129,121,135,148,148,136,119,104,118])
passengers = []
for i in range(12):  # 12年
    trend = i * 20
    seasonal = base * (1 + i * 0.1)
    noise = np.random.normal(0, 5, 12)
    passengers.extend(seasonal + trend + noise)
passengers = passengers[:len(dates)]
df = pd.DataFrame({'date': dates, 'Passengers': passengers})
df.to_parquet('data/transport/airline-passengers.parquet', index=False)

# ========== 3. 每月香槟销量 ==========
os.makedirs('data/sales', exist_ok=True)
dates = pd.date_range('1964-01-01', '1972-09-01', freq='MS')
np.random.seed(123)
trend = np.arange(len(dates)) * 50
seasonal = 2000 + 1500 * np.sin(2 * np.pi * np.arange(len(dates)) / 12)
noise = np.random.normal(0, 300, len(dates))
sales = seasonal + trend + noise
df = pd.DataFrame({'date': dates, 'Sales': sales})
df.to_parquet('data/sales/monthly-champagne-sales.parquet', index=False)

# ========== 4. 每日黄金价格（2016-2017） ==========
os.makedirs('data/finance', exist_ok=True)
dates = pd.date_range('2016-01-01', '2017-12-31', freq='D')
np.random.seed(99)
price = 1200
prices = []
for _ in dates:
    price += np.random.normal(0, 5)
    price = max(1000, min(1400, price))
    prices.append(price)
df = pd.DataFrame({'date': dates, 'Price': prices})
df.to_parquet('data/finance/daily-gold-price.parquet', index=False)

# ========== 5. 每月太阳黑子数 ==========
os.makedirs('data/nature', exist_ok=True)
dates = pd.date_range('1749-01-01', '1983-12-01', freq='MS')
np.random.seed(7)
# 模拟 11 年周期
t = np.arange(len(dates))
sunspots = 50 + 40 * np.sin(2 * np.pi * t / (11 * 12)) + np.random.normal(0, 15, len(dates))
sunspots = np.abs(sunspots)
df = pd.DataFrame({'date': dates, 'Sunspots': sunspots})
df.to_parquet('data/nature/monthly-sunspots.parquet', index=False)

print("All 5 datasets created!")
print("  data/weather/daily-min-temperatures.parquet")
print("  data/transport/airline-passengers.parquet")
print("  data/sales/monthly-champagne-sales.parquet")
print("  data/finance/daily-gold-price.parquet")
print("  data/nature/monthly-sunspots.parquet")