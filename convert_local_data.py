import pandas as pd
import os

# ========== 1. 墨尔本每日最低气温 ==========
df = pd.read_csv('data/weather/daily-min-temperatures.csv', parse_dates=[0])
df.columns = ['date', 'Temp']
df.to_parquet('data/weather/daily-min-temperatures.parquet', index=False)
print('✅ 1/5 墨尔本温度 -> parquet')

# ========== 2. 每月航空乘客数 ==========
df = pd.read_csv('data/transport/airline-passengers.csv', parse_dates=[0])
df.columns = ['date', 'Passengers']
df.to_parquet('data/transport/airline-passengers.parquet', index=False)
print('✅ 2/5 航空乘客 -> parquet')

# ========== 3. 每月香槟销量 ==========
df = pd.read_csv('data/sales/monthly_champagne_sales.csv', parse_dates=[0])
df.columns = ['date', 'Sales']
df.to_parquet('data/sales/monthly-champagne-sales.parquet', index=False)
print('✅ 3/5 香槟销量 -> parquet')

# ========== 4. 每日黄金价格 ==========
df = pd.read_csv('data/finance/gold_daily.csv', parse_dates=[0])
df = df.iloc[:, :2]                          # 只保留前两列
df.columns = ['date', 'Price']
df.to_parquet('data/finance/daily-gold-price.parquet', index=False)
print('✅ 4/5 黄金价格 -> parquet')

# ========== 5. 每月太阳黑子数 ==========
df = pd.read_csv('data/nature/monthly-sunspots.csv', parse_dates=[0])
df.columns = ['date', 'Sunspots']
df.to_parquet('data/nature/monthly-sunspots.parquet', index=False)
print('✅ 5/5 太阳黑子 -> parquet')

print('\n🎉 全部转换完成！')