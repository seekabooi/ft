import pandas as pd
import numpy as np
import os

# 天气数据
os.makedirs("data/weather", exist_ok=True)
dates = pd.date_range('2025-01-01', periods=200, freq='D')
temp = 20 + np.cumsum(np.random.randn(200) * 0.5) + 10 * np.sin(np.arange(200) * (2*np.pi/365))
df = pd.DataFrame({'date': dates, 'temp_max': temp})
df.to_parquet('data/weather/beijing_temperature.parquet', index=False)

# 电力数据
os.makedirs("data/electricity", exist_ok=True)
demand = pd.date_range('2025-01-01', periods=200*24, freq='h')
values = 100 + 20 * np.sin(np.arange(len(demand)) * (2*np.pi/24)) + np.random.randn(len(demand))*5
df2 = pd.DataFrame({'date': demand, 'demand': values})
df2.to_parquet('data/electricity/demand.parquet', index=False)

# 交通数据
os.makedirs("data/traffic", exist_ok=True)
df3 = pd.DataFrame({'date': pd.date_range('2025-01-01', periods=100), 'flow': np.random.randint(50,100,100)})
df3.to_parquet('data/traffic/flow.parquet', index=False)

print("示例数据已生成")