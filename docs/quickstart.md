# 快速开始
1. 复制 .env.example 为 .env 并填入智谱 API Key
2. `pip install -r requirements.txt`
3. `python generate_demo_data.py`
4. `python src/scheduler/daily_runner.py --date 2025-06-01 --num_tasks 3 --model glm-4-flash`
5. `streamlit run src/dashboard/app.py`