import streamlit as st
import pandas as pd
import os
from src.config import STORAGE_DIR, SCORES_DIR
from src.evaluation.leaderboard import get_leaderboard

st.set_page_config(page_title="时序版 FutureX 排行榜")
st.title("📈 LLM Agent 时间序列预测技能评估")

leaderboard = get_leaderboard()
if not leaderboard.empty:
    st.dataframe(leaderboard.sort_values('score', ascending=False))
else:
    st.info("暂无排行榜数据，请运行 daily_runner.py 生成")

score_files = []
if os.path.exists(SCORES_DIR):
    score_files = sorted([f for f in os.listdir(SCORES_DIR) if f.endswith('.parquet')])
if score_files:
    latest = pd.read_parquet(os.path.join(SCORES_DIR, score_files[-1]))
    st.subheader("最新任务评分明细")
    st.dataframe(latest)