#!/usr/bin/env python
"""
深度诊断：在 predict() 方法内部捕获异常
通过 monkey-patch 打印堆栈
"""

import sys
import os
import numpy as np
import traceback
from datetime import datetime

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print("=" * 70)
print("🔍 深度诊断：在 predict() 内部捕获异常")
print("=" * 70)

from src.agents.llm_planner import LLMPlannerAgent
from src.skills.registry import SkillRegistry
from src.skills.naive import NaiveSkill
from src.tasks.instance import TaskInstance

# ★ 备份原始 predict 方法
original_predict = LLMPlannerAgent.predict

def wrapped_predict(self, task):
    """包装 predict 方法，捕获所有异常并打印详细堆栈"""
    try:
        return original_predict(self, task)
    except Exception as e:
        print(f"\n❌ predict() 内部异常: {type(e).__name__}: {e}")
        print("完整堆栈:")
        traceback.print_exc()
        history = np.array(task.history)
        horizon = task.horizon
        print("⚠️ 使用均值回退")
        return np.full(horizon, np.mean(history[-5:]) if len(history) >= 5 else np.mean(history)).tolist()

# ★ 替换方法
LLMPlannerAgent.predict = wrapped_predict

# 创建 agent
registry = SkillRegistry()
registry.register(NaiveSkill())

agent = LLMPlannerAgent(
    model="glm-4.7",
    skill_registry=registry,
    verbose=True,
    use_skills=True,
    rules_file=None
)

# ★ 修正：使用当前时间
task = TaskInstance(
    id="test",
    dataset_id="test",
    template_id="fixed_origin",
    question="",
    question_type="numerical",
    history=np.random.randn(100).tolist(),
    horizon=7,
    frequency="daily",
    prediction_target={},
    resolution_date=datetime.now(),  # ★ 关键修复
    difficulty_level=1,
    ground_truth_extractor="",
    dates=None,
    target_date=""
)

print("\n📌 测试：有策略预测（深度捕获）...")
mock_strategy = {
    "stages": [
        {"steps": 4, "weights": {"naive": 0.7, "naive_drift": 0.3}},
        {"steps": 3, "weights": {"naive": 0.6, "naive_drift": 0.4}}
    ]
}
agent._current_rule_strategy = mock_strategy

try:
    result = agent.predict(task)
    print(f"   ✅ 预测返回，长度: {len(result)}")
except Exception as e:
    print(f"   ❌ 外部捕获: {e}")

print("\n" + "=" * 70)
print("✅ 深度诊断完成")
print("=" * 70)