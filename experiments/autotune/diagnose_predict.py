#!/usr/bin/env python
"""
诊断 predict() 方法中的解包错误
精确定位 "not enough values to unpack (expected 5, got 2)" 的来源
"""

import sys
import os
import numpy as np
from datetime import datetime

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print("=" * 70)
print("🔍 诊断 predict() 方法中的解包错误")
print("=" * 70)

# 导入所需模块
from src.agents.llm_planner import LLMPlannerAgent
from src.skills.registry import SkillRegistry
from src.skills.naive import NaiveSkill
from src.tasks.instance import TaskInstance

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

# ★ 修正：使用当前时间作为 resolution_date
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

print("\n📌 测试 1：无策略预测...")
try:
    agent._current_rule_strategy = None
    result_no = agent.predict(task)
    print(f"   ✅ 无策略预测成功，长度: {len(result_no)}")
except Exception as e:
    print(f"   ❌ 无策略预测失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)

print("\n📌 测试 2：有策略预测（模拟策略）...")
mock_strategy = {
    "stages": [
        {"steps": 4, "weights": {"naive": 0.7, "naive_drift": 0.3}},
        {"steps": 3, "weights": {"naive": 0.6, "naive_drift": 0.4}}
    ]
}

agent._current_rule_strategy = mock_strategy

try:
    result_with = agent.predict(task)
    print(f"   ✅ 有策略预测成功，长度: {len(result_with)}")
except Exception as e:
    print(f"   ❌ 有策略预测失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("✅ 诊断完成")
print("=" * 70)