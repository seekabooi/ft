#!/usr/bin/env python
"""
真实环境诊断：使用完整技能注册表 + 有策略预测
精确定位 'expected 5, got 2' 错误来源
"""

import sys
import os
import numpy as np
import traceback
from datetime import datetime
import json

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print("=" * 70)
print("🔍 真实环境诊断：完整技能 + 有策略预测")
print("=" * 70)

# 导入完整技能注册表
from run_benchmark import build_full_registry
from src.agents.llm_planner import LLMPlannerAgent
from src.tasks.instance import TaskInstance

# ★ 构建完整技能注册表（28个技能）
print("\n📌 构建完整技能注册表...")
full_registry, all_skills = build_full_registry(no_residual=False)
print(f"   注册了 {len(all_skills)} 个技能")
print(f"   技能列表: {[s.name for s in all_skills[:5]]}...")

# ★ 创建 Agent（使用完整注册表）
agent = LLMPlannerAgent(
    model="glm-4.7",
    skill_registry=full_registry,
    verbose=True,
    use_skills=True,
    rules_file=None,
    llm_call_interval=3
)
agent.rule_engine = None  # 禁用规则引擎，只测试策略参考

# ★ 模拟任务
task = TaskInstance(
    id="real_test",
    dataset_id="melbourne_temp",
    template_id="fixed_origin",
    question="",
    question_type="numerical",
    history=np.random.randn(200).tolist(),
    horizon=7,
    frequency="daily",
    prediction_target={},
    resolution_date=datetime.now(),
    difficulty_level=1,
    ground_truth_extractor="",
    dates=None,
    target_date=""
)

print("\n" + "=" * 70)
print("📌 测试 1：无策略预测（基线）")
print("=" * 70)
try:
    agent._current_rule_strategy = None
    result_no = agent.predict(task)
    print(f"   ✅ 无策略预测成功，长度: {len(result_no)}")
    print(f"   预测值: {result_no[:5]}...")
except Exception as e:
    print(f"   ❌ 无策略预测失败: {e}")
    traceback.print_exc()

print("\n" + "=" * 70)
print("📌 测试 2：有策略预测（模拟策略）")
print("=" * 70)

# ★ 模拟一个策略（与真实策略格式相同）
mock_strategy = {
    "stages": [
        {"steps": 4, "weights": {"chunk_ensemble": 0.6, "multi_resolution": 0.4}},
        {"steps": 3, "weights": {"chunk_ensemble": 0.5, "multi_resolution": 0.5}}
    ]
}
print(f"   注入策略: {json.dumps(mock_strategy, indent=2)}")

agent._current_rule_strategy = mock_strategy

try:
    result_with = agent.predict(task)
    print(f"   ✅ 有策略预测成功，长度: {len(result_with)}")
    print(f"   预测值: {result_with[:5]}...")
except Exception as e:
    print(f"   ❌ 有策略预测失败: {e}")
    print("\n完整堆栈:")
    traceback.print_exc()

print("\n" + "=" * 70)
print("✅ 真实环境诊断完成")
print("=" * 70)