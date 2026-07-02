#!/usr/bin/env python
"""
专门诊断 parse_weights_and_interval 在有策略注入时的返回值
"""

import sys
import os
import json
import re

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print("=" * 70)
print("🔍 诊断 parse_weights_and_interval（有策略注入）")
print("=" * 70)

from src.agents.llm_client import LLMClient
from src.agents.llm_planner import LLMPlannerAgent
from run_benchmark import build_full_registry

# ★ 创建 Agent
full_registry, _ = build_full_registry(no_residual=False)
agent = LLMPlannerAgent(
    model="glm-4.7",
    skill_registry=full_registry,
    verbose=False,
    use_skills=True,
    rules_file=None
)

# ★ 模拟注入策略
mock_strategy = {
    "stages": [
        {"steps": 4, "weights": {"chunk_ensemble": 0.6, "multi_resolution": 0.4}},
        {"steps": 3, "weights": {"chunk_ensemble": 0.5, "multi_resolution": 0.5}}
    ]
}
agent._current_rule_strategy = mock_strategy

# ★ 模拟调用 _format_strategy
formatted = agent._format_strategy(mock_strategy)
print(f"\n📌 _format_strategy 输出:\n{formatted}")

# ★ 模拟构建 profile 并调用 parse_weights_and_interval
# 注意：这里无法直接测试 _decide_weights_and_interval，因为它需要大量上下文。

print("\n📌 模拟 LLM 响应解析...")

# 假设 LLM 返回的 content（包含 rule_strategy 信息后的典型响应）
test_content = '''
{
    "skill_weights": {
        "chunk_ensemble": 0.6,
        "multi_resolution": 0.3,
        "naive": 0.1
    },
    "replan_interval": 3,
    "reasoning": "参照策略调整",
    "relation_to_reference": "partially_referenced"
}
'''

print(f"   测试输入: {test_content}")

client = LLMClient(model="glm-4.7", verbose=False)
result = client.parse_weights_and_interval(test_content)

print(f"\n   解析结果:")
print(f"   - weights: {result[0]}")
print(f"   - interval: {result[1]}")
print(f"   - reasoning: {result[2]}")
print(f"   - relation: {result[3]}")
print(f"   - raw_data: {result[4]}")

if isinstance(result, tuple) and len(result) == 5:
    print("\n   ✅ parse_weights_and_interval 返回5个值，正常")
else:
    print(f"\n   ❌ 返回 {len(result)} 个值，异常！")

print("\n" + "=" * 70)
print("✅ 诊断完成")
print("=" * 70)