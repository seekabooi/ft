import numpy as np
from src.agents.llm_planner import LLMPlannerAgent
from src.skills.registry import SkillRegistry
from src.skills.naive import NaiveSkill
from src.tasks.instance import TaskInstance

print("="*70)
print("🔍 诊断 predict() 方法中的解包错误")
print("="*70)

# 创建 agent
registry = SkillRegistry()
registry.register(NaiveSkill())

agent = LLMPlannerAgent(
    model="glm-4.7",
    skill_registry=registry,
    verbose=True,
    use_skills=True
)

# 模拟任务
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
    resolution_date=None,
    difficulty_level=1,
    ground_truth_extractor="",
    dates=None,
    target_date=""
)

print("\n📌 调用 predict()...")

try:
    # 先测试无策略预测
    agent._current_rule_strategy = None
    result_no = agent.predict(task)
    print(f"   ✅ 无策略预测成功，长度: {len(result_no)}")
except Exception as e:
    print(f"   ❌ 无策略预测失败: {e}")
    import traceback
    traceback.print_exc()

# 测试有策略预测（设置一个模拟策略）
mock_strategy = {
    "stages": [
        {"steps": 4, "weights": {"naive": 0.7, "naive_drift": 0.3}},
        {"steps": 3, "weights": {"naive": 0.6, "naive_drift": 0.4}}
    ]
}
agent._current_rule_strategy = mock_strategy

print("\n📌 调用 predict() 有策略模式...")
try:
    result_with = agent.predict(task)
    print(f"   ✅ 有策略预测成功，长度: {len(result_with)}")
except Exception as e:
    print(f"   ❌ 有策略预测失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("✅ 诊断完成")
print("="*70)