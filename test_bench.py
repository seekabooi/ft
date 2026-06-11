print("Script started")
import sys
print("sys.path ok")
try:
    from src.agents.llm_planner import LLMPlannerAgent
    print("Import llm_planner ok")
except Exception as e:
    print(f"Import error: {e}")
    sys.exit(1)

try:
    from src.evaluation.fixed_origin_evaluator import FixedOriginEvaluator
    print("Import evaluator ok")
except Exception as e:
    print(f"Import error: {e}")
    sys.exit(1)

print("All imports successful, now running minimal test")
# 创建一个假的agent和任务，仅测试调用
class DummyAgent:
    def predict(self, task):
        return [1.0] * task.horizon

evaluator = FixedOriginEvaluator(DummyAgent(), min_train_size=10, horizon=2)
try:
    result = evaluator.evaluate("airline_passengers")
    print("Evaluation result:", result)
except Exception as e:
    print("Evaluation error:", e)