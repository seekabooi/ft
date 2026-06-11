import argparse
import json
import os
from datetime import datetime
from src.tasks.generator import TaskGenerator
from src.agents.llm_planner import LLMPlannerAgent
from src.skills.registry import SkillRegistry
from src.evaluation.scorer import Scorer
from src.config import PREDICTIONS_DIR

def main():
    parser = argparse.ArgumentParser(description='时序版 FutureX - 每日预测流水线')
    parser.add_argument('--date', default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument('--model', default='glm-4-flash')
    parser.add_argument('--num_tasks', type=int, default=5)
    parser.add_argument('--start', default=None)
    parser.add_argument('--end', default=None)
    parser.add_argument('--agent', default='llm_planner', choices=['llm_planner'])
    parser.add_argument('--dataset', default=None, help='仅使用指定数据集ID')
    args = parser.parse_args()

    print(f"📋 正在生成 {args.num_tasks} 个任务 (日期: {args.date})")
    gen = TaskGenerator()
    failed_sources = []
    tasks = gen.generate_daily_tasks(
        date_str=args.date,
        num_tasks=args.num_tasks,
        start_override=args.start,
        end_override=args.end,
        failed_collector=failed_sources,
        dataset_filter=args.dataset
    )
    print(f"✅ 成功生成 {len(tasks)} 个任务\n")

    if failed_sources:
        print(f"⚠️ 以下 {len(failed_sources)} 个数据源不可用:")
        for src in failed_sources:
            print(f"   🔴 {src['name']} - {src['error'][:80]}")
        print()

    if len(tasks) == 0:
        print("❌ 没有可用任务")
        return

    registry = SkillRegistry()
    agent = LLMPlannerAgent(model=args.model, skill_registry=registry)

    pred_dir = os.path.join(PREDICTIONS_DIR, args.date)
    os.makedirs(pred_dir, exist_ok=True)
    pred_file = os.path.join(pred_dir, f"agent_{args.agent}.jsonl")

    total = len(tasks)
    with open(pred_file, 'w', encoding='utf-8') as f:
        for i, task in enumerate(tasks, 1):
            print(f"[{i}/{total}] {task.question[:60]}...")
            try:
                pred = agent.predict(task)
                record = {"task_id": task.id, "prediction": pred, "model": args.model}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(f"  ✅ 预测结果: {pred:.4f}\n")
            except Exception as e:
                print(f"  ❌ 失败: {e}\n")

    print("📊 评分中...")
    scorer = Scorer()
    try:
        df = scorer.run(args.date)
        if not df.empty:
            print(f"✅ 平均分: {df['score'].mean():.4f}")
            for _, row in df.iterrows():
                print(f"  {row['dataset']}: 预测={row['prediction']:.4f}, 真实={row['ground_truth']:.4f}, 分={row['score']:.4f}")
        else:
            print("⏳ 无法评分（resolution_date 未到）")
    except Exception as e:
        print(f"⚠️ 评分失败: {e}")

if __name__ == "__main__":
    main()