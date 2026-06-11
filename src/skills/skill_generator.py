import json
import re
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any


class SkillGenerator:
    """技能状态卡生成与失败案例归纳工具，尚未自动生成可执行 skill。目前用于从失败日志中聚类并调用 LLM 归纳状态卡草案，生成的 JSON 仅作参考，需人工确认后转为可执行代码。"""

    def __init__(
        self,
        log_file: str = "storage/logs/agent.log",
        output_dir: str = "src/skills/generated",
        error_threshold: float = 1.5,
    ):
        self.log_file = Path(log_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.error_threshold = error_threshold

    def run_generation(self) -> List[Dict]:
        failures = self.collect_failures()
        if len(failures) < 5:
            print(f"失败案例不足 ({len(failures)} < 5)，跳过技能生成")
            return []

        clusters = self.cluster_failures(failures)
        generated = []

        for i, cluster in enumerate(clusters):
            skill_info = self.generate_skill_from_cluster(cluster, i)
            if skill_info:
                skill_file = self.output_dir / f"{skill_info['name']}.json"
                with open(skill_file, "w", encoding="utf-8") as f:
                    json.dump(skill_info, f, ensure_ascii=False, indent=2)
                generated.append(skill_info)
                print(
                    f"生成新技能草案: {skill_info['name']} "
                    f"(覆盖 {skill_info['cluster_size']} 个案例) - 需人工确认后转为可执行代码"
                )

        return generated

    def collect_failures(self) -> List[Dict]:
        if not self.log_file.exists():
            print(f"日志文件 {self.log_file} 不存在")
            return []

        failures = []
        with open(self.log_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                record = None
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    pass

                if record:
                    error = record.get("error")
                    if error is None and "prediction" in record and "actual" in record:
                        error = abs(record["prediction"] - record["actual"])
                    if error is not None and error > self.error_threshold:
                        failures.append(record)
                else:
                    task_match = re.search(r"任务\s+(\S+)", line)
                    error_match = re.search(
                        r"误差[=:]\s*([\d.]+)|error[=:]\s*([\d.]+)", line, re.IGNORECASE
                    )
                    if task_match and error_match:
                        task_id = task_match.group(1)
                        err_val = float(error_match.group(1) or error_match.group(2))
                        if err_val > self.error_threshold:
                            failures.append(
                                {"task_id": task_id, "error": err_val, "raw_line": line}
                            )

        print(f"从日志中收集到 {len(failures)} 个失败案例")
        return failures

    def cluster_failures(self, failures: List[Dict], n_clusters: int = 3) -> List[List[Dict]]:
        if len(failures) < n_clusters:
            return [failures]

        errors = [f.get("error", 0) for f in failures]
        thresholds = np.percentile(errors, np.linspace(0, 100, n_clusters + 1)[1:-1])
        clusters = [[] for _ in range(n_clusters)]

        for f in failures:
            err = f.get("error", 0)
            idx = int(np.searchsorted(thresholds, err))
            clusters[idx].append(f)

        return [c for c in clusters if len(c) > 0]

    def generate_skill_from_cluster(self, cluster: List[Dict], cluster_id: int) -> Dict:
        if not cluster:
            return {}

        avg_error = float(np.mean([f.get("error", 0) for f in cluster]))
        avg_seasonal = float(
            np.mean(
                [
                    f.get("profile", {}).get("seasonal_strength", 0)
                    for f in cluster
                ]
            )
        )
        avg_trend = float(
            np.mean(
                [f.get("profile", {}).get("trend_strength", 0) for f in cluster]
            )
        )
        avg_adf = float(
            np.mean(
                [f.get("profile", {}).get("adf_pvalue", 0.5) for f in cluster]
            )
        )

        prompt = f"""
以下是一组预测失败案例的共同特征：
- 案例数量：{len(cluster)}
- 平均误差：{avg_error:.2f}
- 平均季节性强度：{avg_seasonal:.2f}
- 平均趋势强度：{avg_trend:.2f}
- 平均 ADF p 值：{avg_adf:.3f}

请根据这些特征，为一项新的时序预测技能生成以下内容（JSON 格式）：
{{
  "name": "技能英文名",
  "description": "技能中文描述",
  "state_card": {{
    "when_to_use": {{"conditions": [], "logic": "AND"}},
    "when_not_to_use": {{"conditions": [], "logic": "OR"}},
    "visible_cues": [],
    "verification_cue": "",
    "available_views": []
  }}
}}

要求：
- 条件字段使用 field/op/value 三元组，例如 {{"field": "adf_pvalue", "op": "<", "value": 0.05}}
- 只输出 JSON，不要任何解释。
"""
        try:
            from openai import OpenAI
            import os
            from dotenv import load_dotenv

            load_dotenv()
            client = OpenAI(
                api_key=os.getenv("ZHIPU_API_KEY"),
                base_url=os.getenv("OPENAI_API_BASE", "https://open.bigmodel.cn/api/paas/v4"),
            )
            resp = client.chat.completions.create(
                model="glm-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=500,
            )
            raw = resp.choices[0].message.content
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw[raw.index("{") :]
                raw = raw[: raw.rindex("}") + 1]
            skill_data = json.loads(raw)
        except Exception as e:
            print(f"LLM 调用失败，使用默认技能模板: {e}")
            skill_data = {
                "name": f"generated_skill_{cluster_id}",
                "description": f"从 {len(cluster)} 个失败案例自动生成",
                "state_card": {
                    "when_to_use": {
                        "conditions": [
                            {"field": "seasonal_strength", "op": ">", "value": 0.5}
                        ],
                        "logic": "AND",
                    },
                    "when_not_to_use": {"conditions": [], "logic": "OR"},
                    "visible_cues": [],
                    "verification_cue": f"误差应小于 {avg_error:.2f}",
                },
            }

        skill_data["cluster_size"] = len(cluster)
        skill_data["avg_error"] = avg_error
        skill_data["created_at"] = datetime.now().isoformat()

        return skill_data