# experiments/autotune/rule_quality_evaluator.py
import json
import re
from typing import Dict, List, Tuple
from experiments.autotune.utils import ProgressLogger


class RuleQualityEvaluator:
    """LLM驱动的规则质量评估与动作决策"""

    def __init__(self, config: Dict, logger: ProgressLogger):
        self.config = config
        self.logger = logger
        self.model = config.get('llm', {}).get('model', 'glm-4')
        self.stop_ratio = config.get('stop_ratio', 0.01)

    def decide_action(self, rules: List[Dict], performance_report: Dict) -> Tuple[str, Dict]:
        """
        返回: (action, extra_info)
        action: 'stop', 'merge', 'patch'
        """
        self.logger.log("🧠 调用LLM评估规则质量并决定下一步...")

        hard_count = performance_report.get('hard_count', 0)
        total_windows = performance_report.get('total_windows', 1)
        hard_ratio = hard_count / total_windows if total_windows > 0 else 0
        worst_3 = performance_report.get('worst_3_window_ids', [])

        report_summary = {
            'avg_mase': performance_report.get('avg_mase', 0),
            'std_mase': performance_report.get('std_mase', 0),
            'hard_window_count': hard_count,
            'total_windows': total_windows,
            'hard_window_ratio': hard_ratio,
            'worst_3_window_ids': worst_3,
            'stop_threshold': self.stop_ratio
        }

        # 规则摘要（包含条件+策略简述）
        rule_summaries = []
        for i, rule in enumerate(rules):
            strategy = rule.get('skill_strategy', {})
            stages = strategy.get('stages', [])
            stage_desc = []
            for st in stages:
                steps = st.get('steps', 0)
                weights = st.get('weights', {})
                w_str = ', '.join([f"{k}:{v:.2f}" for k, v in weights.items()])
                stage_desc.append(f"{steps}步{{{w_str}}}")
            rule_summaries.append(
                f"规则{i}: 条件=[{rule.get('condition', 'True')}] → 策略={' → '.join(stage_desc)}"
            )

        prompt = f"""你是一个时序预测规则优化专家。现有以下规则集，已在验证集上测试。

验证集性能报告：
- 平均MASE: {report_summary['avg_mase']:.4f}
- MASE标准差: {report_summary['std_mase']:.4f}
- 困难窗口数: {report_summary['hard_window_count']}/{report_summary['total_windows']}
- 困难窗口占比: {report_summary['hard_window_ratio']:.2%}
- MASE最差的3个窗口ID: {report_summary['worst_3_window_ids']}

现有规则列表（含条件与策略）：
{chr(10).join(rule_summaries)}

请分析以上规则，决定下一步优化动作（只选一个）：

1. 如果困难窗口占比 < {report_summary['stop_threshold']:.0%}，输出 "stop"。

2. ★ 如果两条规则解决的【时序问题类型】相似（例如都用于强周期、强趋势、高波动、弱季节等场景），即使条件表达式不同，也建议合并。
   合并时：条件取并集（OR连接），策略取两者中MASE更低的那个。
   输出 "merge" 并指明要合并的规则索引（如 [0, 1]）。

3. 如果存在困难窗口（特别是MASE最差的3个窗口），且它们具有相似的异常特征，输出 "patch"，目标为这3个窗口ID。

输出JSON格式：
{{
  "action": "stop|merge|patch",
  "reason": "简短决策理由（约20字）",
  "targets": [索引或窗口ID列表]
}}
只输出JSON，不要解释。"""

        try:
            from src.agents.llm_client import LLMClient
            import sys, io
            old_out = sys.stdout
            sys.stdout = io.StringIO()
            client = LLMClient(model=self.model, log_file=None)
            resp = client.call_with_retry(prompt, max_retries=2)
            sys.stdout = old_out
            content = resp.choices[0].message.content
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                decision = json.loads(json_match.group())
                action = decision.get('action', 'stop')
                targets = decision.get('targets', [])
                reason = decision.get('reason', '')
                self.logger.log(f"   LLM决策: {action} (理由: {reason})")
                if action == 'patch' and not targets:
                    targets = report_summary['worst_3_window_ids']
                    self.logger.log(f"   LLM未指定目标，自动使用MASE最差的3个窗口: {targets}")
                return action, {'targets': targets, 'reason': reason}
            else:
                self.logger.log("⚠️ 无法解析LLM响应，默认停止")
                return 'stop', {}
        except Exception as e:
            self.logger.log(f"⚠️ 评估调用失败: {e}，默认停止")
            return 'stop', {}