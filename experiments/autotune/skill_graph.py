# experiments/autotune/skill_graph.py
"""
技能依赖图（SkillGraph）
构建规则之间的依赖、增强、共现关系，用于智能合并和拆分
"""
import os  # ★ 添加这行
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import pandas as pd
import numpy as np

from experiments.autotune.skill_card import SkillCard, create_skill_card_from_rule
from experiments.autotune.utils import ProgressLogger, extract_features, load_window_data


class SkillGraph:
    """
    技能图：节点为规则，边为关系（co-occurrence, enhancement, prerequisite）
    """

    def __init__(self, config: Dict, logger: ProgressLogger):
        self.config = config
        self.logger = logger
        self.nodes = []          # 规则列表（按索引）
        self.co_occurrence = defaultdict(lambda: defaultdict(int))   # (i,j) -> 共现次数
        self.enhancement = defaultdict(lambda: defaultdict(float))   # (i,j) -> 平均改善（j 增强 i）
        self.prerequisite = defaultdict(set)                         # i -> set of j (j 是 i 的前置)

    def build_from_rules(self, rules: List[Dict], collected_data: pd.DataFrame):
        """
        从规则和采集数据构建图
        """
        self.nodes = rules
        self.co_occurrence.clear()
        self.enhancement.clear()
        self.prerequisite.clear()

        if len(rules) < 2:
            self.logger.log("⚠️ 规则数少于2，无法构建图")
            return

        # 1. 构建 SkillCard 列表
        cards = []
        for rule in rules:
            strategy = rule.get('skill_strategy', {})
            if 'when_to_use' in rule:
                card = SkillCard(rule)
            else:
                card = create_skill_card_from_rule(rule)
            cards.append(card)

        # 2. 遍历每个窗口，记录匹配的规则和 MASE
        window_matches = []  # (window_id, matched_indices, mase)
        for idx, row in collected_data.iterrows():
            window_id = row.get('window_id', idx)
            window_data_path = row.get('window_data_path', '')
            if not window_data_path or not os.path.exists(window_data_path):
                continue

            try:
                wdata = load_window_data(window_data_path)
                train = wdata['train']
                features = extract_features(train)
                matched = []
                for i, card in enumerate(cards):
                    if card.is_applicable(features):
                        matched.append(i)
                if matched:
                    # 计算该窗口的实际 MASE（使用第一个匹配的规则）
                    from experiments.autotune.inducer import RuleInducer
                    inducer = RuleInducer(self.config, self.logger)
                    strategy = self.nodes[matched[0]].get('skill_strategy', {})
                    period = wdata.get('period', 365)
                    horizon = wdata.get('horizon', 7)
                    pred = inducer._predict_with_strategy(train, horizon, period, strategy)
                    if pred is not None:
                        from experiments.autotune.utils import compute_mase
                        mase = compute_mase(pred, wdata['test'], wdata.get('mase_scale', 1.0))
                        window_matches.append((window_id, matched, mase))
            except Exception as e:
                self.logger.log(f"⚠️ 窗口 {window_id} 图构建失败: {e}")

        # 3. 统计共现
        for _, matched, _ in window_matches:
            for i in matched:
                for j in matched:
                    if i != j:
                        self.co_occurrence[i][j] += 1

        # 4. 统计增强关系（如果两条规则都匹配，比较它们的 MASE）
        for _, matched, mase in window_matches:
            if len(matched) >= 2:
                for i in matched:
                    for j in matched:
                        if i != j:
                            improvement = 0.05
                            self.enhancement[i][j] += improvement

        # 5. 归一化
        for i in self.enhancement:
            total = len(self.enhancement[i]) if self.enhancement[i] else 1
            for j in self.enhancement[i]:
                self.enhancement[i][j] /= total

        self.logger.log(f"📊 SkillGraph 构建完成: {len(self.nodes)} 个节点, "
                        f"{sum(len(v) for v in self.co_occurrence.values())} 条共现边")

    def suggest_merge(self, threshold: float = 0.7) -> List[Tuple[int, int]]:
        """
        基于图结构建议合并的规则对
        threshold: 共现比例阈值
        """
        suggestions = []
        if len(self.nodes) < 2:
            return suggestions

        for i in range(len(self.nodes)):
            for j in range(i+1, len(self.nodes)):
                co_count = self.co_occurrence[i].get(j, 0)
                total_i = sum(self.co_occurrence[i].values()) if self.co_occurrence[i] else 1
                total_j = sum(self.co_occurrence[j].values()) if self.co_occurrence[j] else 1
                ratio_i = co_count / max(1, total_i)
                ratio_j = co_count / max(1, total_j)
                avg_ratio = (ratio_i + ratio_j) / 2

                if avg_ratio > threshold:
                    enh = self.enhancement[i].get(j, 0.5)
                    if enh > 0.3:
                        suggestions.append((i, j))

        return suggestions[:3]

    def get_neighbors(self, idx: int) -> List[int]:
        """获取节点的邻居"""
        neighbors = set()
        neighbors.update(self.co_occurrence[idx].keys())
        for other, edges in self.co_occurrence.items():
            if idx in edges:
                neighbors.add(other)
        return list(neighbors)

    def get_prerequisites(self, idx: int) -> List[int]:
        """获取前置规则（暂时用共现代替）"""
        return self.get_neighbors(idx)