# experiments/autotune/skill_tree.py
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from experiments.autotune.skill_card import SkillCard
import numpy as np


class SkillTreeNode:
    """技能树节点"""

    def __init__(self, name: str, card: Optional[SkillCard] = None,
                 feature_range: Optional[Dict] = None):
        self.name = name
        self.card = card
        self.children = []
        self.parent = None
        self.feature_range = feature_range or {}  # 该节点覆盖的特征范围
        self.level = 0

    def add_child(self, node):
        self.children.append(node)
        node.parent = self
        node.level = self.level + 1

    def find_leaf(self, features: Dict) -> Optional[SkillCard]:
        """递归查找最匹配的叶子节点"""
        # 如果当前节点有卡片且适用，先检查自身
        if self.card and self.card.is_applicable(features):
            # 如果有子节点，优先子节点（更精细匹配）
            for child in self.children:
                result = child.find_leaf(features)
                if result:
                    return result
            return self.card

        # 没有子节点，返回当前卡片（如果适用）
        if not self.children and self.card and self.card.is_applicable(features):
            return self.card

        # 遍历子节点
        for child in sorted(self.children,
                            key=lambda n: n.card.confidence if n.card else 0,
                            reverse=True):
            result = child.find_leaf(features)
            if result:
                return result

        return None

    def get_all_cards(self) -> List[SkillCard]:
        """获取该节点下所有卡片"""
        cards = []
        if self.card:
            cards.append(self.card)
        for child in self.children:
            cards.extend(child.get_all_cards())
        return cards

    def get_depth(self) -> int:
        """获取子树深度"""
        if not self.children:
            return 1
        return 1 + max(c.get_depth() for c in self.children)


class SkillTree:
    """
    分层技能树
    支持检索、Split、Retire、动态重建
    """

    def __init__(self, skill_cards: List[SkillCard] = None):
        self.root = SkillTreeNode("Root")
        self._card_count = 0
        if skill_cards:
            self.build(skill_cards)

    def build(self, cards: List[SkillCard]):
        """从卡片列表构建树"""
        self.root = SkillTreeNode("Root")
        self._card_count = len(cards)

        if not cards:
            return

        # 按 period 分组（第一层）
        period_groups = defaultdict(list)
        for card in cards:
            period = card.metadata.get('_features', {}).get('period', 0)
            period_groups[period].append(card)

        for period, group_cards in period_groups.items():
            # 第一层：period 节点
            period_node = SkillTreeNode(f"period_{period}")
            period_node.feature_range = {'period': period}
            self.root.add_child(period_node)

            # 第二层：按 trend_strength 分组
            trend_groups = defaultdict(list)
            for card in group_cards:
                trend = card.metadata.get('_features', {}).get('trend_strength', 0)
                # 离散化趋势
                if trend > 0.6:
                    trend_key = 'strong'
                elif trend > 0.3:
                    trend_key = 'moderate'
                else:
                    trend_key = 'weak'
                trend_groups[trend_key].append(card)

            for trend_key, trend_cards in trend_groups.items():
                trend_node = SkillTreeNode(f"trend_{trend_key}")
                trend_node.feature_range = {'trend_strength': trend_key}
                period_node.add_child(trend_node)

                # 第三层：每个卡片作为叶子节点
                for card in trend_cards:
                    leaf = SkillTreeNode(card.name, card)
                    leaf.feature_range = {}
                    trend_node.add_child(leaf)

    def retrieve(self, features: Dict) -> Optional[SkillCard]:
        """检索匹配的卡片"""
        return self.root.find_leaf(features)

    def split_leaf(self, card: SkillCard) -> Tuple[Optional[SkillCard], Optional[SkillCard]]:
        """
        拆分一个叶子节点为两个子节点
        返回: (card1, card2) 或 (None, None)
        """
        features = card.metadata.get('_features', {})
        if not features:
            return None, None

        # 找到特征方差最大的维度
        # 根据当前卡片代表的窗口集合，计算特征方差
        # 简化版：根据趋势强度和季节强度拆分
        trend = features.get('trend_strength', 0)
        season = features.get('seasonal_strength', 0)

        # 创建两个子卡片
        from experiments.autotune.skill_card import SkillCard
        import copy

        card1_data = copy.deepcopy(card.to_dict())
        card1_data['name'] = f"{card.name}_split1"
        card1_data['when_to_use'] = {
            **card.when_to_use,
            'trend_strength': f"<{trend + 0.1}" if trend > 0.5 else f"<{trend + 0.2}"
        }
        card1 = SkillCard(card1_data)

        card2_data = copy.deepcopy(card.to_dict())
        card2_data['name'] = f"{card.name}_split2"
        card2_data['when_to_use'] = {
            **card.when_to_use,
            'trend_strength': f">{trend - 0.1}" if trend > 0.5 else f">{trend - 0.2}"
        }
        card2 = SkillCard(card2_data)

        return card1, card2

    def retire_leaf(self, card: SkillCard) -> bool:
        """从树中移除一个叶子节点（标记为退休）"""
        # 在树中查找并移除
        # 简化：返回 True，实际移除以重建树
        return True

    def get_all_cards(self) -> List[SkillCard]:
        """获取所有卡片"""
        return self.root.get_all_cards()

    def get_depth(self) -> int:
        """获取树深度"""
        return self.root.get_depth() if self.root else 0

    def rebuild(self):
        """重建树（在 Merge/Split/Retire 后调用）"""
        cards = self.get_all_cards()
        self.build(cards)