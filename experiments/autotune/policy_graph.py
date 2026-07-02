# experiments/autotune/policy_graph.py
"""
Policy Graph - v6 强化学习版
从决策核心转为经验记忆库的组织层
保留了聚类结构用于统计和可视化，但不再参与实时决策

★ 增加新建簇门控 should_create_new_cluster
★ add_policy 时醒目打印簇创建信息
★ 增加复活策略的簇分配逻辑
★ 增加强制建簇方法 force_create_cluster
"""

import json
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import hashlib
import time

from experiments.autotune.skill_policy import SkillPolicy
from experiments.autotune.cluster import PolicySpacePartitioner


@dataclass
class PolicyCluster:
    """
    策略簇 - 仅用于统计和可视化
    """
    id: str
    centroid: List[float]
    scene_label: str
    policies: List[str]
    avg_mase: float
    creation_round: int
    is_active: bool = True
    coverage_score: float = 0.0
    redundancy_score: float = 0.0
    last_updated: str = ""

    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'centroid': self.centroid,
            'scene_label': self.scene_label,
            'policies': self.policies,
            'avg_mase': self.avg_mase,
            'creation_round': self.creation_round,
            'is_active': self.is_active,
            'coverage_score': self.coverage_score,
            'redundancy_score': self.redundancy_score,
            'last_updated': self.last_updated
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'PolicyCluster':
        return cls(
            id=data.get('id', ''),
            centroid=data.get('centroid', []),
            scene_label=data.get('scene_label', ''),
            policies=data.get('policies', []),
            avg_mase=data.get('avg_mase', 0.0),
            creation_round=data.get('creation_round', 0),
            is_active=data.get('is_active', True),
            coverage_score=data.get('coverage_score', 0.0),
            redundancy_score=data.get('redundancy_score', 0.0),
            last_updated=data.get('last_updated', '')
        )


@dataclass
class PolicyGraph:
    clusters: List[PolicyCluster] = field(default_factory=list)
    unassigned: List[str] = field(default_factory=list)
    version: int = 1
    created_at: str = ""
    updated_at: str = ""

    # 配置参数
    CLUSTER_DISTANCE_THRESHOLD: float = 0.3
    MIN_POLICIES_PER_CLUSTER: int = 1
    MAX_POLICIES_PER_CLUSTER: int = 10

    # 经验记录
    _experience_history: List[Dict] = field(default_factory=list)
    _max_experience: int = 1000

    # ---------- 新建簇门控参数（可配置） ----------
    # 调低阈值使簇创建更容易，但不会太频繁
    MIN_CLUSTER_DISTANCE: float = 0.45          # 绝对距离阈值（调低）
    GAP_FACTOR: float = 1.5                    # 相对间隙倍数（调低）
    MIN_SAMPLES: int = 2                       # 最小采样次数（调低）
    MIN_SIMILAR_COUNT: int = 2                 # 相似候选最小数量

    def should_create_new_cluster(self, policy: SkillPolicy, context: Dict) -> bool:
        """
        判断是否应该为策略创建新簇
        context 包含:
            - 'current_round': int
            - 'policy_dict': Dict[str, SkillPolicy]
            - 'global_avg_mase': float
            - 'worst_cluster_avg_mase': float
        """
        if not policy.embedding:
            return False

        # 如果没有现有簇，直接创建
        if not self.clusters:
            return True

        # 获取现有簇信息
        existing_clusters = []
        for cluster in self.clusters:
            if not cluster.is_active:
                continue
            embeds = []
            policy_dict = context.get('policy_dict', {})
            for pid in cluster.policies:
                p = policy_dict.get(pid)
                if p and hasattr(p, 'embedding') and p.embedding:
                    embeds.append(p.embedding)
            existing_clusters.append({
                'centroid': cluster.centroid,
                'policies': embeds,
                'avg_mase': cluster.avg_mase,
                'id': cluster.id
            })

        # 如果没有活跃簇，直接创建
        if not existing_clusters:
            return True

        # 条件 a: 绝对距离阈值
        min_dist = PolicySpacePartitioner.compute_min_distance_to_clusters(
            policy.embedding, existing_clusters
        )
        if min_dist <= self.MIN_CLUSTER_DISTANCE:
            return False

        # 条件 b: 相对间隙
        avg_intra = PolicySpacePartitioner.compute_average_intra_distance(existing_clusters)
        if avg_intra == 0:
            # 如果簇内距离为0（所有策略embedding相同），允许创建
            return True
        if min_dist <= avg_intra * self.GAP_FACTOR:
            return False

        # 条件 c: 历史表现（基于采样次数和 MASE）
        performance = context.get('policy_performance', {}).get(policy.policy_id, {})
        selection_count = performance.get('selection_count', 0)
        avg_mase = performance.get('avg_mase', float('inf'))
        global_avg = context.get('global_avg_mase', float('inf'))
        worst_cluster_avg = context.get('worst_cluster_avg_mase', float('inf'))

        cond_c = False
        if selection_count >= self.MIN_SAMPLES:
            if avg_mase < global_avg or avg_mase < worst_cluster_avg:
                cond_c = True

        # 如果没有任何历史数据，但策略本身MASE很低，也允许创建（新策略可能是好策略）
        if selection_count == 0 and policy.avg_mase < global_avg:
            cond_c = True

        # 最终决策：a and b and c
        return cond_c

    def force_create_cluster(self, policy: SkillPolicy, context: Optional[Dict] = None) -> str:
        """
        强制为策略创建新簇（用于复活策略或特殊情况）
        跳过所有门控条件
        """
        cluster_id = self._create_cluster_for_stat(policy)
        # 醒目打印
        round_info = context.get('current_round', 'N/A') if context else 'N/A'
        print("\n" + "█" * 80)
        print(f"█  🆕 强制创建新簇（用于复活策略）")
        print(f"█  ID: {cluster_id}")
        print(f"█  场景标签: {policy.semantic_description or '未命名'}")
        print(f"█  策略: {policy.name} (MASE={policy.avg_mase:.4f})")
        print(f"█  创建轮次: {round_info}")
        print("█" * 80 + "\n")
        if context and 'logger' in context and context['logger']:
            context['logger'].log(
                f"   🆕 强制创建新簇！ID={cluster_id}, 策略={policy.name}, MASE={policy.avg_mase:.4f}",
                level="INFO"
            )
        return cluster_id

    def add_policy(self, policy: SkillPolicy, cluster_id: Optional[str] = None,
                   context: Optional[Dict] = None) -> str:
        """
        添加策略到图，若满足门控条件则创建新簇，并醒目打印。
        返回分配到的簇 ID
        """
        # 如果已指定 cluster_id，直接加入
        if cluster_id:
            cluster = self.get_cluster(cluster_id)
            if cluster:
                if policy.policy_id not in cluster.policies:
                    cluster.policies.append(policy.policy_id)
                    self._update_cluster_stats(cluster)
                policy.cluster_id = cluster_id
                self.updated_at = time.strftime('%Y-%m-%d %H:%M:%S')
                return cluster_id

        # 未指定 cluster_id，检查是否满足新建簇条件
        if context is None:
            context = {}

        # 检查是否是复活策略（强制建簇）
        if policy.metadata.get('revived', False):
            cluster_id = self.force_create_cluster(policy, context)
            policy.cluster_id = cluster_id
            self.updated_at = time.strftime('%Y-%m-%d %H:%M:%S')
            return cluster_id

        # 正常门控判断
        if self.should_create_new_cluster(policy, context):
            cluster_id = self._create_cluster_for_stat(policy)
            # 醒目打印
            round_info = context.get('current_round', 'N/A')
            print("\n" + "█" * 80)
            print(f"█  🆕 创建新簇！")
            print(f"█  ID: {cluster_id}")
            print(f"█  场景标签: {policy.semantic_description or '未命名'}")
            print(f"█  策略: {policy.name} (MASE={policy.avg_mase:.4f})")
            print(f"█  创建轮次: {round_info}")
            print("█" * 80 + "\n")
            if 'logger' in context and context['logger']:
                context['logger'].log(
                    f"   🆕 创建新簇！ID={cluster_id}, 策略={policy.name}, MASE={policy.avg_mase:.4f}",
                    level="INFO"
                )
            policy.cluster_id = cluster_id
            self.updated_at = time.strftime('%Y-%m-%d %H:%M:%S')
            return cluster_id

        # 尝试找到最近的簇
        cluster_id = self._find_cluster_for_stat(policy)
        if cluster_id:
            cluster = self.get_cluster(cluster_id)
            if cluster:
                if policy.policy_id not in cluster.policies:
                    cluster.policies.append(policy.policy_id)
                    self._update_cluster_stats(cluster)
                policy.cluster_id = cluster_id
                self.updated_at = time.strftime('%Y-%m-%d %H:%M:%S')
                return cluster_id

        # 未分配到任何簇，放入未分配列表
        if policy.policy_id not in self.unassigned:
            self.unassigned.append(policy.policy_id)
        policy.cluster_id = None
        self.updated_at = time.strftime('%Y-%m-%d %H:%M:%S')
        return None

    def _find_cluster_for_stat(self, policy: SkillPolicy) -> Optional[str]:
        """仅用于统计目的，找到最近的簇"""
        if not self.clusters:
            return None

        policy_vec = np.array(policy.embedding[:8]) if policy.embedding else np.zeros(8)

        distances = []
        for cluster in self.clusters:
            if not cluster.is_active:
                continue
            centroid_vec = np.array(cluster.centroid[:8]) if cluster.centroid else np.zeros(8)
            dist = np.linalg.norm(policy_vec - centroid_vec)
            distances.append((cluster.id, dist))

        if not distances:
            return None

        closest_id, closest_dist = min(distances, key=lambda x: x[1])
        if closest_dist < self.CLUSTER_DISTANCE_THRESHOLD:
            return closest_id
        return None

    def _create_cluster_for_stat(self, policy: SkillPolicy) -> str:
        """创建新簇"""
        cluster_id = f"cluster_{len(self.clusters) + 1}_{hashlib.md5(str(time.time()).encode()).hexdigest()[:4]}"

        groups = policy.feature_groups or ['general']
        scene_label = '+'.join(groups) if len(groups) <= 2 else f"{groups[0]}+{groups[1]}"
        centroid = policy.embedding[:8] if policy.embedding else [0.0] * 8

        new_cluster = PolicyCluster(
            id=cluster_id,
            centroid=centroid,
            scene_label=scene_label,
            policies=[policy.policy_id],
            avg_mase=policy.avg_mase,
            creation_round=0,
            is_active=True,
            last_updated=time.strftime('%Y-%m-%d %H:%M:%S')
        )
        self.clusters.append(new_cluster)
        return cluster_id

    def _update_cluster_stats(self, cluster: PolicyCluster):
        """更新簇统计信息（仅用于统计）"""
        pass

    # ---------- 以下保持原有方法不变 ----------
    def get_all_policy_ids(self) -> List[str]:
        all_ids = []
        for cluster in self.clusters:
            all_ids.extend(cluster.policies)
        all_ids.extend(self.unassigned)
        return all_ids

    def get_cluster(self, cluster_id: str) -> Optional[PolicyCluster]:
        for c in self.clusters:
            if c.id == cluster_id:
                return c
        return None

    def get_cluster_policies(self, cluster_id: str) -> List[str]:
        cluster = self.get_cluster(cluster_id)
        return cluster.policies if cluster else []

    def get_policy_cluster_id(self, policy_id: str) -> Optional[str]:
        for cluster in self.clusters:
            if policy_id in cluster.policies:
                return cluster.id
        if policy_id in self.unassigned:
            return None
        return None

    def remove_policy(self, policy_id: str):
        for cluster in self.clusters:
            if policy_id in cluster.policies:
                cluster.policies.remove(policy_id)
                self._update_cluster_stats(cluster)
                return
        if policy_id in self.unassigned:
            self.unassigned.remove(policy_id)

    def get_policies_in_cluster(self, cluster_id: str, policy_dict: Dict[str, SkillPolicy]) -> List[SkillPolicy]:
        cluster = self.get_cluster(cluster_id)
        if not cluster:
            return []
        return [policy_dict[pid] for pid in cluster.policies if pid in policy_dict]

    def to_replay_memory(self) -> List[Dict]:
        return self._experience_history.copy()

    def store_experience(self, experience: Dict):
        self._experience_history.append(experience)
        if len(self._experience_history) > self._max_experience:
            self._experience_history.pop(0)

    def get_experience_stats(self) -> Dict:
        return {
            'total_experiences': len(self._experience_history),
            'max_experiences': self._max_experience,
            'clusters': len(self.clusters),
            'active_clusters': sum(1 for c in self.clusters if c.is_active),
            'total_policies': sum(len(c.policies) for c in self.clusters) + len(self.unassigned)
        }

    def to_dict(self) -> Dict:
        return {
            'clusters': [c.to_dict() for c in self.clusters],
            'unassigned': self.unassigned,
            'version': self.version,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'experience_history': self._experience_history[-100:],
            'max_experience': self._max_experience
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'PolicyGraph':
        clusters = [PolicyCluster.from_dict(c) for c in data.get('clusters', [])]
        graph = cls(
            clusters=clusters,
            unassigned=data.get('unassigned', []),
            version=data.get('version', 1),
            created_at=data.get('created_at', ''),
            updated_at=data.get('updated_at', '')
        )
        graph._experience_history = data.get('experience_history', [])
        graph._max_experience = data.get('max_experience', 1000)
        return graph

    @classmethod
    def from_policies(cls, policies: List[SkillPolicy], config: Optional[Dict] = None) -> 'PolicyGraph':
        graph = cls()
        if not policies:
            return graph

        groups = {}
        for p in policies:
            key = tuple(sorted(p.feature_groups or ['general']))
            if key not in groups:
                groups[key] = []
            groups[key].append(p)

        for group_key, group_policies in groups.items():
            embeds = [p.embedding for p in group_policies if p.embedding]
            centroid = np.mean(embeds, axis=0).tolist() if embeds else [0.0] * 8
            scene_label = '+'.join(group_key)

            cluster = PolicyCluster(
                id=f"cluster_{len(graph.clusters) + 1}",
                centroid=centroid[:8],
                scene_label=scene_label,
                policies=[p.policy_id for p in group_policies],
                avg_mase=np.mean([p.avg_mase for p in group_policies]),
                creation_round=0,
                is_active=True,
                last_updated=time.strftime('%Y-%m-%d %H:%M:%S')
            )
            graph.clusters.append(cluster)

        graph.created_at = time.strftime('%Y-%m-%d %H:%M:%S')
        graph.updated_at = graph.created_at

        return graph

    def get_summary(self) -> Dict:
        total_policies = sum(len(c.policies) for c in self.clusters) + len(self.unassigned)
        active_clusters = sum(1 for c in self.clusters if c.is_active)

        return {
            'total_clusters': len(self.clusters),
            'active_clusters': active_clusters,
            'total_policies': total_policies,
            'unassigned_count': len(self.unassigned),
            'experience_count': len(self._experience_history),
            'clusters': [
                {
                    'id': c.id,
                    'scene_label': c.scene_label,
                    'policy_count': len(c.policies),
                    'avg_mase': c.avg_mase
                }
                for c in self.clusters if c.is_active
            ]
        }

    def get_scene_coverage(self) -> Dict[str, int]:
        coverage = {}
        for cluster in self.clusters:
            if cluster.is_active:
                label = cluster.scene_label or 'general'
                coverage[label] = coverage.get(label, 0) + len(cluster.policies)
        return coverage

    def get_global_summary(self) -> str:
        summary = self.get_summary()
        coverage = self.get_scene_coverage()

        lines = [
            f"全局策略池摘要：",
            f"- 共 {summary['total_policies']} 条策略，分布在 {summary['active_clusters']} 个活跃簇中",
            f"- 未分配策略: {summary['unassigned_count']} 条",
            f"- 经验记录: {summary['experience_count']} 条",
            f"- 场景覆盖: {coverage}",
        ]

        if summary['clusters']:
            best = min(summary['clusters'], key=lambda x: x['avg_mase'])
            lines.append(f"- 最优簇: {best['scene_label']} (MASE={best['avg_mase']:.4f}, 含 {best['policy_count']} 条策略)")

        return '\n'.join(lines)