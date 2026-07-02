# experiments/autotune/cluster.py
"""
Latent Policy Space Partitioning - 稳定性修复版本
增加簇内距离计算等辅助函数
"""

import numpy as np
from typing import Dict, List, Any
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class PolicySpacePartitioner:
    def __init__(self, logger):
        self.logger = logger

    def partition(self, strategies: List[Dict], n_clusters: int = 3) -> List[Dict]:
        if len(strategies) <= 1:
            return [{
                'centroid': strategies[0] if strategies else {},
                'strategies': strategies,
                'avg_mase': strategies[0].get('_mase', 0) if strategies else 0,
                'size': len(strategies)
            }]

        self.logger.log(f"   🔬 Policy Space 分割 (目标分区={n_clusters})")

        all_skills = set()
        for s in strategies:
            for stage in s.get('stages', []):
                for skill in stage.get('weights', {}).keys():
                    all_skills.add(skill)
        all_skills = sorted(list(all_skills))
        skill_to_idx = {skill: i for i, skill in enumerate(all_skills)}

        vectors = []
        for s in strategies:
            skill_vec = np.zeros(len(all_skills))
            stages = s.get('stages', [])
            total_weights = defaultdict(float)
            for stage in stages:
                for skill, weight in stage.get('weights', {}).items():
                    total_weights[skill] += weight
            if total_weights:
                norm = sum(total_weights.values())
                for skill in total_weights:
                    total_weights[skill] /= norm
            for skill, weight in total_weights.items():
                if skill in skill_to_idx:
                    skill_vec[skill_to_idx[skill]] = weight

            features = s.get('_features', {})
            feature_vec = np.array([
                features.get('trend_strength', 0),
                features.get('seasonal_strength', 0),
                features.get('adf_pvalue', 0.5),
                features.get('period', 365) / 365.0,
                features.get('local_slope_120', 0),
                features.get('local_std_ratio_120', 0),
                features.get('cv', 0),
            ])

            vec = np.concatenate([skill_vec, feature_vec])
            vectors.append(vec)

        if not vectors:
            return [{'centroid': strategies[0] if strategies else {}, 'strategies': strategies, 'avg_mase': 0, 'size': len(strategies)}]

        X = np.array(vectors)
        n_samples = len(X)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        actual_clusters = min(n_clusters, n_samples)
        if actual_clusters <= 1:
            return [{'centroid': strategies[0] if strategies else {}, 'strategies': strategies, 'avg_mase': strategies[0].get('_mase', 0) if strategies else 0, 'size': len(strategies)}]

        kmeans = KMeans(n_clusters=actual_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)

        partitions = defaultdict(list)
        for i, label in enumerate(labels):
            partitions[label].append(strategies[i])

        result = []
        for label, partition_strategies in partitions.items():
            mases = [s.get('_mase', 0) for s in partition_strategies]
            avg_mase = np.mean(mases) if mases else 0

            centroid_idx = np.argmin([abs(s.get('_mase', 0) - avg_mase) for s in partition_strategies])
            centroid = partition_strategies[centroid_idx] if centroid_idx < len(partition_strategies) else partition_strategies[0]

            result.append({
                'centroid': centroid,
                'strategies': partition_strategies,
                'avg_mase': avg_mase,
                'size': len(partition_strategies)
            })

            self.logger.log(f"      分区 {label+1}: {len(partition_strategies)} 个策略, 平均MASE={avg_mase:.4f}")

        result.sort(key=lambda x: x['avg_mase'])
        return result

    # ---------- 新增辅助函数：簇内距离、平均距离 ----------
    @staticmethod
    def compute_cluster_intra_distance(cluster_centroid: List[float], policy_embeddings: List[List[float]]) -> float:
        """
        计算簇内平均距离（基于 embedding 的欧氏距离）
        """
        if not policy_embeddings or len(policy_embeddings) < 2:
            return 0.0
        centroid = np.array(cluster_centroid)
        distances = [np.linalg.norm(np.array(emb) - centroid) for emb in policy_embeddings if emb]
        return float(np.mean(distances)) if distances else 0.0

    @staticmethod
    def compute_min_distance_to_clusters(policy_embedding: List[float],
                                         clusters: List[Dict]) -> float:
        """
        计算策略 embedding 到所有簇中心的最小距离
        clusters: 列表，每个元素包含 'centroid' (embedding list)
        """
        if not policy_embedding or not clusters:
            return float('inf')
        vec = np.array(policy_embedding)
        min_dist = float('inf')
        for cluster in clusters:
            cent = cluster.get('centroid')
            if cent is None:
                continue
            dist = np.linalg.norm(vec - np.array(cent))
            if dist < min_dist:
                min_dist = dist
        return min_dist

    @staticmethod
    def compute_average_intra_distance(clusters: List[Dict]) -> float:
        """
        计算所有簇的平均簇内距离
        clusters: 每个元素包含 'centroid' 和 'policies' (embeddings list)
        """
        if not clusters:
            return 0.0
        total = 0.0
        count = 0
        for cluster in clusters:
            centroid = cluster.get('centroid')
            policies = cluster.get('policies', [])
            if centroid is None or not policies:
                continue
            distances = [np.linalg.norm(np.array(emb) - np.array(centroid)) for emb in policies if emb]
            if distances:
                total += np.mean(distances)
                count += 1
        return total / count if count > 0 else 0.0