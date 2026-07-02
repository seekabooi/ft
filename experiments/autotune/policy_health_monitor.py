# experiments/autotune/policy_health_monitor.py
"""
Policy Health Monitor（P1 + P11）
7类指标 + Marginal Coverage + Marginal Utility + Confidence
★ 修复：添加 marginal_value 计算，避免 KeyError
"""

import numpy as np
from typing import Dict, List, Optional, Any
from collections import defaultdict


class PolicyHealthMonitor:
    """
    策略健康监控器

    每条规则记录：
        - 使用指标：activation_count, coverage_rate
        - 效果指标：win_rate, error_mean, error_std, error_trend
        - 奖励指标：reward_ema, utility_ema
        - 活跃度指标：last_active_step, inactive_duration
        - 稀有指标：rare_score, uniqueness
        - 组合价值：marginal_coverage, marginal_utility, marginal_value
        - 置信度：confidence, confidence_lower_bound
    """

    def __init__(self, config: Dict, logger):
        self.config = config
        self.logger = logger

        # 配置
        policy_cfg = config.get('policy_pool', {})
        self.target_policies = policy_cfg.get('target_policies', 8)
        self.hard_max = policy_cfg.get('hard_max', 15)

        # 历史记录
        self.history = []
        self._policy_health_cache = {}

    def update_policy_health(self, policy) -> Dict:
        """
        更新单个策略的健康信息

        Returns:
            {
                'utility_ema': float,
                'coverage_rate': float,
                'win_rate': float,
                'stability': float,
                'rare_score': float,
                'uniqueness': float,
                'marginal_coverage': float,
                'marginal_utility': float,
                'marginal_value': float,    # ★ 新增
                'confidence': float,
                'health_score': float,
                'grade': str
            }
        """
        # 1. 计算稳定性（基于误差趋势）
        if len(policy.error_history) >= 5:
            stability = 1.0 / (1.0 + abs(policy.error_trend))
        else:
            stability = 0.5

        # 2. 计算稀有分数（P5）
        rare_score = policy.rare_score

        # 3. 计算唯一性（P5）
        uniqueness = policy.uniqueness

        # 4. 置信度（P14）
        confidence = policy.confidence

        # 5. ★ 计算 marginal_value（组合边际价值）
        marginal_value = 0.0
        if hasattr(policy, 'marginal_coverage') and hasattr(policy, 'marginal_utility'):
            marginal_value = (policy.marginal_coverage + policy.marginal_utility) / 2.0
        elif hasattr(policy, 'marginal_coverage'):
            marginal_value = policy.marginal_coverage
        elif hasattr(policy, 'marginal_utility'):
            marginal_value = policy.marginal_utility

        # 6. 计算健康分数
        health_score = (
            0.35 * policy.utility_ema +
            0.25 * policy.win_rate +
            0.15 * policy.coverage_rate +
            0.15 * stability +
            0.10 * rare_score
        )

        # 7. 评级（P4）
        if health_score >= 0.7:
            grade = 'A'
        elif health_score >= 0.5:
            grade = 'B'
        elif health_score >= 0.3:
            grade = 'C'
        else:
            grade = 'D'

        health_info = {
            'utility_ema': policy.utility_ema,
            'coverage_rate': policy.coverage_rate,
            'win_rate': policy.win_rate,
            'stability': stability,
            'rare_score': rare_score,
            'uniqueness': uniqueness,
            'marginal_coverage': getattr(policy, 'marginal_coverage', 0.0),
            'marginal_utility': getattr(policy, 'marginal_utility', 0.0),
            'marginal_value': marginal_value,  # ★ 新增
            'confidence': confidence,
            'health_score': health_score,
            'grade': grade,
            'activation_count': policy.activation_count,
            'error_mean': policy.error_mean,
            'error_std': policy.error_std,
            'error_trend': policy.error_trend,
            'status': policy.status,
            'in_cooldown': policy.in_cooldown
        }

        self._policy_health_cache[policy.policy_id] = health_info
        return health_info

    def compute_marginal_values(self, policies: List, validation_func) -> Dict:
        """
        计算所有策略的边际贡献（P11）

        Args:
            policies: 策略列表
            validation_func: 验证函数，接受策略列表返回整体分数

        Returns:
            {
                policy_id: {
                    'marginal_coverage': float,
                    'marginal_utility': float,
                    'marginal_value': float
                }
            }
        """
        if not policies:
            return {}

        # 1. 计算整体基线
        baseline = validation_func(policies)

        marginal_values = {}

        for policy in policies:
            # 2. 模拟删除该策略
            reduced_policies = [p for p in policies if p.policy_id != policy.policy_id]

            if not reduced_policies:
                marginal_values[policy.policy_id] = {
                    'marginal_coverage': baseline,
                    'marginal_utility': baseline,
                    'marginal_value': baseline
                }
                continue

            # 3. 计算删除后的分数
            reduced_score = validation_func(reduced_policies)

            # 4. 边际贡献 = 整体 - 删除后
            marginal_coverage = max(0, baseline - reduced_score)
            marginal_utility = max(0, baseline - reduced_score)

            # 5. 组合边际价值
            marginal_value = 0.5 * marginal_coverage + 0.5 * marginal_utility

            marginal_values[policy.policy_id] = {
                'marginal_coverage': marginal_coverage,
                'marginal_utility': marginal_utility,
                'marginal_value': marginal_value
            }

            # 更新策略的边际指标
            policy.marginal_coverage = marginal_coverage
            policy.marginal_utility = marginal_utility

        return marginal_values

    def compute_marginal_values_efficient(self, policies: List, validation_func,
                                          candidates: Optional[List] = None) -> Dict:
        """
        高效计算边际贡献（P15：近似估算→候选精算）

        Args:
            policies: 策略列表
            validation_func: 验证函数
            candidates: 候选策略（只对候选精算）

        Returns:
            marginal_values
        """
        if not policies:
            return {}

        if candidates is None:
            # 如果没有指定候选，使用所有策略
            candidates = policies

        # 1. 整体基线
        baseline = validation_func(policies)

        marginal_values = {}

        for policy in policies:
            # 只对候选策略精算
            if policy not in candidates:
                # 使用近似值：基于历史指标估算
                approx = policy.utility_ema * 0.5 + policy.coverage_rate * 0.3 + policy.rare_score * 0.2
                marginal_values[policy.policy_id] = {
                    'marginal_coverage': approx * 0.5,
                    'marginal_utility': approx * 0.5,
                    'marginal_value': approx,
                    'approximated': True
                }
                continue

            # 精算
            reduced_policies = [p for p in policies if p.policy_id != policy.policy_id]

            if reduced_policies:
                reduced_score = validation_func(reduced_policies)
                marginal_coverage = max(0, baseline - reduced_score)
                marginal_utility = max(0, baseline - reduced_score)
            else:
                marginal_coverage = baseline
                marginal_utility = baseline

            marginal_value = 0.5 * marginal_coverage + 0.5 * marginal_utility

            marginal_values[policy.policy_id] = {
                'marginal_coverage': marginal_coverage,
                'marginal_utility': marginal_utility,
                'marginal_value': marginal_value,
                'approximated': False
            }

            policy.marginal_coverage = marginal_coverage
            policy.marginal_utility = marginal_utility

        return marginal_values

    def get_health_report(self, policies: List) -> Dict:
        """
        生成健康报告
        ★ 使用 .get() 安全访问 marginal_value，避免 KeyError
        """
        report = {
            'total_policies': len(policies),
            'status_distribution': defaultdict(int),
            'grade_distribution': defaultdict(int),
            'average_health_score': 0.0,
            'policies': []
        }

        total_health = 0.0

        for policy in policies:
            health = self.update_policy_health(policy)
            report['status_distribution'][policy.status] += 1
            report['grade_distribution'][health['grade']] += 1
            total_health += health['health_score']

            report['policies'].append({
                'policy_id': policy.policy_id,
                'name': policy.name,
                'status': policy.status,
                'grade': health['grade'],
                'health_score': health['health_score'],
                'utility_ema': health['utility_ema'],
                'coverage_rate': health['coverage_rate'],
                'win_rate': health['win_rate'],
                'rare_score': health['rare_score'],
                'uniqueness': health['uniqueness'],
                'marginal_value': health.get('marginal_value', 0.0),  # ★ 安全访问
                'confidence': health['confidence'],
                'activation_count': health['activation_count'],
                'in_cooldown': health['in_cooldown']
            })

        if policies:
            report['average_health_score'] = total_health / len(policies)

        return report