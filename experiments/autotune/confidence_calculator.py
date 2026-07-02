# experiments/autotune/confidence_calculator.py
"""
Confidence Calculator（P14）
使用 Wilson Score 计算置信度，作为决策门槛修正项
"""

import numpy as np
from typing import Dict, Optional


class ConfidenceCalculator:
    """
    置信度计算器

    Wilson Score Interval:
        用于计算二项分布比例的置信区间
        适用于 win_rate 等比例指标

    用途：
        - 作为决策门槛修正项
        - 不直接进入 evolution_score
        - 防止小样本导致的误判
    """

    def __init__(self, config: Dict):
        self.config = config
        self.min_activation = config.get('confidence', {}).get('min_activation_threshold', 5)

    def wilson_score(self, successes: int, trials: int, z: float = 1.96) -> Dict:
        """
        计算 Wilson Score

        Args:
            successes: 成功次数
            trials: 总次数
            z: 正态分布分位数（1.96 = 95%置信度）

        Returns:
            {
                'score': 点估计
                'lower_bound': 下限
                'upper_bound': 上限
            }
        """
        if trials == 0:
            return {'score': 0.0, 'lower_bound': 0.0, 'upper_bound': 0.0}

        p = successes / trials
        z2 = z * z

        # Wilson Score 公式
        numerator = p + z2 / (2 * trials)
        denominator = 1 + z2 / trials
        margin = z * np.sqrt((p * (1 - p) + z2 / (4 * trials)) / trials)

        lower = max(0, (numerator - margin) / denominator)
        upper = min(1, (numerator + margin) / denominator)

        return {
            'score': p,
            'lower_bound': lower,
            'upper_bound': upper,
            'trials': trials,
            'successes': successes
        }

    def beta_posterior(self, successes: int, trials: int,
                       alpha_prior: float = 1.0, beta_prior: float = 1.0) -> Dict:
        """
        计算 Beta 后验分布

        Args:
            successes: 成功次数
            trials: 总次数
            alpha_prior: 先验 Alpha
            beta_prior: 先验 Beta

        Returns:
            {
                'alpha': 后验 Alpha,
                'beta': 后验 Beta,
                'mean': 后验均值,
                'lower_credible': 5% 分位数,
                'upper_credible': 95% 分位数
            }
        """
        alpha_post = alpha_prior + successes
        beta_post = beta_prior + (trials - successes)

        # 均值
        mean = alpha_post / (alpha_post + beta_post)

        # 分位数
        try:
            from scipy.stats import beta
            lower_credible = beta.ppf(0.05, alpha_post, beta_post)
            upper_credible = beta.ppf(0.95, alpha_post, beta_post)
        except:
            lower_credible = mean - 0.1
            upper_credible = mean + 0.1

        return {
            'alpha': alpha_post,
            'beta': beta_post,
            'mean': mean,
            'lower_credible': max(0, lower_credible),
            'upper_credible': min(1, upper_credible)
        }

    def compute_policy_confidence(self, policy) -> Dict:
        """
        计算策略的置信度

        Args:
            policy: SkillPolicy 实例

        Returns:
            {
                'wilson_score': Wilson 点估计,
                'wilson_lower': Wilson 下限,
                'beta_mean': Beta 后验均值,
                'adjusted_confidence': 调整后的置信度,
                'is_reliable': 是否可靠（trials >= min_activation）
            }
        """
        successes = int(policy.win_rate * policy.activation_count)
        trials = policy.activation_count

        # Wilson Score
        wilson = self.wilson_score(successes, trials)

        # Beta Posterior
        beta = self.beta_posterior(successes, trials)

        # 调整置信度：使用 Wilson lower bound 作为保守估计
        adjusted = wilson['lower_bound']

        is_reliable = trials >= self.min_activation

        return {
            'wilson_score': wilson['score'],
            'wilson_lower': wilson['lower_bound'],
            'wilson_upper': wilson['upper_bound'],
            'beta_mean': beta['mean'],
            'beta_lower': beta['lower_credible'],
            'beta_upper': beta['upper_credible'],
            'adjusted_confidence': adjusted,
            'trials': trials,
            'successes': successes,
            'is_reliable': is_reliable
        }

    def get_decision_threshold_modifier(self, policy) -> float:
        """
        获取决策门槛修正值（P14）

        用途：作为决策门槛的修正项
        不直接进入 evolution_score

        Returns:
            modifier: 0.0 ~ 1.0
                - 高置信度 → modifier = 1.0（正常决策）
                - 低置信度 → modifier < 1.0（保守决策）
        """
        conf = self.compute_policy_confidence(policy)

        if not conf['is_reliable']:
            # 样本不足，保守处理
            return max(0.1, conf['wilson_lower'] * 0.5)

        # 使用 Wilson lower bound 作为置信度
        # 映射到 0.3~1.0 范围
        base = conf['wilson_lower']
        return max(0.3, min(1.0, base * 1.2))


def compute_wilson_score_from_history(history: list, threshold: float = 0.5) -> float:
    """
    从历史记录计算 Wilson Score

    Args:
        history: 二值列表（1=成功，0=失败）
        threshold: 成功阈值

    Returns:
        Wilson lower bound
    """
    if not history:
        return 0.0

    successes = sum(1 for v in history if v >= threshold)
    trials = len(history)

    if trials == 0:
        return 0.0

    p = successes / trials
    z = 1.96
    z2 = z * z

    numerator = p + z2 / (2 * trials)
    denominator = 1 + z2 / trials
    margin = z * np.sqrt((p * (1 - p) + z2 / (4 * trials)) / trials)

    lower = max(0, (numerator - margin) / denominator)
    return lower