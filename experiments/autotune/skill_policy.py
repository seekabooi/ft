# experiments/autotune/skill_policy.py
"""
SPLS 唯一核心对象：SkillPolicy - v6 强化学习版
★ 增加 regime_performance 字段（Regime 偏置支持）
★ 增加 update_regime_performance() 方法
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import numpy as np
import hashlib
import time


@dataclass
class SkillPolicy:
    """唯一核心对象：SkillPolicy - v6 强化学习版"""

    # === 标识 ===
    policy_id: str
    name: str
    version: int = 1

    # === Policy Embedding ===
    embedding: List[float] = field(default_factory=list)

    # === 状态条件 ===
    state_condition: Dict[str, Any] = field(default_factory=dict)
    feature_groups: List[str] = field(default_factory=list)

    # === 执行策略 ===
    skill_strategy: Dict[str, Any] = field(default_factory=dict)

    # === 策略状态 ===
    status: str = "ACTIVE"
    trial_start_step: int = 0
    trial_end_step: int = 0

    # === 冷却字段 ===
    last_evolution_step: int = 0
    cooldown_period: int = 200
    in_cooldown: bool = False

    # === Refresh标记 ===
    refresh_count: int = 0
    last_refresh_step: int = 0

    # === 健康指标 ===
    activation_count: int = 0
    coverage_rate: float = 0.0
    win_rate: float = 0.0
    avg_mase: float = 1.0
    error_mean: float = 1.0
    error_std: float = 0.0
    error_trend: float = 0.0
    reward_ema: float = 0.5
    utility_ema: float = 0.5
    rare_score: float = 0.0
    uniqueness: float = 0.0
    marginal_coverage: float = 0.0
    marginal_utility: float = 0.0
    marginal_value: float = 0.0

    # === 置信度 ===
    confidence: float = 0.5
    confidence_lower_bound: float = 0.0
    min_activation_threshold: int = 5

    # === Error Memory ===
    error_history: List[float] = field(default_factory=list)
    state_error_map: Dict[str, float] = field(default_factory=dict)
    dynamic_confidence: float = 0.5
    temperature: float = 1.0
    selection_prob: float = 0.0

    # === Internal State ===
    internal_state: float = 0.0
    state_decay: float = 0.9
    state_history: List[float] = field(default_factory=list)

    # === Dominance ===
    dominance_counter: int = 0
    penalty_factor: float = 1.0

    # === 生命周期 ===
    usage_count: int = 0
    last_used: Optional[str] = None
    created_at: str = ""

    # === 策略间关系 ===
    enhanced_by: List[str] = field(default_factory=list)
    suppresses: List[str] = field(default_factory=list)

    # === 元数据 ===
    metadata: Dict = field(default_factory=dict)

    # === 语义描述 ===
    semantic_description: str = ""

    # === 簇归属 ===
    cluster_id: Optional[str] = None

    # ★★★ RL 相关字段 ★★★
    logit_weight: float = 0.0
    cumulative_reward: float = 0.0
    selection_count: int = 0

    # ★★★ 稳定性相关 ★★★
    _error_volatility: float = 0.0
    _error_history_window: int = 10

    # ★★★ Regime 性能记录（新增） ★★★
    regime_performance: Dict[str, float] = field(default_factory=dict)  # regime_key -> avg_mase

    def __post_init__(self):
        self._sync_utility()

    def _sync_utility(self):
        if self.reward_ema > 0:
            self.utility_ema = max(0.01, min(0.99, self.reward_ema))
        elif self.avg_mase > 0 and self.avg_mase < 10:
            self.reward_ema = 1.0 / (self.avg_mase + 0.01)
            self.utility_ema = max(0.01, min(0.99, self.reward_ema))
        elif self.error_mean > 0 and self.error_mean < 10:
            self.reward_ema = 1.0 / (self.error_mean + 0.01)
            self.utility_ema = max(0.01, min(0.99, self.reward_ema))

    # ==================== 核心方法 ====================

    def compute_applicability_score(self, state: Dict[str, float]) -> float:
        """软条件：计算适用性分数"""
        if not self.feature_groups:
            return 0.5

        if self.embedding:
            state_vector = []
            for group in self.feature_groups:
                state_vector.append(state.get(group, 0.0))

            if state_vector:
                state_vec = np.array(state_vector)
                policy_vec = np.array(self.embedding[:len(state_vector)])

                if len(policy_vec) < len(state_vec):
                    policy_vec = np.pad(policy_vec, (0, len(state_vec) - len(policy_vec)))
                elif len(policy_vec) > len(state_vec):
                    state_vec = np.pad(state_vec, (0, len(policy_vec) - len(state_vec)))

                norm_s = np.linalg.norm(state_vec)
                norm_p = np.linalg.norm(policy_vec)
                if norm_s > 0 and norm_p > 0:
                    cos_sim = np.dot(state_vec, policy_vec) / (norm_s * norm_p)
                    score = (cos_sim + 1.0) / 2.0
                    hard_match = self._hard_condition_match(state)
                    return 0.7 * score + 0.3 * hard_match

        return self._hard_condition_match(state)

    def _hard_condition_match(self, state: Dict[str, float]) -> float:
        if not self.state_condition:
            return 1.0

        matches = 0
        total = len(self.state_condition)

        for key, condition in self.state_condition.items():
            value = state.get(key, 0)
            if self._eval_condition(value, condition):
                matches += 1

        return matches / total if total > 0 else 0.5

    def _eval_condition(self, value: float, condition: Any) -> bool:
        if isinstance(condition, str):
            condition = condition.strip()
            if condition.startswith(">="):
                return value >= float(condition[2:])
            elif condition.startswith("<="):
                return value <= float(condition[2:])
            elif condition.startswith("=="):
                return value == float(condition[2:])
            elif condition.startswith("!="):
                return value != float(condition[2:])
            elif condition.startswith(">"):
                return value > float(condition[1:])
            elif condition.startswith("<"):
                return value < float(condition[1:])
            else:
                try:
                    return value == float(condition)
                except:
                    return True
        elif isinstance(condition, (int, float)):
            return value == condition
        return True

    def is_applicable(self, state: Dict[str, float]) -> bool:
        return self.compute_applicability_score(state) > 0.3

    def execute(self, history: np.ndarray, horizon: int, period: int) -> Optional[np.ndarray]:
        if not self.skill_strategy:
            return None

        stages = self.skill_strategy.get('stages', [])
        if not stages:
            return None

        try:
            from src.skills.registry import SkillRegistry
            from run_benchmark import build_full_registry

            full_registry, _ = build_full_registry()

            total_steps = sum(s.get('steps', 0) for s in stages)
            if total_steps != horizon:
                last_stage = stages[-1]
                diff = horizon - total_steps
                if diff > 0:
                    last_stage['steps'] = last_stage.get('steps', 0) + diff

            predictions = []
            current_hist = history.copy()

            self._update_internal_state(history[-1] if len(history) > 0 else 0)

            for stage in stages:
                steps = stage.get('steps', 0)
                weights = stage.get('weights', {})

                for _ in range(steps):
                    pred_val = 0.0
                    total_w = 0.0
                    for skill_name, weight in weights.items():
                        skill = full_registry.get(skill_name)
                        if skill and weight > 0:
                            try:
                                forecast = skill.execute(current_hist, 1, period=period)
                                if forecast is not None and len(forecast) > 0:
                                    pred_val += forecast[0] * weight
                                    total_w += weight
                            except:
                                pass
                    if total_w > 0:
                        pred_val /= total_w
                    else:
                        pred_val = np.mean(current_hist[-5:]) if len(current_hist) >= 5 else np.mean(current_hist)

                    predictions.append(pred_val)
                    current_hist = np.append(current_hist, pred_val)

            return np.array(predictions[:horizon])
        except Exception:
            return None

    def _update_internal_state(self, value: float):
        self.internal_state = self.state_decay * self.internal_state + (1 - self.state_decay) * value
        self.state_history.append(self.internal_state)
        if len(self.state_history) > 50:
            self.state_history.pop(0)

    # ★★★ 修改：record_error 增加 regime_key 参数 ★★★
    def record_error(self, error: float, state_key: str = None, regime_key: str = None):
        self.error_history.append(error)
        if len(self.error_history) > 20:
            self.error_history.pop(0)

        if state_key:
            if state_key not in self.state_error_map:
                self.state_error_map[state_key] = error
            else:
                self.state_error_map[state_key] = 0.8 * self.state_error_map[state_key] + 0.2 * error

        # ★★★ 更新 Regime 性能 ★★★
        if regime_key:
            if regime_key not in self.regime_performance:
                self.regime_performance[regime_key] = error
            else:
                self.regime_performance[regime_key] = 0.9 * self.regime_performance[regime_key] + 0.1 * error

        recent_errors = self.error_history[-10:] if len(self.error_history) >= 10 else self.error_history
        if recent_errors:
            avg_recent_error = np.mean(recent_errors)
            self.dynamic_confidence = np.exp(-avg_recent_error / 0.5)
            self.dynamic_confidence = max(0.1, min(0.95, self.dynamic_confidence))

        self.error_mean = np.mean(self.error_history[-10:]) if len(self.error_history) >= 10 else self.error_mean
        if len(self.error_history) >= 5:
            self.error_std = np.std(self.error_history[-10:])
            if len(self.error_history) >= 10:
                self.error_trend = (self.error_history[-1] - self.error_history[-5]) / (self.error_history[-5] + 0.01)

        if len(self.error_history) >= self._error_history_window:
            window = self.error_history[-self._error_history_window:]
            self._error_volatility = min(1.0, np.std(window) / (abs(np.mean(window)) + 0.01))
        else:
            self._error_volatility = 0.0

        if self.error_history:
            self.avg_mase = np.mean(self.error_history[-5:]) if len(self.error_history) >= 5 else np.mean(
                self.error_history)

    def update_reward(self, reward: float):
        if self.reward_ema == 0.0:
            self.reward_ema = reward
        else:
            self.reward_ema = 0.9 * self.reward_ema + 0.1 * reward

        self.utility_ema = max(0.01, min(0.99, self.reward_ema))

    # ★★★ 新增：更新 Regime 性能（可单独调用） ★★★
    def update_regime_performance(self, regime_key: str, mase: float):
        if regime_key not in self.regime_performance:
            self.regime_performance[regime_key] = mase
        else:
            self.regime_performance[regime_key] = 0.9 * self.regime_performance[regime_key] + 0.1 * mase

    # ★★★ 获取 Regime 偏置（用于 logit 注入） ★★★
    def get_regime_bonus(self, regime_key: str, alpha: float = 0.3) -> float:
        if regime_key in self.regime_performance:
            avg_mase = self.regime_performance[regime_key]
            # 如果该 regime 下历史 MASE 较低，则给正 bonus
            return alpha * max(0, 1.0 - avg_mase)  # avg_mase 越低，bonus 越高
        return 0.0

    def get_stability_score(self) -> float:
        if len(self.error_history) < 5:
            return 0.6
        long_term = self.reward_ema
        short_term = 1.0 - self._error_volatility
        stability = 0.6 * long_term + 0.4 * short_term
        return max(0.0, min(1.0, stability))

    def update_rl_stats(self, reward: float):
        self.selection_count += 1
        self.cumulative_reward += reward

    def record_activation(self):
        self.activation_count += 1

    def record_win(self, is_win: bool):
        if is_win:
            self.win_rate = (self.win_rate * (self.activation_count - 1) + 1.0) / max(1, self.activation_count)
        else:
            self.win_rate = (self.win_rate * (self.activation_count - 1)) / max(1, self.activation_count)

    def update_coverage(self, coverage: float):
        self.coverage_rate = 0.9 * self.coverage_rate + 0.1 * coverage

    def update_rare_score(self, peak_reward: float, decay_factor: float, recent_tail_win_rate: float):
        self.rare_score = peak_reward * decay_factor * recent_tail_win_rate

    def update_confidence(self, wilson_score: float, lower_bound: float):
        self.confidence = wilson_score
        self.confidence_lower_bound = lower_bound

    def is_evolution_ready(self, current_step: int) -> bool:
        if not self.in_cooldown:
            return True
        return (current_step - self.last_evolution_step) > self.cooldown_period

    def start_cooldown(self, current_step: int):
        self.last_evolution_step = current_step
        self.in_cooldown = True

    def refresh(self, new_embedding: List[float], new_condition: Dict, current_step: int):
        self.embedding = new_embedding
        self.state_condition = new_condition
        self.refresh_count += 1
        self.last_refresh_step = current_step
        self.start_cooldown(current_step)

    def _to_serializable(self, value):
        if isinstance(value, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(value)
        elif isinstance(value, (np.floating, np.float64, np.float32, np.float16)):
            return float(value)
        elif isinstance(value, np.ndarray):
            return value.tolist()
        elif isinstance(value, dict):
            return {k: self._to_serializable(v) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            return [self._to_serializable(v) for v in value]
        else:
            return value

    def to_dict(self) -> Dict:
        data = {
            'policy_id': self.policy_id,
            'name': self.name,
            'version': self.version,
            'embedding': self.embedding,
            'state_condition': self.state_condition,
            'feature_groups': self.feature_groups,
            'skill_strategy': self.skill_strategy,
            'status': self.status,
            'trial_start_step': self.trial_start_step,
            'trial_end_step': self.trial_end_step,
            'last_evolution_step': self.last_evolution_step,
            'cooldown_period': self.cooldown_period,
            'in_cooldown': self.in_cooldown,
            'refresh_count': self.refresh_count,
            'last_refresh_step': self.last_refresh_step,
            'activation_count': self.activation_count,
            'coverage_rate': self.coverage_rate,
            'win_rate': self.win_rate,
            'avg_mase': self.avg_mase,
            'error_mean': self.error_mean,
            'error_std': self.error_std,
            'error_trend': self.error_trend,
            'reward_ema': self.reward_ema,
            'utility_ema': self.utility_ema,
            'rare_score': self.rare_score,
            'uniqueness': self.uniqueness,
            'marginal_coverage': self.marginal_coverage,
            'marginal_utility': self.marginal_utility,
            'marginal_value': self.marginal_value,
            'confidence': self.confidence,
            'confidence_lower_bound': self.confidence_lower_bound,
            'error_history': self.error_history,
            'state_error_map': self.state_error_map,
            'dynamic_confidence': self.dynamic_confidence,
            'temperature': self.temperature,
            'selection_prob': self.selection_prob,
            'internal_state': self.internal_state,
            'state_decay': self.state_decay,
            'state_history': self.state_history,
            'dominance_counter': self.dominance_counter,
            'penalty_factor': self.penalty_factor,
            'usage_count': self.usage_count,
            'last_used': self.last_used,
            'created_at': self.created_at,
            'enhanced_by': self.enhanced_by,
            'suppresses': self.suppresses,
            'metadata': self.metadata,
            'semantic_description': self.semantic_description,
            'cluster_id': self.cluster_id,
            'logit_weight': self.logit_weight,
            'cumulative_reward': self.cumulative_reward,
            'selection_count': self.selection_count,
            '_error_volatility': self._error_volatility,
            'regime_performance': self.regime_performance,   # ★★★ 新字段
        }
        return self._to_serializable(data)

    @classmethod
    def from_dict(cls, data: Dict) -> 'SkillPolicy':
        policy = cls(
            policy_id=data.get('policy_id', ''),
            name=data.get('name', ''),
            version=data.get('version', 1),
            embedding=data.get('embedding', []),
            state_condition=data.get('state_condition', {}),
            feature_groups=data.get('feature_groups', []),
            skill_strategy=data.get('skill_strategy', {}),
            status=data.get('status', 'ACTIVE'),
            trial_start_step=data.get('trial_start_step', 0),
            trial_end_step=data.get('trial_end_step', 0),
            last_evolution_step=data.get('last_evolution_step', 0),
            cooldown_period=data.get('cooldown_period', 200),
            in_cooldown=data.get('in_cooldown', False),
            refresh_count=data.get('refresh_count', 0),
            last_refresh_step=data.get('last_refresh_step', 0),
            activation_count=data.get('activation_count', 0),
            coverage_rate=data.get('coverage_rate', 0.0),
            win_rate=data.get('win_rate', 0.0),
            avg_mase=data.get('avg_mase', 1.0),
            error_mean=data.get('error_mean', 1.0),
            error_std=data.get('error_std', 0.0),
            error_trend=data.get('error_trend', 0.0),
            reward_ema=data.get('reward_ema', 0.5),
            utility_ema=data.get('utility_ema', 0.5),
            rare_score=data.get('rare_score', 0.0),
            uniqueness=data.get('uniqueness', 0.0),
            marginal_coverage=data.get('marginal_coverage', 0.0),
            marginal_utility=data.get('marginal_utility', 0.0),
            marginal_value=data.get('marginal_value', 0.0),
            confidence=data.get('confidence', 0.5),
            confidence_lower_bound=data.get('confidence_lower_bound', 0.0),
            error_history=data.get('error_history', []),
            state_error_map=data.get('state_error_map', {}),
            dynamic_confidence=data.get('dynamic_confidence', 0.5),
            temperature=data.get('temperature', 1.0),
            selection_prob=data.get('selection_prob', 0.0),
            internal_state=data.get('internal_state', 0.0),
            state_decay=data.get('state_decay', 0.9),
            state_history=data.get('state_history', []),
            dominance_counter=data.get('dominance_counter', 0),
            penalty_factor=data.get('penalty_factor', 1.0),
            usage_count=data.get('usage_count', 0),
            last_used=data.get('last_used'),
            created_at=data.get('created_at', ''),
            enhanced_by=data.get('enhanced_by', []),
            suppresses=data.get('suppresses', []),
            metadata=data.get('metadata', {}),
            semantic_description=data.get('semantic_description', ''),
            cluster_id=data.get('cluster_id', None),
            logit_weight=data.get('logit_weight', 0.0),
            cumulative_reward=data.get('cumulative_reward', 0.0),
            selection_count=data.get('selection_count', 0),
        )
        policy._error_volatility = data.get('_error_volatility', 0.0)
        policy.regime_performance = data.get('regime_performance', {})   # ★★★ 兼容旧 checkpoint
        return policy

    def get_summary(self) -> Dict:
        return {
            'policy_id': self.policy_id,
            'name': self.name,
            'status': self.status,
            'utility_ema': self.utility_ema,
            'avg_mase': self.avg_mase,
            'coverage_rate': self.coverage_rate,
            'win_rate': self.win_rate,
            'confidence': self.confidence,
            'activation_count': self.activation_count,
            'in_cooldown': self.in_cooldown,
            'feature_groups': self.feature_groups,
            'marginal_value': self.marginal_value,
            'condition_hint': self._get_condition_hint(),
            'semantic_description': self.semantic_description,
            'cluster_id': self.cluster_id,
            'logit_weight': self.logit_weight,
            'selection_count': self.selection_count,
            'cumulative_reward': self.cumulative_reward,
            'stability_score': self.get_stability_score(),
            'error_volatility': self._error_volatility,
            'regime_performance': self.regime_performance,
        }

    def _get_condition_hint(self) -> str:
        if self.feature_groups:
            return f"groups: {', '.join(self.feature_groups)}"
        if self.state_condition:
            first_key = next(iter(self.state_condition))
            return f"{first_key}: {self.state_condition[first_key]}"
        return "通用"


def create_policy_from_legacy_rule(rule: Dict, config: Optional[Dict] = None) -> 'SkillPolicy':
    config = config or {}
    min_groups = config.get('policy', {}).get('min_feature_groups', 2)

    strategy = rule.get('skill_strategy', {})
    condition = rule.get('condition', 'True')

    state_condition = {}
    feature_groups = []

    if condition != 'True':
        parts = condition.split(' and ')
        for part in parts:
            part = part.strip()
            for op in ['>=', '<=', '==', '!=', '>', '<']:
                if op in part:
                    k, v = part.split(op, 1)
                    k = k.strip()
                    try:
                        state_condition[k] = f"{op} {float(v.strip())}"
                        feature_groups.append(k)
                    except:
                        state_condition[k] = f"{op} {v.strip()}"
                        feature_groups.append(k)
                    break

    if len(feature_groups) < min_groups:
        extra_features = ['trend_strength', 'seasonal_strength', 'cv']
        for feat in extra_features:
            if feat not in feature_groups:
                feature_groups.append(feat)
                if len(feature_groups) >= min_groups:
                    break

    policy_id = hashlib.md5(f"{condition}_{time.time()}".encode()).hexdigest()[:8]
    embedding_dim = config.get('policy', {}).get('embedding_dim', 8)
    embedding = list(np.random.randn(embedding_dim) * 0.1)

    avg_mase = rule.get('avg_mase', 1.0)
    utility = 1.0 / (avg_mase + 0.01)

    return SkillPolicy(
        policy_id=policy_id,
        name=rule.get('name', f'policy_{policy_id}'),
        embedding=embedding,
        state_condition=state_condition,
        feature_groups=feature_groups,
        skill_strategy=strategy,
        avg_mase=avg_mase,
        error_mean=avg_mase,
        reward_ema=utility,
        utility_ema=utility,
        marginal_value=0.0,
        confidence=rule.get('confidence', 0.5),
        created_at=time.strftime('%Y-%m-%d %H:%M:%S'),
        metadata=rule.get('metadata', {}),
        semantic_description=rule.get('semantic_description', ''),
        cluster_id=rule.get('cluster_id', None)
    )