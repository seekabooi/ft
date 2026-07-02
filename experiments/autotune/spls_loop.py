# experiments/autotune/spls_loop.py
"""
SPLS 主循环核心 - v6 强化学习版
集成 Policy Gradient + Online Learning
★ 增加预测缓存支持
★ UCB 探索替代随机 ε-greedy
★ Reward/Advantage 裁剪
★ Regime 历史偏置注入
★ 强制休眠"烂策略"
★ ★ ★ 2026-06-28 新策略试用期冻结（方案 B）
★ ★ ★ 2026-06-28 分策略学习率（方案 C）
★ ★ ★ 2026-06-28 当前轮次管理
★ ★ ★ ★ 2026-06-29 软冻结：冻结期策略允许 θ 更新（仅探索样本）
★ ★ ★ ★ ★ 2026-07-XX 增加 TRIAL → ACTIVE 晋升机制（温和晋升）
★ ★ ★ ★ ★ ★ 2026-07-XX 晋升阈值优化：MASE ≤ 0.95，采样 ≥ 5 次（与退休策略配合）
★ ★ ★ ★ ★ ★ ★ 2026-08-XX 复活策略不冻结，立即参与演化
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from collections import deque
import random
import time
import hashlib

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from experiments.autotune.skill_policy import SkillPolicy
from experiments.autotune.state_encoder import StateEncoder
from experiments.autotune.utils import ProgressLogger, compute_mase
from experiments.autotune.policy_distribution import PolicyDistributionModel
from experiments.autotune.rl_components import compute_reward, compute_advantage, BaselineTracker
from experiments.autotune.replay_memory import ReplayMemory


class SPLSLoop:
    def __init__(self, config: Dict, logger: ProgressLogger, branch_loader=None):
        self.config = config
        self.logger = logger
        self.branch_loader = branch_loader
        self.state_encoder = StateEncoder(config)
        self.policies: List[SkillPolicy] = []
        self.history = []
        self.step_counter = 0
        self.total_loss = 0.0
        self.loss_history = []

        # RL 配置
        rl_cfg = config.get('rl', {})
        self.rl_enabled = rl_cfg.get('enabled', True)
        self.learning_rate = rl_cfg.get('learning_rate', 0.01)
        self.temperature = rl_cfg.get('temperature', 1.0)
        self.baseline_decay = rl_cfg.get('baseline_decay', 0.9)
        self.exploration = rl_cfg.get('exploration', 0.1)
        self.top_k_sampling = rl_cfg.get('top_k_sampling', 3)

        # ★★★ Patch 3：θ bias injection 参数 ★★★
        self.theta_bias_alpha = rl_cfg.get('theta_bias_alpha', 0.20)

        # 分布模型
        theta_decay = rl_cfg.get('theta_decay', 0.01)
        self.distribution_model = PolicyDistributionModel(
            learning_rate=self.learning_rate,
            temperature=self.temperature,
            theta_decay=theta_decay
        )
        self.baseline_tracker = BaselineTracker(initial=0.0, decay=self.baseline_decay)
        self.replay_memory = ReplayMemory(max_size=rl_cfg.get('replay_memory', {}).get('max_size', 1000))

        # 原有 soft mixture 配置（降级为后备）
        sm_cfg = config.get('soft_mixture', {})
        self.use_soft_mixture = sm_cfg.get('enabled', True) and not self.rl_enabled
        self.top_k = sm_cfg.get('top_k', 4)
        self.temperature_old = sm_cfg.get('temperature', 1.0)
        self.entropy_min = sm_cfg.get('entropy_min', 0.1)
        self.entropy_collapse_threshold = sm_cfg.get('entropy_collapse_threshold', 0.05)

        dom_cfg = sm_cfg.get('dominance', {})
        self.dominance_enabled = dom_cfg.get('enabled', True)
        self.dominance_threshold = dom_cfg.get('consecutive_threshold', 3)
        self.dominance_penalty_factor = dom_cfg.get('penalty_factor', 0.7)

        # reward 权重
        reward_cfg = config.get('reward', {})
        self.reward_mase_weight = reward_cfg.get('mase_weight', 1.0)
        self.reward_stability_weight = reward_cfg.get('stability_penalty_weight', 0.1)
        self.reward_consistency_weight = reward_cfg.get('consistency_bonus_weight', 0.05)
        self.reward_ema_decay = reward_cfg.get('ema_decay', 0.9)

        evo_cfg = config.get('continuous_evolution', {})
        self.ema_decay = evo_cfg.get('ema_decay', 0.95)
        self.learning_rate_old = evo_cfg.get('learning_rate', 0.005)

        self._last_retrieved_policy = None
        self._last_entropy = 1.0
        self._last_reward = 0.0
        self._dominance_history = deque(maxlen=20)
        self._dominance_counts = {}
        self.shared_reward = 0.0
        self.reward_view = {}

        # ★★★ RL 阶段统计 ★★★
        self._rl_step_count = 0
        self._rl_total_reward = 0.0
        self._rl_avg_mase = 0.0
        self._rl_mase_history = []

        # ★★★ 预测缓存 ★★★
        self._cache = {}  # key: (policy_id, window_data_path) -> {'pred': array, 'mase': float}

        # ★★★ ★★★ ★★★ 当前轮次（用于试用期冻结） ★★★ ★★★ ★★★
        self.current_round = 0

        # ★★★ ★★★ ★★★ TRIAL → ACTIVE 晋升阈值（温和晋升，与退休策略配合） ★★★ ★★★ ★★★
        self.trial_promotion_threshold = 0.95   # avg_mase <= 0.95 时晋升（比退休阈值 0.4 高，形成安全缓冲）
        self.trial_min_activations = 5          # 至少被采样 5 次才晋升（统计可靠性）

    # ★★★ 设置缓存 ★★★
    def set_cache(self, cache_dict: Dict):
        """设置预测缓存，key为(policy_id, window_data_path)"""
        self._cache = cache_dict

    def get_cache(self, policy_id: str, window_data_path: str) -> Optional[Dict]:
        key = (policy_id, window_data_path)
        return self._cache.get(key, None)

    # ★★★ ★★★ ★★★ 设置当前轮次（用于试用期冻结） ★★★ ★★★ ★★★
    def set_current_round(self, round_num: int):
        self.current_round = round_num

    # ★★★ ★★★ ★★★ 判断策略是否处于冻结期 ★★★ ★★★ ★★★
    def _is_policy_frozen(self, policy: SkillPolicy) -> bool:
        """判断策略是否处于试用期冻结状态"""
        if policy.status != 'TRIAL':
            return False
        # ★★★ 复活策略不冻结 ★★★
        if policy.metadata.get('revived', False):
            return False
        trial_start = policy.metadata.get('trial_start_round', 0)
        trial_freeze = policy.metadata.get('trial_freeze_rounds', 2)
        return self.current_round - trial_start < trial_freeze

    # ★★★ ★★★ ★★★ TRIAL → ACTIVE 晋升检查（温和晋升） ★★★ ★★★ ★★★
    def _promote_trial_to_active(self, policy: SkillPolicy) -> bool:
        """
        检查 TRIAL 策略是否应该晋升为 ACTIVE
        条件：冻结期结束 + avg_mase <= 0.95 + 采样次数 >= 5
        """
        if policy.status != 'TRIAL':
            return False

        # 检查冻结期是否结束
        trial_start = policy.metadata.get('trial_start_round', 0)
        trial_freeze = policy.metadata.get('trial_freeze_rounds', 2)
        is_frozen = self.current_round - trial_start < trial_freeze
        if is_frozen:
            return False

        # 检查采样次数（至少被采样 5 次才有足够数据判断）
        if policy.selection_count < self.trial_min_activations:
            return False

        # 检查 MASE 是否达标
        if policy.avg_mase <= self.trial_promotion_threshold:
            return True

        return False

    def _promote_trial_policies(self):
        """晋升所有符合条件的 TRIAL 策略为 ACTIVE（温和晋升，每轮不限数量）"""
        promoted = []
        for policy in self.policies:
            if self._promote_trial_to_active(policy):
                policy.status = 'ACTIVE'
                promoted.append(policy.name)
                self.logger.log(f"      ⭐ 晋升 TRIAL → ACTIVE: {policy.name} "
                               f"(MASE={policy.avg_mase:.4f}, selection_count={policy.selection_count})")
        return promoted

    # ==================== 核心 step 方法 ====================
    def step(self, observation: np.ndarray, horizon: int, window_data_path: str = None) -> Dict:
        """执行一步预测，如果提供了window_data_path且缓存命中，则跳过执行"""
        self.step_counter += 1

        state = self.state_encoder.encode(observation)
        temperature = state.get('recommended_temperature', 1.0)

        # 选择策略
        if self.rl_enabled and len(self.policies) > 1:
            # RL 分支：选择策略（可能带探索）
            sampled_policy, rl_info = self._rl_select_policy(state, observation)
            self._last_entropy = rl_info.get('entropy', 1.0)
            policy = sampled_policy
        elif self.use_soft_mixture and len(self.policies) > 1:
            # Soft Mixture 后备
            prediction, mixture_info = self._soft_mixture_predict(observation, horizon, state, temperature)
            self._last_entropy = mixture_info.get('entropy', 1.0) if mixture_info else 1.0
            return {
                'step': self.step_counter,
                'state': state,
                'prediction': prediction,
                'period': self._estimate_period(observation),
                'temperature_used': temperature,
                'policy': None,
                'entropy': self._last_entropy,
                'mixture_info': mixture_info
            }
        else:
            # Fallback
            policy = self._retrieve_policy(state)
            if policy is None:
                policy = self._get_default_policy()

        # 检查缓存
        pred = None
        mase = None
        if window_data_path is not None and policy is not None:
            cached = self.get_cache(policy.policy_id, window_data_path)
            if cached is not None:
                pred = cached['pred']
                mase = cached['mase']
                self.logger.log(f"   ✅ 缓存命中: policy={policy.name[:8]}, window={window_data_path[-20:]}, MASE={mase:.4f}")

        if pred is None:
            # 未命中缓存，执行预测
            period = self._estimate_period(observation)
            pred = policy.execute(observation, horizon, period)
            if pred is None:
                pred = np.full(horizon, np.mean(observation[-5:]))

        self._last_retrieved_policy = policy

        return {
            'step': self.step_counter,
            'state': state,
            'prediction': pred,
            'period': self._estimate_period(observation),
            'temperature_used': temperature,
            'policy': policy,
            'entropy': self._last_entropy,
            'mixture_info': None,
            'cached_mase': mase,
        }

    # ★★★ RL 策略选择（含 UCB 探索 + 试用期冻结） ★★★
    def _rl_select_policy(self, state: Dict, observation: np.ndarray) -> Tuple[SkillPolicy, Dict]:
        numeric_state = state.get('numeric', {})
        active_policies = [p for p in self.policies if p.status not in ['ARCHIVE', 'DELETE']]

        # ★★★ ★★★ ★★★ 分离冻结期 TRIAL 策略和可用策略 ★★★ ★★★ ★★★
        frozen_policies = [p for p in active_policies if self._is_policy_frozen(p)]
        exploitable_policies = [p for p in active_policies if not self._is_policy_frozen(p)]

        # 如果没有可用策略（所有策略都冻结），回退到所有策略
        if not exploitable_policies:
            exploitable_policies = active_policies

        # 获取分布（利用分支使用 exploitable_policies）
        dist = self.distribution_model.get_distribution(numeric_state, exploitable_policies)
        policy_ids = list(dist.keys())
        probs = list(dist.values())

        # ★★★ UCB 探索（替代随机 ε-greedy） ★★★
        if np.random.random() < self.exploration:
            # 探索时可以从所有策略中选择（包括冻结期 TRIAL），让新策略有机会被尝试
            all_policies = active_policies
            N = sum(p.selection_count for p in all_policies) + 1e-5
            ucb_scores = []
            for p in all_policies:
                mean_reward = -p.avg_mase if p.avg_mase > 0 else -1.0
                n = p.selection_count + 1e-5
                ucb = mean_reward + 1.0 * np.sqrt(2.0 * np.log(N) / n)
                ucb_scores.append(ucb)
            best_idx = np.argmax(ucb_scores)
            sampled_policy = all_policies[best_idx]
            is_exploration = True
            # 如果探索选中了冻结期策略，记录日志
            if self._is_policy_frozen(sampled_policy):
                self.logger.log(f"   🧭 UCB 探索选中冻结期策略: {sampled_policy.name[:8]} (UCB={ucb_scores[best_idx]:.3f})")
            else:
                self.logger.log(f"   🧭 UCB 探索: {sampled_policy.name[:8]} (UCB={ucb_scores[best_idx]:.3f})")
        else:
            # 按分布采样（利用）- 只从 exploitable_policies 中采样
            sampled_id = np.random.choice(policy_ids, p=probs)
            sampled_policy = next(p for p in exploitable_policies if p.policy_id == sampled_id)
            is_exploration = False

        # 保存当前采样信息供后续更新
        self._last_sampled_policy = sampled_policy
        self._last_dist = dist
        self._last_state = numeric_state
        self._last_is_exploration = is_exploration

        entropy = -np.sum(probs * np.log(np.array(probs) + 1e-8))
        self._last_entropy = entropy

        # 打印采样信息
        top_3 = sorted(dist.items(), key=lambda x: x[1], reverse=True)[:3]
        top_str = ", ".join([f"{pid[:6]}:{p:.3f}" for pid, p in top_3])
        frozen_tag = " (冻结期)" if self._is_policy_frozen(sampled_policy) else ""
        self.logger.log(
            f"   🎯 分布 Top-3: [{top_str}] | 采样: {sampled_policy.policy_id[:8]}{frozen_tag} {'(探索)' if is_exploration else '(利用)'}")

        return sampled_policy, {'entropy': entropy, 'distribution': dist, 'is_exploration': is_exploration}

    # ★★★ step_with_ground_truth（带缓存支持） ★★★
    def step_with_ground_truth(self, observation: np.ndarray, horizon: int,
                               ground_truth: np.ndarray, split: str = 'test',
                               window_data_path: str = None) -> Dict:
        """带 Ground Truth 的步骤，包含 RL 更新和详细日志，支持缓存"""
        self.logger.log(f"   🔍 [step_with_gt] 开始处理窗口, horizon={horizon}, len(gt)={len(ground_truth)}")

        result = self.step(observation, horizon, window_data_path=window_data_path)
        self.logger.log(
            f"   🔍 [step_with_gt] step() 完成, prediction 长度={len(result['prediction']) if result['prediction'] is not None else 'None'}")

        # 如果缓存中有 MASE，直接使用
        cached_mase = result.get('cached_mase')
        if cached_mase is not None:
            mase = cached_mase
            self.logger.log(f"   🔍 [step_with_gt] 使用缓存 MASE={mase:.6f}")
        else:
            # 否则计算 MASE
            if result['prediction'] is not None and len(result['prediction']) == len(ground_truth):
                period = result['period']
                mase_scale = self._compute_mase_scale(observation, period)
                mase = compute_mase(result['prediction'], ground_truth, mase_scale)
                self.logger.log(f"   🔍 [step_with_gt] MASE={mase:.6f}")
            else:
                self.logger.log(f"   ⚠️ 预测长度不匹配，使用 fallback")
                mase = 10.0  # 惩罚值

        result['mase'] = mase
        result['loss'] = mase

        # ★★★ Reward 裁剪（方案 A） ★★★
        reward = self._compute_unified_reward(mase, result)
        reward = max(reward, -10.0)  # 裁剪 Reward 下限
        result['reward'] = reward
        self.shared_reward = reward
        self.reward_view = result.get('reward_detail', {})
        self.logger.log(f"   🔍 [step_with_gt] reward={reward:.4f} (裁剪后)")

        # RL 更新
        if self.rl_enabled and len(self.policies) > 1:
            self.logger.log(f"   🔍 [step_with_gt] 执行 RL 更新...")
            self._rl_update(result, reward, observation, horizon, ground_truth, window_data_path)
            self.logger.log(f"   🔍 [step_with_gt] RL 更新完成")

        # 更新策略统计（包括 Regime）
        self._update_policies_with_reward(mase, reward, result['state'], window_data_path)
        self.logger.log(f"   🔍 [step_with_gt] 策略更新完成")

        self.total_loss += mase
        self.loss_history.append(mase)
        self.history.append({
            'step': self.step_counter,
            'mase': mase,
            'reward': reward,
            'entropy': self._last_entropy,
            'state': result['state']
        })
        self.logger.log(f"   ✅ [step_with_gt] 窗口处理完成, MASE={mase:.6f}")

        return result

    # ★★★ ★★★ ★★★ 软冻结：允许冻结期策略更新 θ（仅探索样本） ★★★ ★★★ ★★★
    def _rl_update(self, result: Dict, reward: float,
                   observation: np.ndarray, horizon: int, ground_truth: np.ndarray,
                   window_data_path: str = None):
        self._rl_step_count += 1
        self.logger.log(f"      🔍 [_rl_update] 开始 RL 更新 #{self._rl_step_count}")

        if not hasattr(self, '_last_sampled_policy'):
            self.logger.log(f"      ⚠️ [_rl_update] 没有 _last_sampled_policy，跳过更新")
            return

        sampled_policy = self._last_sampled_policy
        sampled_id = sampled_policy.policy_id
        is_exploration = getattr(self, '_last_is_exploration', False)

        # ★★★ ★★★ ★★★ 软冻结：允许冻结期策略更新 θ（仅通过探索采样，频率低） ★★★ ★★★ ★★★
        if self._is_policy_frozen(sampled_policy):
            self.logger.log(f"      ❄️ 策略 {sampled_policy.name} 处于冻结期（软冻结），允许 θ 更新（探索样本）")
            # ★ 不 return，继续执行下面的正常更新逻辑 ★

        baseline = self.baseline_tracker.get()
        # ★★★ Advantage 裁剪（方案 A） ★★★
        advantage = compute_advantage(reward, baseline)
        advantage = np.clip(advantage, -10.0, 10.0)
        self.logger.log(f"      🔍 [_rl_update] baseline={baseline:.4f}, advantage={advantage:.4f} (裁剪后)")

        old_baseline = baseline
        self.baseline_tracker.update(reward)
        new_baseline = self.baseline_tracker.get()
        self.logger.log(f"      🔍 [_rl_update] baseline 更新: {old_baseline:.4f} -> {new_baseline:.4f}")

        state = result['state']
        numeric = state.get('numeric', {})
        regime = self.state_encoder.extract_regime(numeric) if hasattr(self.state_encoder, 'extract_regime') else {}
        base_lr = self._get_regime_lr(regime, self.learning_rate)

        # ★★★ ★★★ ★★★ 分策略学习率（方案 C） ★★★ ★★★ ★★★
        if sampled_policy.status == 'TRIAL':
            lr_multiplier = 2.0  # TRIAL 策略高学习率
        else:
            lr_multiplier = 0.5  # ACTIVE 策略低学习率

        effective_lr = base_lr * lr_multiplier
        effective_lr = max(1e-5, min(0.02, effective_lr))  # 限幅
        self.logger.log(f"      🔍 [_rl_update] regime={regime}, base_lr={base_lr:.5f}, lr_multiplier={lr_multiplier:.1f}, effective_lr={effective_lr:.5f}")

        self.logger.log(f"      🔍 [_rl_update] sampled_policy={sampled_id[:8]}, is_exploration={is_exploration}")

        old_theta = self.distribution_model.get_theta(sampled_id)
        self.logger.log(f"      🔍 [_rl_update] old_theta={old_theta:.4f}")

        self.distribution_model.set_learning_rate(effective_lr)
        self.distribution_model.update(sampled_id, advantage, policy=sampled_policy)

        new_theta = self.distribution_model.get_theta(sampled_id)
        self.logger.log(f"      🔍 [_rl_update] new_theta={new_theta:.4f}")

        # ★★★ 强制休眠"烂策略" ★★★
        if sampled_policy.avg_mase > 2.0 and sampled_policy.selection_count > 10:
            self.logger.log(f"      ⚠️ 策略 {sampled_policy.name} 长期表现差 (MASE={sampled_policy.avg_mase:.2f})，强制休眠 (θ=-5.0)")
            self.distribution_model.set_theta(sampled_id, -5.0)
            new_theta = -5.0
            sampled_policy.logit_weight = -5.0

        # 同步 logit_weight
        sampled_policy.logit_weight = new_theta
        sampled_policy.update_rl_stats(reward)

        # 存储经验
        self.replay_memory.store({
            'state': self._last_state,
            'sampled_policy_id': sampled_id,
            'reward': reward,
            'advantage': advantage,
            'timestamp': time.time()
        })
        self.logger.log(f"      🔍 [_rl_update] 经验已存储, replay_memory 大小={len(self.replay_memory)}")

        # 详细日志
        mase = result.get('mase', 0.0)
        entropy = result.get('entropy', self._last_entropy)
        dist = getattr(self, '_last_dist', {})
        top_3 = sorted(dist.items(), key=lambda x: x[1], reverse=True)[:3] if dist else []
        top_str = ", ".join([f"{pid[:6]}:θ={self.distribution_model.get_theta(pid):.3f},p={prob:.3f}"
                             for pid, prob in top_3])
        regime_str = ", ".join([f"{k}={v}" for k, v in regime.items() if v == 1]) if regime else "平稳"

        self.logger.log("")
        self.logger.log(f"   ┌─── 📊 RL Step #{self._rl_step_count} (窗口 {result.get('step', '?')}) ───")
        self.logger.log(f"   │  📌 Regime: {regime_str} | 熵: {entropy:.4f}")
        self.logger.log(f"   │  🎯 分布 Top-3: {top_str}")
        self.logger.log(f"   │  🎲 采样: {sampled_id[:8]} {'(探索)' if is_exploration else '(利用)'} "
                        f"| θ_old={old_theta:.4f} → θ_new={new_theta:.4f} | lr={effective_lr:.5f}")
        self.logger.log(f"   │  📈 预测 MASE: {mase:.6f}")
        self.logger.log(
            f"   │  💰 Reward: {reward:.4f} (error={result.get('reward_detail', {}).get('error_term', 0):.4f}, "
            f"stability={result.get('reward_detail', {}).get('stability_penalty', 0):.4f})")
        self.logger.log(f"   │  📉 Advantage: {advantage:.4f} (reward - baseline={old_baseline:.4f})")
        self.logger.log(f"   │  📊 Baseline: {old_baseline:.4f} → {new_baseline:.4f}")
        self.logger.log(f"   └───")

        self._rl_total_reward += reward
        self._rl_mase_history.append(mase)
        if len(self._rl_mase_history) > 100:
            self._rl_mase_history.pop(0)
        self._rl_avg_mase = np.mean(self._rl_mase_history[-20:]) if self._rl_mase_history else 0
        self.logger.log(f"      ✅ [_rl_update] RL 更新完成, avg_mase={self._rl_avg_mase:.6f}")

        # ★★★ ★★★ ★★★ 每轮更新后检查 TRIAL → ACTIVE 晋升（温和晋升） ★★★ ★★★ ★★★
        if self.current_round > 0:
            promoted = self._promote_trial_policies()
            if promoted:
                self.logger.log(f"      ⭐ 晋升 {len(promoted)} 条策略为 ACTIVE: {promoted}")

    # ★★★ 更新策略统计（含 Regime 记录） ★★★
    def _update_policies_with_reward(self, mase: float, reward: float, state: Dict, window_data_path: str = None):
        numeric_state = state.get('numeric', {})
        regime = self.state_encoder.extract_regime(numeric_state) if hasattr(self.state_encoder, 'extract_regime') else {}
        regime_key = "_".join([f"{k}_{v}" for k, v in regime.items() if v == 1]) if regime else "平稳"

        for policy in self.policies:
            if not policy.is_applicable(numeric_state):
                continue
            state_key = self._generate_state_key(numeric_state)
            policy.record_error(mase, state_key, regime_key)
            policy.update_reward(reward)
            recent_errors = policy.error_history[-5:] if policy.error_history else [mase]
            avg_error = np.mean(recent_errors) if recent_errors else mase
            new_conf = np.exp(-avg_error / 0.5)
            policy.dynamic_confidence = (1 - self.ema_decay) * policy.dynamic_confidence + self.ema_decay * new_conf
            policy.dynamic_confidence = max(0.1, min(0.95, policy.dynamic_confidence))
            policy.usage_count += 1

    # ---------- 其他方法（保持不变） ----------
    def _soft_mixture_predict(self, observation: np.ndarray, horizon: int,
                              state: Dict, temperature: float) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        """后备 Soft Mixture，同样增加 Regime 偏置"""
        if not self.policies:
            return None, None

        numeric_state = state.get('numeric', {})
        active_policies = [p for p in self.policies if p.status not in ['ARCHIVE', 'DELETE']]

        if len(active_policies) == 0:
            return None, None

        regime = self.state_encoder.extract_regime(numeric_state) if hasattr(self.state_encoder, 'extract_regime') else {}
        regime_key = "_".join([f"{k}_{v}" for k, v in regime.items() if v == 1]) if regime else "平稳"

        scored = []
        for policy in active_policies:
            semantic_score = policy.compute_applicability_score(numeric_state)
            theta_norm = np.tanh(policy.logit_weight)
            theta_bias = self.theta_bias_alpha * theta_norm
            regime_bonus = policy.get_regime_bonus(regime_key, alpha=0.3) if regime_key else 0.0
            final_logit = semantic_score + theta_bias + regime_bonus
            scored.append((policy, final_logit, semantic_score, theta_bias, regime_bonus))

        scored.sort(key=lambda x: x[1], reverse=True)
        top_k = min(self.top_k, len(scored))
        top_policies = scored[:top_k]

        logits = np.array([s[1] for s in top_policies])
        exp_logits = np.exp(logits / self.temperature_old)
        weights = exp_logits / (np.sum(exp_logits) + 1e-8)

        period = self._estimate_period(observation)
        combined_pred = np.zeros(horizon)
        for i, (policy, _, _, _, _) in enumerate(top_policies):
            pred = policy.execute(observation, horizon, period)
            if pred is not None and len(pred) == horizon:
                combined_pred += pred * weights[i]
            else:
                combined_pred += np.full(horizon, np.mean(observation[-5:])) * weights[i]

        entropy = -np.sum(weights * np.log(weights + 1e-8))
        self._last_entropy = entropy

        if self.dominance_enabled and len(weights) > 0:
            max_weight_idx = np.argmax(weights)
            if weights[max_weight_idx] > 0.8:
                policy_id = top_policies[max_weight_idx][0].policy_id
                self._dominance_counts[policy_id] = self._dominance_counts.get(policy_id, 0) + 1
                if self._dominance_counts[policy_id] >= self.dominance_threshold:
                    for p in self.policies:
                        if p.policy_id == policy_id:
                            p.penalty_factor *= self.dominance_penalty_factor
                            p.penalty_factor = max(0.5, p.penalty_factor)
                            break
                    self._dominance_counts[policy_id] = 0
            else:
                for pid in self._dominance_counts:
                    self._dominance_counts[pid] = 0

        mixture_info = {
            'top_policies': [(p[0].policy_id, p[1]) for p in top_policies],
            'weights': weights.tolist(),
            'entropy': entropy
        }

        return combined_pred, mixture_info

    def _retrieve_policy(self, state: Dict) -> Optional[SkillPolicy]:
        if not self.policies:
            return None
        numeric = state.get('numeric', {})
        scored = []
        for policy in self.policies:
            if policy.status in ['ARCHIVE', 'DELETE']:
                continue
            score = policy.compute_applicability_score(numeric)
            scored.append((policy, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        if scored and scored[0][1] > 0.3:
            self._last_retrieved_policy = scored[0][0]
            scored[0][0].record_activation()
            return scored[0][0]
        return self._get_default_policy()

    def _get_default_policy(self) -> SkillPolicy:
        if self.policies:
            active = [p for p in self.policies if p.status == 'ACTIVE']
            if active:
                return max(active, key=lambda p: p.utility_ema)
        import hashlib, time
        default = SkillPolicy(
            policy_id=hashlib.md5(f"default_{time.time()}".encode()).hexdigest()[:8],
            name="default_policy",
            state_condition={},
            feature_groups=['trend_strength', 'seasonal_strength'],
            skill_strategy={'stages': [{'steps': 4, 'weights': {'chunk_ensemble': 0.7, 'multi_resolution': 0.3}},
                                       {'steps': 3, 'weights': {'chunk_ensemble': 0.6, 'multi_resolution': 0.4}}]},
            avg_mase=1.0,
            confidence=0.5,
            dynamic_confidence=0.5,
            reward_ema=0.5,
            utility_ema=0.5
        )
        self.policies.append(default)
        return default

    def _estimate_period(self, series: np.ndarray) -> int:
        try:
            from src.skills.data_profiler import DataProfiler
            return DataProfiler._auto_period(series)
        except:
            return 365 if len(series) > 365 else 7

    def _compute_mase_scale(self, series: np.ndarray, period: int) -> float:
        n = len(series)
        if n >= 2 * period:
            seasonal_errors = np.abs(series[period:] - series[:-period])
            return np.mean(seasonal_errors) if len(seasonal_errors) > 0 else 1.0
        else:
            naive_errors = np.abs(np.diff(series))
            return np.mean(naive_errors) if len(naive_errors) > 0 else 1.0

    def _compute_unified_reward(self, mase: float, result: Dict) -> float:
        error_term = -self.reward_mase_weight * mase
        entropy = result.get('entropy', self._last_entropy)
        entropy_change = abs(entropy - self._last_entropy)
        stability_penalty = -self.reward_stability_weight * entropy_change
        consistency_bonus = 0.0
        if self._last_retrieved_policy is not None:
            current_policy_id = result.get('policy', {}).policy_id if hasattr(result.get('policy'),
                                                                              'policy_id') else None
            if current_policy_id == getattr(self._last_retrieved_policy, 'policy_id', None):
                consistency_bonus = self.reward_consistency_weight * 0.5
        reward = error_term + stability_penalty + consistency_bonus
        result['reward_detail'] = {
            'error_term': error_term,
            'stability_penalty': stability_penalty,
            'consistency_bonus': consistency_bonus,
            'mase': mase,
            'entropy': entropy,
            'entropy_change': entropy_change
        }
        return reward

    def _get_regime_lr(self, regime: Dict, base_lr: float) -> float:
        if regime.get('volatility', 0) == 1:
            return base_lr * 1.5
        elif regime.get('seasonality', 0) == 1:
            return base_lr * 1.2
        else:
            return base_lr * 0.8

    def _generate_state_key(self, numeric_state: Dict) -> str:
        keys = ['trend_strength', 'seasonal_strength', 'cv']
        parts = []
        for k in keys:
            val = numeric_state.get(k, 0)
            if k == 'trend_strength':
                bucket = 'high' if val > 0.5 else 'low'
            elif k == 'seasonal_strength':
                bucket = 'high' if val > 0.3 else 'low'
            else:
                bucket = 'high' if val > 0.3 else 'low'
            parts.append(f"{k}_{bucket}")
        return '_'.join(parts)

    def load_policies(self, policies: List[SkillPolicy]):
        self.policies = policies
        for p in policies:
            if p.policy_id not in self.distribution_model.theta:
                self.distribution_model.theta[p.policy_id] = 0.0
            p.logit_weight = self.distribution_model.get_theta(p.policy_id)

    def save_policies(self, file_path: str):
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump({
                'policies': [p.to_dict() for p in self.policies],
                'total_loss': self.total_loss,
                'step_count': self.step_counter,
                'shared_reward': self.shared_reward,
                'distribution': self.distribution_model.to_dict(),
                'baseline': self.baseline_tracker.get(),
                'replay_memory': self.replay_memory.to_dict()
            }, f, ensure_ascii=False, indent=2)

    def get_state(self) -> Dict:
        avg_loss = np.mean(self.loss_history[-10:]) if self.loss_history else 0
        return {
            'step': self.step_counter,
            'policy_count': len(self.policies),
            'history_count': len(self.history),
            'total_loss': self.total_loss,
            'avg_loss': avg_loss,
            'entropy': self._last_entropy,
            'shared_reward': self.shared_reward,
            'policies': [p.get_summary() for p in self.policies],
            'rl_step_count': self._rl_step_count,
            'rl_avg_reward': self._rl_total_reward / self._rl_step_count if self._rl_step_count > 0 else 0,
            'rl_avg_mase': self._rl_avg_mase
        }

    def get_shared_reward(self) -> float:
        return self.shared_reward

    def get_reward_view(self) -> Dict:
        return self.reward_view

    def compute_policy_entropy(self) -> float:
        if not self.policies:
            return 1.0
        scores = [p.utility_ema + 0.1 for p in self.policies]
        exp_scores = np.exp(np.array(scores) / self.temperature)
        probs = exp_scores / (np.sum(exp_scores) + 1e-8)
        entropy = -np.sum(probs * np.log(probs + 1e-8))
        return float(entropy)

    def get_rl_stats(self) -> Dict:
        return {
            'step_count': self._rl_step_count,
            'total_reward': self._rl_total_reward,
            'avg_reward': self._rl_total_reward / self._rl_step_count if self._rl_step_count > 0 else 0,
            'avg_mase': self._rl_avg_mase,
            'mase_history_len': len(self._rl_mase_history),
            'baseline': self.baseline_tracker.get(),
            'theta_stats': {
                'min': min(self.distribution_model.theta.values()) if self.distribution_model.theta else 0,
                'max': max(self.distribution_model.theta.values()) if self.distribution_model.theta else 0,
                'mean': np.mean(list(self.distribution_model.theta.values())) if self.distribution_model.theta else 0
            }
        }