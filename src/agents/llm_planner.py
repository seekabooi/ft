import numpy as np
import random
import pandas as pd
import re
import json
import os
from tqdm import tqdm
from src.agents.base import BaseAgent
from src.skills.registry import SkillRegistry
from src.skills.data_profiler import DataProfiler
from src.skills.skill_matcher import SkillMatcher
from src.agents.llm_client import LLMClient
from src.agents.llm_prompts import build_prompt, build_preprocess_prompt, build_post_enhance_prompt

LONG_SKILLS = ['chunk_ensemble', 'multi_resolution', 'residual_correction_advanced']

# 导入预处理和后处理技能
from src.skills.preprocess_skills import (
    FillMissing, ClipOutliers, IdentityPre, ZScoreNormalize,
    LinearDetrend, BoxCoxTransform
)
from src.skills.postprocess_skills import (
    InvertIdentity, InvertZScore, InvertDetrend, InvertBoxCox,
    IdentityEnhance, ResidualARCorrection, QuantileCalibration
)

# 尝试导入规则引擎（若存在）
try:
    from experiments.autotune.rule_engine import RuleEngine
except ImportError:
    RuleEngine = None


class LLMPlannerAgent(BaseAgent):
    def __init__(self, model="glm-4", skill_registry=None, verbose=False,
                 log_file=None, use_skills=True, min_confidence=0.3,
                 llm_call_interval=1, rules_file=None, logger=None):
        self.model = model
        self.skills = skill_registry or SkillRegistry()
        self.log_file = log_file
        self.use_skills = use_skills
        self.llm_call_interval = llm_call_interval
        self._step_counter = 0
        self._current_dates = None
        self._skill_recent_mae = {}
        self.llm_client = LLMClient(model=model, log_file=log_file, logger=logger)
        self.uncertainty_threshold = 2.5
        self.verbose = verbose
        self.logger = logger

        # ★ 存储当前匹配的规则策略（供LLM参考）
        self._current_rule_strategy = None
        self._current_rule_policy = None  # ★ 新增：存储对应的策略对象

        # 预处理技能
        self.pre_skills = {
            'fill_missing': FillMissing(),
            'clip_outliers': ClipOutliers(),
            'identity_pre': IdentityPre(),
            'zscore_normalize': ZScoreNormalize(),
            'linear_detrend': LinearDetrend(),
            'boxcox_transform': BoxCoxTransform()
        }

        # 后处理增强技能
        self.enhance_skills = {
            'enhance_identity': IdentityEnhance(),
            'residual_ar': ResidualARCorrection(),
            'quantile_calibration': QuantileCalibration()
        }

        # 逆变换映射
        self.inverse_map = {
            'identity': InvertIdentity(),
            'zscore': InvertZScore(),
            'detrend': InvertDetrend(),
            'boxcox': InvertBoxCox()
        }

        # ★ 加载规则文件
        self.rules_file = rules_file
        self.rule_engine = None
        if rules_file and os.path.exists(rules_file) and RuleEngine is not None:
            try:
                with open(rules_file, 'r', encoding='utf-8') as f:
                    rules = json.load(f)
                self.rule_engine = RuleEngine(rules)
                if self.verbose:
                    self._log_info(f"📋 已加载规则文件: {rules_file}")
            except Exception as e:
                if self.verbose:
                    self._log_warning(f"⚠️ 规则文件加载失败: {e}")

        skill_names = list(self.skills._skills.keys())
        if self.verbose:
            self._log_info(f"📋 核心预测技能({len(skill_names)}): {', '.join(skill_names)}")
            self._log_info(f"📋 预处理技能: {list(self.pre_skills.keys())}")
            self._log_info(f"📋 后处理增强: {list(self.enhance_skills.keys())}")
        if self.rule_engine and self.verbose:
            self._log_info("📋 规则引擎已启用（策略将作为LLM参考建议）")

    def _log_info(self, msg: str):
        print(msg)
        if self.logger is not None:
            self.logger.log(msg)

    def _log_warning(self, msg: str):
        print(msg)
        if self.logger is not None:
            self.logger.log(msg, level="WARNING")

    def _log(self, data):
        if self.log_file:
            import json
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(data, ensure_ascii=False, default=str) + "\n")

    def _select_preprocessor(self, history, profile):
        prompt = build_preprocess_prompt(profile, history)
        try:
            resp = self.llm_client.call_with_retry(prompt)
            content = resp.choices[0].message.content
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                chosen_list = data.get("preprocess", [])
                if not isinstance(chosen_list, list):
                    chosen_list = [chosen_list] if chosen_list else []
                valid = [name for name in chosen_list if name in self.pre_skills]
                if valid:
                    return valid
        except Exception as e:
            if self.verbose:
                self._log_warning(f"⚠️ 预处理LLM决策失败，回退 ['identity_pre']: {e}")
        return ["identity_pre"]

    def _select_enhancement(self, history, profile, forecast_raw, horizon):
        n = len(history)
        if n < 20:
            return "enhance_identity"
        residuals = []
        if n < 15:
            return "enhance_identity"
        val_len = min(10, n // 3)
        start = n - val_len - 1
        for i in range(start, n - 1):
            pred = np.mean(history[max(0, i - 5):i + 1])
            residuals.append(history[i + 1] - pred)
        residuals = np.array(residuals)
        if len(residuals) < 5:
            return "enhance_identity"
        acf_lag1 = np.corrcoef(residuals[:-1], residuals[1:])[0, 1] if len(residuals) > 2 else 0.0
        if np.isnan(acf_lag1):
            acf_lag1 = 0.0
        var_ratio = np.var(residuals[-5:]) / (np.var(residuals) + 1e-8)
        residual_stats = {'acf_lag1': acf_lag1, 'var_ratio': var_ratio}

        prompt = build_post_enhance_prompt(profile, residual_stats, horizon)
        try:
            resp = self.llm_client.call_with_retry(prompt)
            content = resp.choices[0].message.content
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                chosen = data.get("enhance_skill", "enhance_identity")
                if chosen in self.enhance_skills:
                    return chosen
        except Exception as e:
            if self.verbose:
                self._log_warning(f"⚠️ 增强LLM决策失败，回退 enhance_identity: {e}")
        return "enhance_identity"

    def _compute_skill_local_error(self, skill, history, period, horizon):
        n = len(history)
        effective_horizon = min(horizon, 5)
        if n < max(skill.min_data_points, 5 + effective_horizon):
            return None
        if self._current_dates is not None:
            skill_dates = self._current_dates[-len(history):] if len(self._current_dates) >= len(history) else self._current_dates
        else:
            skill_dates = None

        if not skill.requires_full_history:
            val_len = min(10, n - skill.min_data_points - effective_horizon + 1)
            if val_len < 2:
                return None
            val_start = n - val_len - effective_horizon + 1
            train = history[:val_start]
            test = history[val_start:val_start + val_len]
            train_dates = skill_dates[:val_start] if skill_dates is not None else None
            errors = []
            for i in range(len(test)):
                cur_hist = np.concatenate([train, test[:i]])
                if skill_dates is not None:
                    cur_dates = np.concatenate([train_dates, skill_dates[val_start:val_start + i]]) if train_dates is not None else None
                else:
                    cur_dates = None
                try:
                    pred = skill.execute(cur_hist, effective_horizon, period=period, dates=cur_dates)[0]
                except:
                    pred = skill.execute(cur_hist, effective_horizon, period=period)[0]
                errors.append(abs(pred - test[i]))
            error_val = float(np.mean(errors)) if errors else None
        else:
            holdout_len = min(10, max(3, n // 5))
            if n - holdout_len < skill.min_data_points:
                return None
            train = history[:n - holdout_len]
            test = history[n - holdout_len:]
            train_dates = skill_dates[:n - holdout_len] if skill_dates is not None else None
            try:
                forecast = skill.execute(train, holdout_len, period=period, dates=train_dates)
                errors = np.abs(forecast[:effective_horizon] - test[:effective_horizon])
                error_val = float(np.mean(errors))
            except:
                return None

        if error_val is not None and n > 400 and skill.name in LONG_SKILLS:
            error_val = error_val * 0.8
        return error_val

    def _weighted_predict_multi(self, weight_dict, data_slices, history, period, horizon):
        preds = np.zeros(horizon)
        total_weight = 0.0
        for name, weight in weight_dict.items():
            sk = self.skills.get(name)
            if not sk or weight <= 0:
                continue
            try:
                slice_spec = data_slices.get(name, "all") if data_slices else "all"
                hist_segment = self._get_sliced_history(history, sk, slice_spec)
                if self._needs_dates(sk) and self._current_dates is not None:
                    date_segment = self._get_sliced_dates(self._current_dates, sk, slice_spec)
                    if date_segment is not None and len(date_segment) != len(hist_segment):
                        date_segment = date_segment[-len(hist_segment):]
                else:
                    date_segment = None
                if date_segment is not None:
                    if hasattr(date_segment, 'tolist'):
                        date_segment = date_segment.tolist()
                    elif isinstance(date_segment, pd.DatetimeIndex):
                        date_segment = date_segment.to_list()
                    forecast = sk.execute(hist_segment, horizon, period=period, dates=date_segment)
                else:
                    forecast = sk.execute(hist_segment, horizon, period=period)
                preds += np.array(forecast) * weight
                total_weight += weight
            except Exception as e:
                self._log({"event": "weighted_error", "skill": name, "error": str(e)})
                continue
        if total_weight > 0:
            return preds / total_weight
        return None

    def _get_sliced_history(self, history, skill, slice_spec):
        if skill.requires_full_history or slice_spec == "all":
            return history
        if slice_spec.startswith("last_"):
            try:
                n = int(slice_spec.split("_")[1])
                if n > 0 and n <= len(history):
                    return history[-n:]
            except:
                pass
        if not skill.requires_full_history:
            return history[-20:] if len(history) >= 20 else history
        return history

    def _get_sliced_dates(self, dates, skill, slice_spec):
        if dates is None:
            return None
        if skill.requires_full_history or slice_spec == "all":
            return dates
        if slice_spec.startswith("last_"):
            try:
                n = int(slice_spec.split("_")[1])
                if n > 0 and n <= len(dates):
                    return dates[-n:]
            except:
                pass
        if not skill.requires_full_history:
            return dates[-20:] if len(dates) >= 20 else dates
        return dates

    def _needs_dates(self, skill):
        required = getattr(skill, 'required_features', [])
        return 'has_dates' in required or 'month_of_year' in required or 'year' in required

    # ★★★ 核心修改：_format_strategy 增加语义摘要行 ★★★
    def _format_strategy(self, strategy):
        """
        将策略格式化为可读字符串，包含自然语言描述帮助LLM决策
        ★ 2026-06-26 增加语义摘要行（从策略描述中提取凝练信息）
        """
        if not strategy:
            return "无"

        stages = strategy.get('stages', [])
        if not stages:
            return "无有效阶段"

        # ★★★ 新增：提取策略描述作为语义摘要 ★★★
        strategy_desc = strategy.get('description', '')
        semantic_summary = ""
        if strategy_desc and len(strategy_desc) > 10:
            # 截取前40字作为摘要
            semantic_summary = f"【语义】{strategy_desc[:40]}{'...' if len(strategy_desc) > 40 else ''}\n"

        # 1. 提取阶段描述
        stage_desc = []
        total_steps = 0
        skill_usage = {}

        for i, st in enumerate(stages):
            steps = st.get('steps', 0)
            total_steps += steps
            weights = st.get('weights', {})
            w_str = ', '.join([f"{k}:{v:.2f}" for k, v in weights.items()])
            stage_desc.append(f"阶段{i+1}: 预测{steps}步, 权重{{{w_str}}}")

            for skill, weight in weights.items():
                skill_usage[skill] = skill_usage.get(skill, 0) + weight * steps

        if total_steps != 7:
            stage_desc.append(f"(总步数={total_steps}，注意：标准预测为7步)")

        # 2. 生成自然语言摘要
        if skill_usage:
            sorted_skills = sorted(skill_usage.items(), key=lambda x: x[1], reverse=True)
            dominant_skill = sorted_skills[0][0] if sorted_skills else "未知"
            dominant_weight = sorted_skills[0][1] if sorted_skills else 0
            if len(sorted_skills) > 1 and sorted_skills[1][1] > 0.1:
                second_skill = sorted_skills[1][0]
                summary = f"以「{dominant_skill}」为主导（权重占比{dominant_weight:.1%}），辅以「{second_skill}」"
            else:
                summary = f"高度依赖「{dominant_skill}」（权重占比{dominant_weight:.1%}）"

        if len(stages) == 1:
            rhythm = "全程使用固定权重组合，适合平稳序列"
        elif len(stages) <= 2:
            rhythm = "分为两个阶段，适合有明确转折点的序列"
        elif len(stages) <= 3:
            rhythm = "分为三个阶段，适合多阶段变化的序列"
        else:
            rhythm = "分为多个精细阶段，适合复杂动态序列"

        step_distribution = [s.get('steps', 0) for s in stages]
        if len(step_distribution) >= 2:
            if step_distribution[0] > sum(step_distribution[1:]) * 0.5:
                rhythm += "，前期投入较大，较为激进"
            elif step_distribution[-1] > sum(step_distribution[:-1]) * 0.5:
                rhythm += "，后期发力，较为稳健"
            else:
                rhythm += "，步数分布均匀，较为平衡"

        natural_desc = f"【策略意图】{summary}。{rhythm}。"

        # ★★★ 组合输出：语义摘要 + 自然语言描述 + 详细阶段 ★★★
        full_desc = f"{semantic_summary}{natural_desc}\n详细阶段: {' → '.join(stage_desc)}"

        return full_desc

    # ★★★ 以下方法保持不变 ★★★
    def _decide_weights_and_interval(self, history, dates, period, horizon):
        from src.skills.data_profiler import DataProfiler
        from src.skills.skill_matcher import SkillMatcher

        self._log_info(f"  🔍 [DEBUG] _decide_weights_and_interval 被调用")
        self._log_info(f"  🔍 [DEBUG] len(history)={len(history)}, period={period}, horizon={horizon}")
        self._log_info(f"  🔍 [DEBUG] _current_rule_strategy={self._current_rule_strategy is not None}")

        matcher = SkillMatcher(list(self.skills._skills.values()))
        candidates = matcher.match(history, top_k=5)

        data_len = len(history)
        if data_len > 400:
            existing_names = {c['skill'].name for c in candidates}
            for skill_name in LONG_SKILLS:
                skill = self.skills.get(skill_name)
                if skill and skill_name not in existing_names:
                    candidates.append({
                        'skill': skill,
                        'prototype_similarity': 0.9,
                        'state_card': skill.state_card,
                        'visible_cues': [],
                        'verification_cue': '',
                        'failure_mode': '',
                        'fallback_skill': 'naive'
                    })
            candidates.sort(key=lambda x: x['prototype_similarity'], reverse=True)

        has_dates = dates is not None and len(dates) > 0
        if has_dates and data_len >= 24 and not any(c['skill'].name == 'calendar' for c in candidates):
            cal_skill = self.skills.get('calendar')
            if cal_skill:
                candidates.append({
                    'skill': cal_skill,
                    'prototype_similarity': 0.3,
                    'state_card': cal_skill.state_card,
                    'visible_cues': [],
                    'verification_cue': '',
                    'failure_mode': '',
                    'fallback_skill': 'seasonal_naive'
                })

        if candidates:
            required_set = set()
            for c in candidates:
                required_set.update(c['skill'].required_features)
            required_set.add('period')
            feature_list = list(required_set)
        else:
            feature_list = ['adf_pvalue', 'seasonal_strength', 'trend_strength', 'period', 'data_length']

        profile = DataProfiler.profile_selected(history, feature_list, freq=None, dates=dates)
        local_errors = {}
        for c in candidates:
            err = self._compute_skill_local_error(c['skill'], history, period, horizon)
            if err is not None:
                local_errors[c['skill'].name] = err
        profile['_local_errors'] = local_errors

        if self._current_rule_strategy:
            profile['rule_strategy'] = self._format_strategy(self._current_rule_strategy)

        if data_len > 400 and period == 365:
            if 'chunk_ensemble' in local_errors:
                best_rec_error = min([local_errors.get(s, float('inf')) for s in LONG_SKILLS])
                if local_errors['chunk_ensemble'] > best_rec_error * 1.1:
                    local_errors['chunk_ensemble'] = best_rec_error * 0.95

        prompt = build_prompt(profile, history, candidates, local_errors, LONG_SKILLS, self._step_counter)
        try:
            self._log_info(f"  🔍 [DEBUG] 调用 LLM...")
            resp = self.llm_client.call_with_retry(prompt)
            content = resp.choices[0].message.content
            self._log_info(f"  🔍 [DEBUG] LLM 响应长度: {len(content)}")

            self._log_info(f"  🔍 [DEBUG] 调用 parse_weights_and_interval...")
            parsed = self.llm_client.parse_weights_and_interval(content)
            self._log_info(f"  🔍 [DEBUG] parse_weights_and_interval 返回类型: {type(parsed)}")
            self._log_info(
                f"  🔍 [DEBUG] parse_weights_and_interval 返回值数量: {len(parsed) if isinstance(parsed, tuple) else 'not tuple'}")

            if isinstance(parsed, tuple):
                if len(parsed) >= 5:
                    weights, interval, reasoning, relation, raw_data = parsed
                    self._log_info(
                        f"  🔍 [DEBUG] 解包5个值成功: weights={list(weights.keys()) if weights else None}, interval={interval}")
                else:
                    self._log_info(f"  🔍 [DEBUG] 返回 {len(parsed)} 个值，少于5个，使用兼容模式")
                    weights = parsed[0] if len(parsed) > 0 else None
                    interval = parsed[1] if len(parsed) > 1 else 2
                    self._log_info(
                        f"  🔍 [DEBUG] 兼容模式: weights={list(weights.keys()) if weights else None}, interval={interval}")
            else:
                self._log_info(f"  🔍 [DEBUG] 返回的不是元组，使用默认值")
                weights = None
                interval = 2

            if weights:
                total = sum(weights.values())
                if total > 0:
                    weights = {k: round(v / total, 10) for k, v in weights.items()}
                rng = random.Random(self._step_counter)
                for k in weights:
                    weights[k] += rng.uniform(-0.0000000002, 0.0000000002)
                total2 = sum(weights.values())
                if total2 > 0:
                    weights = {k: max(0.0000000001, round(v / total2, 10)) for k, v in weights.items()}
                if data_len < 200 and len(weights) > 2:
                    sorted_items = sorted(weights.items(), key=lambda x: x[1], reverse=True)
                    weights = dict(sorted_items[:2])
                    total = sum(weights.values())
                    if total > 0:
                        weights = {k: round(v / total, 10) for k, v in weights.items()}
                if data_len > 400:
                    rec_errors = {s: local_errors.get(s, float('inf')) for s in LONG_SKILLS}
                    best_rec = min(rec_errors, key=rec_errors.get)
                    total_rec_weight = sum(weights.get(s, 0) for s in LONG_SKILLS)
                    if total_rec_weight < 0.8:
                        weights = {best_rec: 1.0}
                self._log_info(f"  🔍 [DEBUG] 返回 weights={list(weights.keys())}, interval={interval}")
                return weights, interval
        except Exception as e:
            self._log_info(f"  🔍 [DEBUG] LLM决策异常: {type(e).__name__}: {e}")
            if self.verbose:
                self._log_warning(f"⚠️ LLM决策失败，使用兜底: {e}")

        self._log_info(f"  🔍 [DEBUG] 使用兜底逻辑")
        total_inv = 0.0
        temp = {}
        for c in candidates:
            name = c['skill'].name
            err = local_errors.get(name, 1.0)
            w = 1.0 / (err + 1e-10)
            temp[name] = w
            total_inv += w
        if total_inv > 0:
            weights = {k: round(v / total_inv, 10) for k, v in temp.items()}
            if len(weights) > 2:
                sorted_items = sorted(weights.items(), key=lambda x: x[1], reverse=True)
                weights = dict(sorted_items[:2])
                total = sum(weights.values())
                if total > 0:
                    weights = {k: round(v / total, 10) for k, v in weights.items()}
            self._log_info(f"  🔍 [DEBUG] 兜底返回 weights={list(weights.keys())}, interval=2")
            return weights, 2
        self._log_info(f"  🔍 [DEBUG] 最终回退 naive")
        return {"naive": 1.0}, 2

    def _predict_with_fixed_strategy(self, task, strategy):
        history = np.array(task.history)
        horizon = task.horizon
        period = DataProfiler._auto_period(history, freq=task.frequency, dates=task.dates)

        stages = strategy.get('stages', [])
        if not stages:
            return self.predict(task)

        total_steps = sum(s.get('steps', 0) for s in stages)
        if total_steps != horizon:
            last_stage = stages[-1]
            diff = horizon - total_steps
            if diff > 0:
                last_stage['steps'] = last_stage.get('steps', 0) + diff

        predictions = []
        current_hist = history.copy()

        for stage in stages:
            steps = stage.get('steps', 0)
            weights = stage.get('weights', {})

            for _ in range(steps):
                pred_val = 0.0
                total_w = 0.0
                for skill_name, weight in weights.items():
                    skill = self.skills.get(skill_name)
                    if skill and weight > 0:
                        try:
                            forecast = skill.execute(current_hist, 1, period=period)
                            if forecast is not None and len(forecast) > 0:
                                pred_val += forecast[0] * weight
                                total_w += weight
                        except Exception as e:
                            self._log({"event": "fixed_strategy_error", "skill": skill_name, "error": str(e)})
                if total_w > 0:
                    pred_val = pred_val / total_w
                else:
                    pred_val = np.mean(current_hist[-5:]) if len(current_hist) >= 5 else np.mean(current_hist)

                predictions.append(pred_val)
                current_hist = np.append(current_hist, pred_val)

        return predictions[:horizon]

    # ------------------ 主预测入口 ------------------
    def predict(self, task):
        self._current_rule_strategy = None
        self._current_rule_policy = None
        if self.rule_engine is not None:
            history = np.array(task.history)
            try:
                from experiments.autotune.utils import extract_features
                features = extract_features(history)
                strategy = self.rule_engine.get_strategy(features)
                if strategy:
                    self._current_rule_strategy = strategy
                    # ★ 尝试获取对应的策略对象（如果有 policy_id）
                    # 这里简化处理，只存储策略字典
            except Exception as e:
                pass

        self._step_counter += 1
        history = np.array(task.history)
        dates = task.dates
        horizon = task.horizon

        if not self.use_skills:
            if self.verbose:
                self._log_info("[无技能模式] 使用 LLM 直接预测（无统计技能）")
            recent_points = history[-20:].tolist()
            prompt = f"""你是一个时间序列预测专家。请根据以下历史数据（最近20个点）预测未来 {horizon} 个点的数值。
历史数据（按时间顺序，最近20个点，越靠右越新）：
{recent_points}

请输出一个 JSON 数组，长度为 {horizon}，包含预测值，保留两位小数。
例如：[100.5, 102.3, 105.1, ...]
只输出 JSON 数组，不要任何解释。"""
            try:
                resp = self.llm_client.call_with_retry(prompt, max_retries=2)
                content = resp.choices[0].message.content
                json_match = re.search(r'\[.*\]', content, re.DOTALL)
                if json_match:
                    pred_list = json.loads(json_match.group())
                    if len(pred_list) == horizon:
                        return pred_list
            except Exception as e:
                if self.verbose:
                    self._log_warning(f"⚠️ LLM 直接预测失败，回退到均值: {e}")
            forecast = np.full(horizon, np.mean(history[-5:]) if len(history) >= 5 else np.mean(history))
            return forecast.tolist()

        if len(self.skills._skills) == 1:
            only_skill = list(self.skills._skills.values())[0]
            if self.verbose:
                self._log_info(f"[单技能模式] 直接使用 {only_skill.name}")
            tmp_profile = DataProfiler.profile_selected(history, ['period'], freq=task.frequency, dates=dates)
            period = tmp_profile.get('period', 12)
            forecast = only_skill.execute(history, horizon, period=period)
            if self.verbose:
                self._log_info(f"  🧠 最终组合: {{'{only_skill.name}': 1.0}}")
            return forecast.tolist()

        # =============================================================
        # 阶段 1/4：预处理
        # =============================================================
        if self.verbose:
            self._log_info("\n" + "=" * 50)
            self._log_info("📍 [阶段 1/4] 预处理阶段 (Preprocessing)")
            self._log_info("=" * 50)

        profile = DataProfiler.profile_selected(history, ['skewness','cv','trend_strength','adf_pvalue','period','data_length','missing_rate'], dates=dates)
        pre_names = self._select_preprocessor(history, profile)

        transformed_hist = history.copy()
        context = {"method": "identity"}
        for pre_name in pre_names:
            pre_skill = self.pre_skills[pre_name]
            transformed_hist, ctx = pre_skill.execute_with_context(transformed_hist)
            if ctx.get('method') != 'identity':
                context = ctx
            if self.verbose:
                self._log_info(f"  ✅ 执行预处理: {pre_name}")

        context['orig_len'] = len(history)
        if self.verbose:
            self._log_info(f"  📊 变换后均值={np.mean(transformed_hist):.4f}, 标准差={np.std(transformed_hist):.4f}")

        if dates is not None:
            if hasattr(dates, 'tolist'):
                dates = dates.tolist()
            elif not isinstance(dates, (list, np.ndarray)):
                dates = list(dates)
        self._current_dates = dates

        # =============================================================
        # 阶段 2/4：核心预测
        # =============================================================
        if self.verbose:
            self._log_info("\n" + "=" * 50)
            self._log_info("📍 [阶段 2/4] 核心预测阶段 (Core Forecasting)")
            self._log_info("=" * 50)

        period = DataProfiler._auto_period(transformed_hist, freq=task.frequency, dates=dates)
        if period is None:
            period = 7
        if self.verbose:
            self._log_info(f"  🔄 检测周期: {period}")

        weights, interval = self._decide_weights_and_interval(
            transformed_hist, dates, period, horizon=horizon
        )
        if self.verbose:
            weight_str = ", ".join([f"{k}: {v}" for k, v in weights.items()])
            self._log_info(f"  📌 LLM决策权重: {{{weight_str}}} (重决策间隔={interval})")

        pred_arr = self._weighted_predict_multi(weights, {}, transformed_hist, period, horizon=horizon)
        if pred_arr is None or len(pred_arr) == 0:
            pred_arr = np.full(horizon, np.mean(transformed_hist[-5:]) if len(transformed_hist) >= 5 else np.mean(transformed_hist))
        predictions_transformed = pred_arr.tolist()

        if self.verbose:
            self._log_info(f"  ✅ 预测完成，共 {len(predictions_transformed)} 步")

        forecast_transformed = np.array(predictions_transformed)

        # =============================================================
        # 阶段 3/4：逆变换
        # =============================================================
        if self.verbose:
            self._log_info("\n" + "=" * 50)
            self._log_info("📍 [阶段 3/4] 逆变换阶段 (Inverse Transform)")
            self._log_info("=" * 50)

        inv_method = context.get('method', 'identity')
        inverse_skill = self.inverse_map.get(inv_method, InvertIdentity())
        if self.verbose:
            self._log_info(f"  🔗 自动绑定逆变换: {inverse_skill.name}")
        forecast_raw = inverse_skill.execute(forecast_transformed, context=context)
        if self.verbose:
            self._log_info(f"  ✅ 已还原到原始尺度 (均值={np.mean(forecast_raw):.4f})")

        # =============================================================
        # 阶段 4/4：后处理增强
        # =============================================================
        if self.verbose:
            self._log_info("\n" + "=" * 50)
            self._log_info("📍 [阶段 4/4] 后处理增强阶段 (Post-Processing Enhancement)")
            self._log_info("=" * 50)

        enhance_name = self._select_enhancement(history, profile, forecast_raw, horizon)
        enhance_skill = self.enhance_skills[enhance_name]
        if self.verbose:
            self._log_info(f"  🚀 选择增强方法: {enhance_name}")

        final_forecast = enhance_skill.execute(forecast_raw, history=history, horizon=horizon)
        if self.verbose:
            self._log_info(f"  ✅ 增强完成，最终预测均值={np.mean(final_forecast):.4f}")
            self._log_info("=" * 50 + "\n")

        return final_forecast.tolist()

    # ------------------ 带轨迹预测 ------------------
    def predict_with_trajectory(self, task):
        self._step_counter += 1
        history = np.array(task.history)
        dates = task.dates
        horizon = task.horizon

        if not self.use_skills:
            return self.predict(task), []

        if len(self.skills._skills) == 1:
            return self.predict(task), []

        self._current_rule_strategy = None
        self._current_rule_policy = None
        if self.rule_engine is not None:
            try:
                from experiments.autotune.utils import extract_features
                features = extract_features(history)
                strategy = self.rule_engine.get_strategy(features)
                if strategy:
                    self._current_rule_strategy = strategy
            except:
                pass

        if dates is not None:
            if hasattr(dates, 'tolist'):
                dates = dates.tolist()
            elif not isinstance(dates, (list, np.ndarray)):
                dates = list(dates)
        self._current_dates = dates

        from src.skills.data_profiler import DataProfiler
        tmp_profile = DataProfiler.profile_selected(history, ['period'], freq=task.frequency, dates=dates)
        period = tmp_profile.get('period', 12)

        weights, interval = self._decide_weights_and_interval(
            history, dates, period, horizon=horizon
        )
        trajectory = [{
            'step': 0,
            'weights': weights.copy(),
            'interval': interval
        }]

        pred_arr = self._weighted_predict_multi(weights, {}, history, period, horizon=horizon)
        if pred_arr is None or len(pred_arr) == 0:
            pred_arr = np.full(horizon, np.mean(history[-5:]) if len(history) >= 5 else np.mean(history))
        predictions = pred_arr.tolist()

        return predictions, trajectory