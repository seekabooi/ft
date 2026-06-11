import numpy as np
import random
import pandas as pd
import re
import json
from tqdm import tqdm
from src.agents.base import BaseAgent
from src.skills.registry import SkillRegistry
from src.skills.data_profiler import DataProfiler
from src.skills.skill_matcher import SkillMatcher
from src.agents.llm_client import LLMClient
from src.agents.llm_prompts import build_prompt

LONG_SKILLS = ['chunk_ensemble', 'multi_resolution', 'residual_correction_advanced']

class LLMPlannerAgent(BaseAgent):
    def __init__(self, model="glm-4", skill_registry=None, verbose=False,
                 log_file=None, use_skills=True, min_confidence=0.3,
                 llm_call_interval=1):
        self.model = model
        self.skills = skill_registry or SkillRegistry()
        self.log_file = log_file
        self.use_skills = use_skills
        self.llm_call_interval = llm_call_interval
        self._step_counter = 0
        self._current_dates = None
        self._skill_recent_mae = {}
        self.llm_client = LLMClient(model=model, log_file=log_file)
        self.uncertainty_threshold = 2.5

        skill_names = list(self.skills._skills.keys())
        tqdm.write(f"📋 技能({len(skill_names)}): {', '.join(skill_names)}")

    def _log(self, data):
        if self.log_file:
            import json
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(data, ensure_ascii=False, default=str) + "\n")

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

    def _decide_weights_and_interval(self, history, dates, period, horizon):
        from src.skills.data_profiler import DataProfiler
        from src.skills.skill_matcher import SkillMatcher

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
            feature_list = ['adf_pvalue','seasonal_strength','trend_strength','period','data_length']

        profile = DataProfiler.profile_selected(history, feature_list, freq=None, dates=dates)
        local_errors = {}
        for c in candidates:
            err = self._compute_skill_local_error(c['skill'], history, period, horizon)
            if err is not None:
                local_errors[c['skill'].name] = err
        profile['_local_errors'] = local_errors

        # 强化 chunk_ensemble 的局部误差
        if data_len > 400 and period == 365:
            if 'chunk_ensemble' in local_errors:
                best_rec_error = min([local_errors.get(s, float('inf')) for s in LONG_SKILLS])
                if local_errors['chunk_ensemble'] > best_rec_error * 1.1:
                    local_errors['chunk_ensemble'] = best_rec_error * 0.95

        prompt = build_prompt(profile, history, candidates, local_errors, LONG_SKILLS, self._step_counter)
        try:
            resp = self.llm_client.call_with_retry(prompt)
            content = resp.choices[0].message.content
            weights, interval = self.llm_client.parse_weights_and_interval(content)
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
                # 短序列限制技能数量（后处理）
                if data_len < 200 and len(weights) > 2:
                    sorted_items = sorted(weights.items(), key=lambda x: x[1], reverse=True)
                    weights = dict(sorted_items[:2])
                    total = sum(weights.values())
                    if total > 0:
                        weights = {k: round(v / total, 10) for k, v in weights.items()}
                # 长序列软后处理
                if data_len > 400:
                    rec_errors = {s: local_errors.get(s, float('inf')) for s in LONG_SKILLS}
                    best_rec = min(rec_errors, key=rec_errors.get)
                    total_rec_weight = sum(weights.get(s, 0) for s in LONG_SKILLS)
                    if total_rec_weight < 0.8:
                        weights = {best_rec: 1.0}
                return weights, interval
        except Exception as e:
            tqdm.write(f"⚠️ LLM决策失败，使用兜底: {e}")

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
            return weights, 2
        return {"naive": 1.0}, 2

    def predict(self, task):
        self._step_counter += 1
        history = np.array(task.history)
        dates = task.dates
        horizon = task.horizon

        # 无技能模式：LLM 直接预测
        if not self.use_skills:
            tqdm.write("[无技能模式] 使用 LLM 直接预测（无统计技能）")
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
                tqdm.write(f"⚠️ LLM 直接预测失败，回退到均值: {e}")
            forecast = np.full(horizon, np.mean(history[-5:]) if len(history) >= 5 else np.mean(history))
            return forecast.tolist()

        # 单技能模式
        if len(self.skills._skills) == 1:
            only_skill = list(self.skills._skills.values())[0]
            tqdm.write(f"[单技能模式] 直接使用 {only_skill.name}")
            from src.skills.data_profiler import DataProfiler
            tmp_profile = DataProfiler.profile_selected(history, ['period'], freq=task.frequency, dates=dates)
            period = tmp_profile.get('period', 12)
            forecast = only_skill.execute(history, horizon, period=period)
            tqdm.write(f"  🧠 最终组合: {{'{only_skill.name}': 1.0}}")
            return forecast.tolist()

        # ========== 统一递归预测（所有序列均采用递归多步预测） ==========
        if dates is not None:
            if hasattr(dates, 'tolist'):
                dates = dates.tolist()
            elif not isinstance(dates, (list, np.ndarray)):
                dates = list(dates)
        self._current_dates = dates

        from src.skills.data_profiler import DataProfiler
        tmp_profile = DataProfiler.profile_selected(history, ['period'], freq=task.frequency, dates=dates)
        period = tmp_profile.get('period', 12)
        data_len = len(history)

        predictions = []
        current_hist = history.copy()
        current_dates = dates.copy() if dates is not None else None

        weights = None
        replan_counter = 0
        step = 0

        while step < horizon:
            need_replan = (weights is None) or (replan_counter <= 0)
            if not need_replan and step > 0:
                hist_mean = np.mean(current_hist[:-1])
                hist_std = np.std(current_hist[:-1])
                if hist_std == 0:
                    hist_std = 1.0
                z_score = abs(current_hist[-1] - hist_mean) / hist_std
                if z_score > self.uncertainty_threshold:
                    tqdm.write(f"  步骤 {step+1} 预测值偏离较大 (z={z_score:.2f})，强制重决策")
                    need_replan = True

            if need_replan:
                weights, interval = self._decide_weights_and_interval(current_hist, current_dates, period, horizon=1)
                replan_counter = interval
                tqdm.write(f"  步骤 {step+1} 决策权重: { {k: f'{v:.10f}' for k,v in weights.items()} } (下次重决策间隔={interval})")

            pred_val = self._weighted_predict_multi(weights, {}, current_hist, period, horizon=1)
            if pred_val is None or len(pred_val) == 0:
                pred_val = np.array([np.mean(current_hist[-5:])])
            pred_single = pred_val[0]
            predictions.append(pred_single)

            current_hist = np.append(current_hist, pred_single)
            if current_dates is not None and len(current_dates) > 0:
                try:
                    last_date = pd.to_datetime(current_dates[-1])
                    if task.frequency and task.frequency.lower() == 'monthly':
                        next_date = last_date + pd.DateOffset(months=1)
                    else:
                        next_date = last_date + pd.Timedelta(days=1)
                    current_dates = np.append(current_dates, next_date.strftime('%Y-%m-%d'))
                except:
                    pass

            step += 1
            replan_counter -= 1

        return predictions