import re, json, numpy as np, time, random
from openai import OpenAI
from tqdm import tqdm
from scipy import stats as scipy_stats
from src.agents.base import BaseAgent
from src.skills.registry import SkillRegistry
from src.config import ZHIPU_API_KEY, OPENAI_API_BASE

class LLMPlannerAgent(BaseAgent):
    def __init__(self, model="glm-4", skill_registry=None, verbose=False,
                 log_file=None, use_skills=True, min_confidence=0.3,
                 llm_call_interval=1):
        self.model = model
        self.client = OpenAI(api_key=ZHIPU_API_KEY, base_url=OPENAI_API_BASE, timeout=30)
        self.skills = skill_registry or SkillRegistry()
        self.log_file = log_file
        self.use_skills = use_skills
        self.llm_call_interval = llm_call_interval
        self._last_used_skill = None
        self._skill_streak = 0
        self._step_counter = 0

        # 缓存（在固定起源模式下不再使用增量，但保留字段兼容）
        self._cached_history = None
        self._cached_profile = None
        self._cached_errors = {}
        self._error_cache_counter = 0
        self._last_plan = None
        self._last_pred_cache = {}

        self._current_dates = None
        self._current_full_history = None
        self._skill_recent_mae = {}

        skill_names = list(self.skills._skills.keys())
        tqdm.write(f"📋 技能({len(skill_names)}): {', '.join(skill_names)}")

    def _log(self, data: dict):
        if self.log_file:
            try:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(data, ensure_ascii=False, default=str) + "\n")
            except Exception:
                pass

    def _fast_update_profile(self, history, period):
        profile = self._cached_profile.copy()
        n = len(history)
        profile['data_length'] = n
        if n >= 5:
            x = np.arange(5); y = history[-5:]
            slope, _, _, _, _ = scipy_stats.linregress(x, y)
            profile['local_slope'] = round(slope, 6)
        else:
            profile['local_slope'] = 0.0
        if n >= 5 and np.std(history) > 0:
            profile['recent_volatility'] = float(np.std(history[-5:]) / np.std(history))
        else:
            profile['recent_volatility'] = 1.0
        if n >= 20:
            recent_var = np.var(history[-20:])
            total_var = np.var(history)
            profile['change_point_detected'] = (total_var > 0 and recent_var / total_var > 3.0)
        else:
            profile['change_point_detected'] = False
        snapshots = {}
        w = len(history)
        if w <= len(history):
            seg = history[-w:]
            seg_min, seg_max = np.min(seg), np.max(seg)
            seg_norm = (seg - seg_min) / (seg_max - seg_min + 1e-8)
            snapshots[f'w{w}'] = seg_norm.tolist()
        if len(history) >= 2:
            diff = np.diff(history[-20:])
            d_min, d_max = np.min(diff), np.max(diff)
            if d_max > d_min:
                diff_norm = (diff - d_min) / (d_max - d_min)
            else:
                diff_norm = np.zeros_like(diff)
            snapshots['diff20'] = diff_norm.tolist()
        profile['snapshots'] = snapshots
        profile['snapshot_vector'] = np.concatenate([np.array(v) for v in snapshots.values()]).tolist() if snapshots else []
        return profile

    def _needs_dates(self, skill):
        required = getattr(skill, 'required_features', [])
        return 'has_dates' in required or 'month_of_year' in required or 'year' in required

    def _compute_skill_local_error(self, skill, history: np.ndarray, period: int):
        n = len(history)
        if n < max(skill.min_data_points, 5):
            return None
        if self._needs_dates(skill) and self._current_dates is not None:
            skill_dates = self._current_dates[-len(history):] if len(self._current_dates) >= len(history) else self._current_dates
        else:
            skill_dates = None

        if not skill.requires_full_history:
            val_len = min(10, n - skill.min_data_points)
            if val_len < 2:
                return None
            val_start = n - val_len
            train = history[:val_start]
            test = history[val_start:]
            train_dates = skill_dates[:val_start] if skill_dates is not None else None
            errors = []
            for i in range(len(test)):
                cur_hist = np.concatenate([train, test[:i]])
                if skill_dates is not None:
                    cur_dates = np.concatenate([train_dates, skill_dates[val_start:val_start+i]]) if train_dates is not None else None
                else:
                    cur_dates = None
                try:
                    pred = skill.execute(cur_hist, 1, period=period, dates=cur_dates)[0]
                except:
                    pred = skill.execute(cur_hist, 1, period=period)[0]
                errors.append(abs(pred - test[i]))
            return float(np.mean(errors)) if errors else None
        else:
            holdout_len = min(10, max(3, n // 5))
            if n - holdout_len < skill.min_data_points:
                return None
            train = history[:n - holdout_len]
            test = history[n - holdout_len:]
            train_dates = skill_dates[:n - holdout_len] if skill_dates is not None else None
            try:
                forecast = skill.execute(train, holdout_len, period=period, dates=train_dates)
                return float(np.mean(np.abs(forecast - test)))
            except:
                return None

    def _get_sliced_history(self, history: np.ndarray, skill, slice_spec: str):
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

    def _get_sliced_dates(self, dates, skill, slice_spec: str):
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

    # ---------- 多步加权预测 ----------
    def _weighted_predict_multi(self, weight_dict: dict, data_slices: dict,
                                history: np.ndarray, period: int, horizon: int):
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

    # 保留单步兼容（旧接口不使用，但保留）
    def _weighted_predict(self, weight_dict, data_slices, history, period, horizon):
        arr = self._weighted_predict_multi(weight_dict, data_slices, history, period, horizon)
        if arr is not None:
            return float(np.mean(arr))
        return None

    def _try_plan(self, plan: dict, history: np.ndarray, period: int, horizon: int):
        weights = plan.get("skill_weights")
        data_slices = plan.get("data_slices", {})
        if weights and isinstance(weights, dict) and len(weights) > 0:
            valid = {k: v for k, v in weights.items() if v > 0 and self.skills.get(k)}
            if valid:
                pred_arr = self._weighted_predict_multi(valid, data_slices, history, period, horizon)
                if pred_arr is not None:
                    detail = ", ".join([f"{k}:{v:.6f}" for k, v in valid.items()])
                    tqdm.write(f"  🧠 组合({len(valid)}个): {detail}")
                    return pred_arr, True
        skill_name = plan.get("best_skill")
        if skill_name:
            sk = self.skills.get(skill_name)
            if sk:
                try:
                    slice_spec = data_slices.get(skill_name, "all") if data_slices else "all"
                    hist_segment = self._get_sliced_history(history, sk, slice_spec)
                    if self._needs_dates(sk) and self._current_dates is not None:
                        date_segment = self._get_sliced_dates(self._current_dates, sk, slice_spec)
                        forecast = sk.execute(hist_segment, horizon, period=period, dates=date_segment)
                    else:
                        forecast = sk.execute(hist_segment, horizon, period=period)
                    ver = sk.verify_prediction(history, float(np.mean(forecast)))
                    tqdm.write(f"  ⚙️ 单技能 {sk.name}")
                    if ver.get("valid"):
                        return np.array(forecast), True
                    else:
                        self._log({"event": "verify_fail", "skill": sk.name, "z_score": ver.get("z_score")})
                except Exception as e:
                    self._log({"event": "skill_error", "skill": skill_name, "error": str(e)})
        return None, False

    def _update_skill_recent_mae(self, local_errors):
        for name, mae in local_errors.items():
            if name not in self._skill_recent_mae:
                self._skill_recent_mae[name] = []
            self._skill_recent_mae[name].append(mae)
            if len(self._skill_recent_mae[name]) > 5:
                self._skill_recent_mae[name].pop(0)

    def _get_skill_performance_hint(self):
        hints = []
        for name, mae_list in self._skill_recent_mae.items():
            if len(mae_list) >= 3:
                avg_mae = np.mean(mae_list)
                hints.append(f"{name} 近{len(mae_list)}步平均MAE: {avg_mae:.6f}")
        if hints:
            return "技能近期表现：\n" + "\n".join(hints) + "\n"
        return ""

    def _call_llm_with_retry(self, prompt, max_retries=2):
        last_exception = None
        for attempt in range(max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role":"user","content":prompt}],
                    temperature=0.35,
                    max_tokens=400,
                    timeout=20
                )
                return resp
            except Exception as e:
                last_exception = e
                if attempt < max_retries:
                    tqdm.write(f"  ⚠️ LLM调用失败 (尝试 {attempt+1}/{max_retries+1}), 重试...")
                    time.sleep(2)
                else:
                    raise last_exception

    def _llm_plan_decision(self, candidates, profile, history, dates):
        seas = profile.get('seasonal_strength', 0)
        trend = profile.get('trend_strength', 0)
        period = profile.get('period', 12)
        local_slope = profile.get('local_slope', 0.0)
        change_detected = profile.get('change_point_detected', False)
        local_errors = profile.get('_local_errors', {})
        data_len = len(history)
        month = profile.get('month_of_year', 0)
        year = profile.get('year', 0)
        quarter = profile.get('quarter', 0)
        is_month_end = profile.get('is_month_end', False)
        days_from_start = profile.get('days_from_start', data_len)

        recent_points = list(history[-10:])
        recent_str = ", ".join([f"{v:.1f}" for v in recent_points])
        if len(history) >= 5:
            win_mean = np.mean(history[-5:])
            win_std = np.std(history[-5:])
            win_trend = "上升" if history[-1] > history[-5] else "下降"
        else:
            win_mean, win_std, win_trend = 0.0, 0.0, "未知"

        streak_warning = ""
        if self._skill_streak >= 3:
            streak_warning = f"⚠️ 技能 '{self._last_used_skill}' 已连续使用 {self._skill_streak} 次，建议考虑其他技能或加权组合。\n"

        skill_info_lines = []
        error_list = []
        for c in candidates:
            sk = c['skill']
            mae = local_errors.get(sk.name)
            mae_str = f"{mae:.6f}" if mae is not None else "未计算"
            error_list.append((sk.name, mae))
            min_data = sk.min_data_points
            full_hist = "是" if sk.requires_full_history else "否"
            tags = ", ".join(sk.strength_tags) if sk.strength_tags else "通用"
            hint = sk.decision_hint if sk.decision_hint else "无特殊建议"
            skill_info_lines.append(
                f"- {sk.name}: MAE={mae_str} | 最少数据:{min_data} | 需全历史:{full_hist} | 擅长:{tags}\n  使用建议: {hint}"
            )
        skill_info = "\n".join(skill_info_lines)

        valid_errors = [(n, e) for n, e in error_list if e is not None]
        error_hint = ""
        if len(valid_errors) >= 2:
            best_name, best_mae = min(valid_errors, key=lambda x: x[1])
            worst_name, worst_mae = max(valid_errors, key=lambda x: x[1])
            error_hint = f"最低MAE技能: {best_name} ({best_mae:.6f}), 最高MAE技能: {worst_name} ({worst_mae:.6f})。\n"

        season_hint = ""
        if seas > 0.5:
            season_hint = (
                f"序列季节性较强（{seas:.6f}），季节类技能（如 holt_winters, seasonal_naive, prophet, calendar, multi_seasonal_naive）可能更有优势。\n"
            )

        calendar_hint = ""
        if seas > 0.5 and data_len >= 24 and profile.get('has_dates', False):
            calendar_hint = (
                "⚠️ 该序列季节性极强且有完整日期信息，你必须在组合中包含 calendar 技能（权重≥0.10）。\n"
            )

        precision_hint = (
            "请输出精确到六位小数的权重（如 0.723456, 0.186723, 0.089821），绝对不要使用 0.1, 0.2 这种整十数值。\n"
        )

        performance_hint = self._get_skill_performance_hint()

        date_info = ""
        if profile.get('has_dates', False):
            date_info = f"- 当前时间点：{year}年{month}月 (Q{quarter})，{'月末' if is_month_end else '非月末'}，距起始 {days_from_start} 天\n"

        prompt = f"""你是时间序列预测专家。请根据以下特征决定最佳预测方案，可使用1~3个技能加权组合。

序列特征：
- 长度:{data_len}，季节:{seas:.6f}，趋势:{trend:.6f}，周期:{period}
- 局部斜率:{local_slope:.6f}，突变:{change_detected}
- 近期5点均值:{win_mean:.6f}，波动:{win_std:.6f}，走势:{win_trend}
- 最近10点: {recent_str}
{date_info}
{streak_warning}
候选技能对比：
{skill_info}
{error_hint}{season_hint}{calendar_hint}{precision_hint}{performance_hint}
数据切片："all" 用全部历史；"last_N" 用最近N个点。

要求：
1. 输出 "skill_weights" 字段，权重必须精确到六位小数（如 0.723456, 0.186723）。
2. 权重之和不必为1，系统会自动归一化。
3. 为每个技能指定 "data_slices"（轻量默认 "last_20"，模型默认 "all"）。
4. 额外输出一个 "reasoning" 字段，用简短中文解释选择逻辑。

输出纯JSON，格式如：
{{"skill_weights": {{"holt_winters": 0.723456, "calendar": 0.186723, "seasonal_naive": 0.089821}}, "data_slices": {{...}}, "reasoning": "强季节且近期趋势平缓，主用 holt_winters，calendar 作为基准"}}
不要任何解释。"""

        try:
            resp = self._call_llm_with_retry(prompt, max_retries=2)
            content = resp.choices[0].message.content
            self._log({"event": "llm_raw_response", "content": content})
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                try:
                    plan = json.loads(json_match.group())
                    # 强制注入 calendar
                    if seas > 0.5 and data_len >= 24 and profile.get('has_dates', False):
                        plan.setdefault("skill_weights", {})
                        if "calendar" not in plan["skill_weights"] or plan["skill_weights"]["calendar"] < 0.10:
                            plan["skill_weights"]["calendar"] = 0.15
                    # 第一次归一化
                    total = sum(plan["skill_weights"].values())
                    if total > 0:
                        for k in plan["skill_weights"]:
                            plan["skill_weights"][k] = round(plan["skill_weights"][k] / total, 6)
                    # 轻微扰动，打破整十模式
                    rng = random.Random(self._step_counter)
                    for k in plan["skill_weights"]:
                        plan["skill_weights"][k] += rng.uniform(-0.000002, 0.000002)
                    total2 = sum(plan["skill_weights"].values())
                    for k in plan["skill_weights"]:
                        plan["skill_weights"][k] = max(0.000001, round(plan["skill_weights"][k] / total2, 6))
                    self._log({"event": "llm_plan_parsed", "plan": plan})
                    return plan
                except json.JSONDecodeError as e:
                    self._log({"event": "llm_json_error", "error": str(e), "content": content})
                    return {}
            else:
                self._log({"event": "llm_no_json", "content": content})
                return {}
        except Exception as e:
            self._log({"event": "llm_error", "error": str(e)})
            tqdm.write(f"  ⚠️ LLM调用持续失败，启用规则兜底")
            return {}

    # ==================== 主预测接口（固定起源多步） ====================
    def predict(self, task):
        self._step_counter += 1
        history = np.array(task.history)
        dates = task.dates
        horizon = task.horizon
        self._current_dates = dates
        self._current_full_history = history.copy()

        from src.skills.data_profiler import DataProfiler
        from src.skills.skill_matcher import SkillMatcher

        matcher = SkillMatcher(list(self.skills._skills.values()))
        candidates = matcher.match(history, top_k=5)

        # 强制将 calendar 加入候选
        has_dates = dates is not None and len(dates) > 0
        if has_dates and len(history) >= 24 and not any(c['skill'].name == 'calendar' for c in candidates):
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

        # 固定起源：每次都是全新评估，不使用增量缓存
        profile = DataProfiler.profile_selected(history, feature_list, freq=task.frequency, dates=dates)
        # 清除缓存，避免下次误用
        self._cached_history = None
        self._cached_profile = None

        log_profile = {k: v for k, v in profile.items() if k != 'snapshot_vector'}
        self._log({"event": "profile", "task_id": task.id, "profile": log_profile})

        period = profile.get("period", 12)

        if self.use_skills:
            if not candidates:
                naive = self.skills.get("naive")
                if naive:
                    return naive.execute(history, horizon, period=period).tolist()
                return np.full(horizon, float(np.mean(history))).tolist()

            # 计算局部误差（一次性）
            local_errors = {}
            for c in candidates:
                sk = c['skill']
                err = self._compute_skill_local_error(sk, history, period)
                if err is not None:
                    local_errors[sk.name] = err
            profile['_local_errors'] = local_errors
            self._update_skill_recent_mae(local_errors)
            self._log({"event": "local_errors", "task_id": task.id, "errors": local_errors})

            # LLM 决策
            plan = self._llm_plan_decision(candidates, profile, history, dates)
            self._log({"event": "llm_plan_new", "task_id": task.id, "plan": plan})

            if "skill_weights" not in plan or not plan["skill_weights"]:
                weights = {}
                total_inv = 0.0
                epsilon = 1e-6
                for c in candidates:
                    name = c['skill'].name
                    err = local_errors.get(name)
                    if err is not None:
                        w = 1.0 / (err + epsilon)
                    else:
                        w = 0.1
                    weights[name] = w
                    total_inv += w
                if total_inv > 0:
                    weights = {k: round(v / total_inv, 6) for k, v in weights.items()}
                else:
                    weights = {"naive_drift": 0.7, "naive": 0.3}
                seas = profile.get('seasonal_strength', 0)
                if has_dates and seas > 0.5 and len(history) >= 24 and "calendar" not in weights:
                    weights["calendar"] = 0.15
                    total = sum(weights.values())
                    weights = {k: round(v / total, 6) for k, v in weights.items()}
                plan["skill_weights"] = weights
                plan["data_slices"] = {n: ("last_20" if not self.skills.get(n).requires_full_history else "all") for n in weights}
                detail = ", ".join([f"{k}:{v:.6f}" for k, v in weights.items()])
                tqdm.write(f"  ⚡ 兜底组合({len(weights)}个): {detail}")
                self._log({"event": "auto_weights", "weights": weights})

            data_slices = plan.get("data_slices", {})
            pred_array = self._weighted_predict_multi(plan["skill_weights"], data_slices, history, period, horizon)

            if pred_array is not None:
                detail = ", ".join([f"{k}:{v:.6f}" for k, v in plan["skill_weights"].items()])
                tqdm.write(f"  🧠 最终组合({len(plan['skill_weights'])}个): {detail}")
                return pred_array.tolist()

            # 回退：逐个候选技能
            for c in candidates:
                sk = c['skill']
                try:
                    forecast = sk.execute(history, horizon, period=period)
                    return forecast.tolist()
                except:
                    continue

            naive = self.skills.get("naive")
            if naive:
                try:
                    return naive.execute(history, horizon, period=period).tolist()
                except:
                    pass
            return np.full(horizon, float(np.mean(history))).tolist()

        naive = self.skills.get("naive")
        if naive:
            return naive.execute(history, horizon, period=period).tolist()
        return np.full(horizon, float(np.mean(history))).tolist()

 