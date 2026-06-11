from typing import List, Dict, Any
import numpy as np
from src.skills.base import BaseSkill
from src.skills.data_profiler import DataProfiler

RECOMMENDED_LONG_SKILLS = ['multi_resolution', 'chunk_ensemble', 'residual_correction_advanced']

class SkillMatcher:
    def __init__(self, skills: List[BaseSkill], window_sizes=None, use_prototype=True):
        self.skills = skills
        self.window_sizes = window_sizes or [10, 30, 60]
        self.use_prototype = use_prototype
        self.baseline_similarity = 0.3
        self.use_counter = {skill.name: 0 for skill in skills}
        self.alpha_explore = 0.2
        self.dormant_skills = set()
        self.steps_since_selected = {skill.name: 0 for skill in skills}
        self.dormant_threshold = 12

    def match(self, history: np.ndarray, top_k: int = 5, dynamic_k: bool = True) -> List[Dict]:
        profile = DataProfiler.profile(history, window_sizes=self.window_sizes)
        data_len = len(history)
        period = profile.get('period', 12)
        seas = profile.get('seasonal_strength', 0)
        trend = profile.get('trend_strength', 0)

        if dynamic_k:
            active_count = len(self.skills) - len(self.dormant_skills)
            if data_len > 500:
                top_k = min(8, active_count)
            elif seas > 0.3 and trend > 0.3:
                top_k = min(7, active_count)
            elif seas < 0.1 and trend < 0.1:
                top_k = max(3, top_k)
            else:
                top_k = min(5, active_count)

        passed = []
        for skill in self.skills:
            if skill.name in self.dormant_skills:
                continue

            effective_min = max(skill.min_data_points, 2 * period if skill.name in ['seasonal_naive', 'multi_seasonal_naive'] else 3)
            if data_len < effective_min:
                continue

            if data_len > 500 and skill.name in ['croston', 'residual_correction']:
                continue

            check = skill.check_state_card(profile)
            if not check['applicable']:
                continue

            proto_sim = 0.0
            if self.use_prototype and skill.prototypes:
                cur_vec_old = np.array(profile.get('snapshot_vector', []))
                cur_trend = np.array(profile.get('trend_snapshot', []))
                cur_season = np.array(profile.get('seasonal_snapshot', []))
                best_old = 0.0
                best_new = 0.0
                for p in skill.prototypes:
                    pvec = np.array(p.get('vector', []))
                    if len(cur_vec_old) > 0 and len(pvec) > 0:
                        dist_old = DataProfiler.compute_dtw_distance(cur_vec_old, pvec)
                        sim_old = 1.0 / (1.0 + dist_old)
                        if sim_old > best_old:
                            best_old = sim_old
                    p_trend = np.array(p.get('trend_snapshot', []))
                    p_season = np.array(p.get('seasonal_snapshot', []))
                    if len(cur_trend) > 0 and len(p_trend) > 0 and len(cur_season) > 0 and len(p_season) > 0:
                        dist_trend = DataProfiler.compute_dtw_distance(cur_trend, p_trend)
                        dist_season = DataProfiler.compute_dtw_distance(cur_season, p_season)
                        sim_trend = 1.0 / (1.0 + dist_trend)
                        sim_season = 1.0 / (1.0 + dist_season)
                        sim_new = 0.6 * sim_trend + 0.4 * sim_season
                        if sim_new > best_new:
                            best_new = sim_new
                proto_sim = max(best_old, best_new) if max(best_old, best_new) > 0 else self.baseline_similarity
            else:
                proto_sim = self.baseline_similarity

            use_count = self.use_counter.get(skill.name, 0)
            explore_factor = 100.0 / data_len if data_len > 100 else 1.0
            explore_bonus = self.alpha_explore * (1.0 / (use_count + 1)) * explore_factor
            adjusted_sim = min(proto_sim + explore_bonus, 1.0)

            route_bonus = 0.0
            if data_len > 400 and seas > 0.4:
                if skill.name in RECOMMENDED_LONG_SKILLS:
                    route_bonus += 0.25
            if data_len < 100:
                if skill.name in ['naive', 'naive_drift', 'auto_arima']:
                    route_bonus += 0.1
            if trend > 0.6:
                if skill.name in ['holt_winters', 'theta', 'trend_forecaster', 'local_drift']:
                    route_bonus += 0.1
            if data_len > 300 and seas < 0.2 and skill.name == 'auto_arima':
                route_bonus += 0.1
            if data_len > 200 and data_len < 1000 and period == 365:
                if skill.name in ['fourier', 'calendar']:
                    route_bonus += 0.08
            adjusted_sim = min(adjusted_sim + route_bonus, 1.0)

            passed.append({
                'skill': skill, 'prototype_similarity': adjusted_sim,
                'state_card': skill.state_card,
                'visible_cues': check['visible_cues'],
                'verification_cue': check['verification_cue'],
                'failure_mode': check['failure_mode'],
                'fallback_skill': check['fallback_skill']
            })
        passed.sort(key=lambda x: x['prototype_similarity'], reverse=True)
        return passed[:top_k]

    def increment_use(self, skill_name):
        if skill_name in self.use_counter:
            self.use_counter[skill_name] += 1
            self.steps_since_selected[skill_name] = 0
        for name in self.steps_since_selected:
            if name != skill_name:
                self.steps_since_selected[name] += 1
                if self.steps_since_selected[name] >= self.dormant_threshold:
                    self.dormant_skills.add(name)

    def awaken_skill(self, skill_name):
        self.dormant_skills.discard(skill_name)
        self.steps_since_selected[skill_name] = 0