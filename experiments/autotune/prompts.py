# experiments/autotune/prompts.py
"""
提示词加载器（从 prompts.yaml 读取模板，使用 Jinja2 渲染）
所有提示词内容均在 prompts.yaml 中管理。
"""

import os
import yaml
import numpy as np
from jinja2 import Template

# 全局缓存模板
_PROMPT_TEMPLATES = None

def _load_templates():
    global _PROMPT_TEMPLATES
    if _PROMPT_TEMPLATES is None:
        yaml_path = os.path.join(os.path.dirname(__file__), 'prompts.yaml')
        with open(yaml_path, 'r', encoding='utf-8') as f:
            _PROMPT_TEMPLATES = yaml.safe_load(f)
    return _PROMPT_TEMPLATES


def build_prompt(profile, history, candidates, local_errors, LONG_SKILLS, step_counter):
    """核心预测提示词（从 YAML 加载）"""
    templates = _load_templates()
    template = templates.get('build_prompt', '')

    # 提取所有变量（与原函数完全一致）
    seas = profile.get('seasonal_strength', 0)
    trend = profile.get('trend_strength', 0)
    period = profile.get('period', 12)
    local_slope = profile.get('local_slope', 0.0)
    change_detected = profile.get('change_point_detected', False)
    data_len = len(history)
    adf_pvalue = profile.get('adf_pvalue', 0.5)
    missing_rate = profile.get('missing_rate', 0.0)
    recent_volatility = profile.get('recent_volatility', 1.0)
    acf_peak_lag = profile.get('acf_peak_lag', 0)
    diff_adf_pvalue = profile.get('diff_adf_pvalue', 0.5)
    sample_entropy = profile.get('sample_entropy', 0.0)
    spectral_entropy = profile.get('spectral_entropy', 0.0)
    fft_peak_freq = profile.get('fft_peak_freq', 0.0)
    acf_365 = profile.get('acf_365', 0.0)
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
        error_hint = f"最低MAE技能: {best_name} ({best_mae:.6f}), 最高MAE技能: {worst_name} ({worst_mae:.6f}).\n"

    season_hint = ""
    if seas > 0.5:
        season_hint = f"序列季节性较强（{seas:.6f}）。建议选择适合季节性的技能，并可考虑多个技能加权组合。\n"

    calendar_hint = ""
    if seas > 0.5 and data_len >= 24 and profile.get('has_dates', False):
        calendar_hint = "日历技能（calendar）可作为辅助，与其他技能组合。\n"

    precision_hint = "请输出精确到4~6位小数的权重（如 0.7235, 0.1867, 0.0898），不要输出过多尾随零。"
    performance_hint = ""

    date_info = ""
    if profile.get('has_dates', False):
        date_info = f"- 当前时间点：{year}年{month}月 (Q{quarter})，{'月末' if is_month_end else '非月末'}，距起始 {days_from_start} 天\n"

    long_hint = ""
    if data_len > 400:
        rec_names = ", ".join(LONG_SKILLS)
        long_hint = f"💡 提示：对于长度>400的长序列，{rec_names} 等技能在多步预测上表现优异，可以分配较高权重（如0.5~0.8）。\n"

    rule_strategy = profile.get('rule_strategy', None)
    rule_hint = ""
    if rule_strategy:
        rule_hint = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🚨 【核心要求】你是一个独立的预测专家，必须独立思考！

参考策略（仅供参考，禁止照搬）：
{rule_strategy}

⭐ 你的输出要求：
1. ★ 你必须基于当前窗口的【实际特征】独立生成策略
2. ★ 禁止直接复制参考策略的权重和阶段结构
3. ★ 如果你的策略与参考策略相似度 > 70%，将被视为无效并惩罚
4. ★ 鼓励使用与参考策略【完全不同】的技能组合

📌 必须输出的字段 "relation_to_reference"：
   - "completely_different": ★ 必须选择此项！表示你完全独立生成了策略
   - "partially_referenced": 仅当你借鉴了非常少的思路（不推荐）
   - "adopted_with_modifications": 仅当你做了重大调整（不推荐）

⚠️ 记住：你是独立专家，不是复制机器！必须输出 completely_different！
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

    t = Template(template)
    return t.render(
        data_len=data_len,
        seas=f"{seas:.6f}",
        trend=f"{trend:.6f}",
        period=period,
        adf_pvalue=f"{adf_pvalue:.6f}",
        diff_adf_pvalue=f"{diff_adf_pvalue:.6f}",
        missing_rate=f"{missing_rate:.4f}",
        recent_volatility=f"{recent_volatility:.4f}",
        local_slope=f"{local_slope:.6f}",
        change_detected=change_detected,
        acf_peak_lag=acf_peak_lag,
        acf_365=f"{acf_365:.4f}",
        sample_entropy=f"{sample_entropy:.4f}",
        spectral_entropy=f"{spectral_entropy:.4f}",
        fft_peak_freq=f"{fft_peak_freq:.4f}",
        win_mean=f"{win_mean:.6f}",
        win_std=f"{win_std:.6f}",
        win_trend=win_trend,
        recent_str=recent_str,
        date_info=date_info,
        rule_hint=rule_hint,
        skill_info=skill_info,
        error_hint=error_hint,
        season_hint=season_hint,
        calendar_hint=calendar_hint,
        precision_hint=precision_hint,
        performance_hint=performance_hint,
        long_hint=long_hint,
    )


def build_preprocess_prompt(profile: dict, history: np.ndarray) -> str:
    templates = _load_templates()
    template = templates.get('build_preprocess_prompt', '')
    t = Template(template)
    return t.render(
        skewness=f"{profile.get('skewness', 0):.6f}",
        cv=f"{profile.get('cv', 0):.6f}",
        trend_strength=f"{profile.get('trend_strength', 0):.6f}",
        min_value=f"{np.min(history):.6f}",
        adf_pvalue=f"{profile.get('adf_pvalue', 0.5):.6f}"
    )


def build_post_enhance_prompt(profile: dict, residual_stats: dict, horizon: int) -> str:
    templates = _load_templates()
    template = templates.get('build_post_enhance_prompt', '')
    t = Template(template)
    return t.render(
        acf1=f"{residual_stats.get('acf_lag1', 0.0):.3f}",
        var_ratio=f"{residual_stats.get('var_ratio', 1.0):.3f}",
        horizon=horizon
    )


def build_strategy_generation_prompt(features: dict, trajectory: list, window_id: int, horizon: int) -> str:
    """
    策略归纳提示词（从 YAML 加载）
    完全保留旧版内容，只支持动态 horizon 替换
    """
    templates = _load_templates()
    template = templates.get('build_strategy_generation_prompt', '')

    feat_desc = [f"{k}: {v:.3f}" for k, v in features.items() if isinstance(v, (int, float))]
    traj_desc = []
    for step_info in trajectory[:10]:
        step = step_info.get('step', 0)
        weights = step_info.get('weights', {})
        w_str = ', '.join([f"{k}: {v:.4f}" for k, v in weights.items()])
        traj_desc.append(f"第{step}步: {{{w_str}}}")
    traj_summary = '\n'.join(traj_desc) if traj_desc else "无轨迹数据"

    t = Template(template)
    return t.render(
        feat_desc=', '.join(feat_desc),
        traj_summary=traj_summary,
        horizon=horizon
    )