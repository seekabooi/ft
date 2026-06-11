import numpy as np

def build_prompt(profile, history, candidates, local_errors, LONG_SKILLS, step_counter):
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
        mae_str = f"{mae:.10f}" if mae is not None else "未计算"
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
        error_hint = f"最低MAE技能: {best_name} ({best_mae:.10f}), 最高MAE技能: {worst_name} ({worst_mae:.10f})。\n"

    season_hint = ""
    if seas > 0.5:
        season_hint = f"序列季节性较强（{seas:.10f}）。建议选择适合季节性的技能，并可考虑多个技能加权组合。\n"

    calendar_hint = ""
    if seas > 0.5 and data_len >= 24 and profile.get('has_dates', False):
        calendar_hint = "日历技能（calendar）可作为辅助，与其他技能组合。\n"

    precision_hint = "请输出精确到十位小数的权重（如 0.7234567890, 0.1867234567）。"

    performance_hint = ""

    date_info = ""
    if profile.get('has_dates', False):
        date_info = f"- 当前时间点：{year}年{month}月 (Q{quarter})，{'月末' if is_month_end else '非月末'}，距起始 {days_from_start} 天\n"

    long_hint = ""
    if data_len > 400:
        rec_names = ", ".join(LONG_SKILLS)
        long_hint = f"💡 提示：对于长度>400的长序列，{rec_names} 等技能在多步预测上表现优异，可以分配较高权重（如0.5~0.8）。\n"

    prompt = f"""你是时间序列预测专家。请根据以下特征决定最佳预测方案，使用 1~3 个技能加权组合（允许单技能，也鼓励多技能组合）。

序列特征：
- 长度:{data_len}，季节强度:{seas:.10f}，趋势强度:{trend:.10f}，周期:{period}
- 平稳性(ADF p-value):{adf_pvalue:.6f} (越小越平稳)
- 差分后平稳性(1阶差分 ADF):{diff_adf_pvalue:.6f}
- 缺失率:{missing_rate:.4f}
- 近期波动比(最近5点/整体):{recent_volatility:.4f}
- 局部斜率:{local_slope:.10f}，突变:{change_detected}
- 自相关峰值 lag:{acf_peak_lag}，年自相关(365):{acf_365:.4f}
- 样本熵(复杂度):{sample_entropy:.4f}，频谱熵:{spectral_entropy:.4f}
- 主频(FFT峰值):{fft_peak_freq:.4f} (0~0.5，高频表示短期波动)
- 近期5点均值:{win_mean:.10f}，波动:{win_std:.10f}，走势:{win_trend}
- 最近10点: {recent_str}
{date_info}
候选技能对比：
{skill_info}
{error_hint}{season_hint}{calendar_hint}{precision_hint}{performance_hint}
{long_hint}

要求：
1. 输出 JSON 格式，包含两个字段：
   - "skill_weights": 技能名称到权重的字典，权重十位小数，总和为1。
   - "replan_interval": 整数 (1~5)，表示多少步后需要强制重新决策（基于当前序列的稳定性，步数越短重决策越频繁）。
2. 输出示例：{{"skill_weights": {{"multi_resolution": 0.7, "calendar": 0.3}}, "replan_interval": 3}}
3. 不要输出任何解释。"""
    return prompt