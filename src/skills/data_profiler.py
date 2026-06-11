import numpy as np
from statsmodels.tsa.stattools import adfuller, acf
from scipy import stats as scipy_stats
import pandas as pd
from scipy.fft import fft, fftfreq

class DataProfiler:
    @staticmethod
    def _date_features(dates, n):
        if dates is None or len(dates) == 0:
            return {
                'has_dates': False, 'month_of_year': 0, 'year': 0,
                'quarter': 0, 'is_month_end': False,
                'days_from_start': n, 'time_since_last': 0
            }
        try:
            dts = pd.to_datetime(dates)
            months = int(dts.month.values[-1])
            years = int(dts.year.values[-1])
            quarter = (months - 1) // 3 + 1
            is_month_end = bool(dts[-1].is_month_end)
            days_from_start = int((dts[-1] - dts[0]).days)
            time_since_last = int((dts[-1] - dts[-2]).days) if n >= 2 else 0
            return {
                'has_dates': True, 'month_of_year': months, 'year': years,
                'quarter': quarter, 'is_month_end': is_month_end,
                'days_from_start': days_from_start, 'time_since_last': time_since_last
            }
        except:
            return {
                'has_dates': False, 'month_of_year': 0, 'year': 0,
                'quarter': 0, 'is_month_end': False,
                'days_from_start': n, 'time_since_last': 0
            }

    @staticmethod
    def _compute_multi_view_snapshots(history, period=12):
        n = len(history)
        views = {}
        if n >= 10:
            ma = pd.Series(history).rolling(window=min(7, n//2), center=True, min_periods=2).mean().bfill().values
            trend_component = ma - ma[0]
            t_min, t_max = np.min(trend_component), np.max(trend_component)
            if t_max > t_min:
                trend_norm = (trend_component - t_min) / (t_max - t_min)
            else:
                trend_norm = np.zeros_like(trend_component)
            views['trend_view'] = trend_norm.tolist()
        else:
            views['trend_view'] = list(history)

        if n >= 2 * period:
            seasonal_diff = history[period:] - history[:-period]
            s_min, s_max = np.min(seasonal_diff), np.max(seasonal_diff)
            if s_max > s_min:
                season_norm = (seasonal_diff - s_min) / (s_max - s_min)
            else:
                season_norm = np.zeros_like(seasonal_diff)
            views['seasonal_view'] = season_norm.tolist()
        else:
            views['seasonal_view'] = list(history)
        return views

    @staticmethod
    def _multi_resolution_profile(history, resample_factors=[7, 30]):
        profiles = {}
        for factor in resample_factors:
            if len(history) >= 2 * factor:
                n_full = (len(history) // factor) * factor
                if n_full >= factor:
                    downsampled = np.mean(history[:n_full].reshape(-1, factor), axis=1)
                    d_min, d_max = np.min(downsampled), np.max(downsampled)
                    if d_max > d_min:
                        downsampled = (downsampled - d_min) / (d_max - d_min)
                    profiles[f'resample_{factor}'] = downsampled.tolist()
        return profiles

    @staticmethod
    def _sample_entropy(series, m=2, r=None):
        """计算样本熵 (Sample Entropy) 衡量序列复杂度"""
        n = len(series)
        if n < m + 1:
            return 0.0
        if r is None:
            r = 0.2 * np.std(series)
        if r == 0:
            return 0.0

        def _maxdist(xi, xj):
            return max(abs(a - b) for a, b in zip(xi, xj))

        def _phi(m):
            patterns = [tuple(series[i:i+m]) for i in range(n - m + 1)]
            counts = 0
            for i in range(len(patterns)):
                for j in range(i+1, len(patterns)):
                    if _maxdist(patterns[i], patterns[j]) <= r:
                        counts += 1
            return counts / ((n - m + 1) * (n - m) / 2) if n - m + 1 > 1 else 0.0

        phi_m = _phi(m)
        phi_m1 = _phi(m+1)
        if phi_m == 0 or phi_m1 == 0:
            return 0.0
        return -np.log(phi_m1 / phi_m)

    @staticmethod
    def _spectral_entropy(series, fs=1.0):
        """频谱熵：衡量频率分布的信息熵"""
        n = len(series)
        if n < 10:
            return 0.0
        fft_vals = fft(series)
        power = np.abs(fft_vals[:n//2]) ** 2
        power_norm = power / (power.sum() + 1e-8)
        entropy = -np.sum(power_norm * np.log(power_norm + 1e-8))
        return entropy / np.log(len(power_norm)) if len(power_norm) > 1 else 0.0

    @staticmethod
    def profile(history: np.ndarray, window_sizes: list = None,
                auto_period: bool = True, freq: str = None, dates=None) -> dict:
        if window_sizes is None:
            window_sizes = [len(history)]
        base = DataProfiler._statistical_profile(history, auto_period, freq, dates)
        date_feat = DataProfiler._date_features(dates, len(history))
        base.update(date_feat)

        snapshots = {}
        for w in window_sizes:
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
        base['snapshots'] = snapshots
        base['snapshot_vector'] = np.concatenate([np.array(v) for v in snapshots.values()]).tolist() if snapshots else []

        period = base.get('period', 12)
        multi_views = DataProfiler._compute_multi_view_snapshots(history, period)
        base['trend_snapshot'] = multi_views.get('trend_view', [])
        base['seasonal_snapshot'] = multi_views.get('seasonal_view', [])

        multi_res = DataProfiler._multi_resolution_profile(history)
        base.update(multi_res)

        return base

    @staticmethod
    def profile_selected(history: np.ndarray, feature_names: list,
                         auto_period: bool = True, freq: str = None, dates=None) -> dict:
        n = len(history)
        result = {
            'adf_pvalue': 0.5, 'seasonal_strength': 0.0, 'trend_strength': 0.0,
            'missing_rate': 0.0, 'data_length': n, 'recent_volatility': 0.0,
            'period': 7, 'local_slope': 0.0, 'change_point_detected': False,
            'snapshots': {}, 'snapshot_vector': [],
            'trend_snapshot': [], 'seasonal_snapshot': [],
            'has_dates': False, 'month_of_year': 0, 'year': 0,
            'quarter': 0, 'is_month_end': False,
            'days_from_start': n, 'time_since_last': 0,
            # 新增特征默认值
            'acf_peak_lag': 0, 'diff_adf_pvalue': 0.5, 'sample_entropy': 0.0,
            'spectral_entropy': 0.0, 'fft_peak_freq': 0.0, 'acf_365': 0.0
        }

        need_basic = any(f in feature_names for f in ['adf_pvalue','seasonal_strength','trend_strength',
                                                       'recent_volatility','local_slope','change_point_detected',
                                                       'acf_peak_lag','diff_adf_pvalue','sample_entropy',
                                                       'spectral_entropy','fft_peak_freq','acf_365'])
        if need_basic:
            base = DataProfiler._statistical_profile(history, auto_period, freq, dates)
            for k, v in base.items():
                if k in result:
                    result[k] = v

        if 'has_dates' in feature_names or any(f in feature_names for f in ['month_of_year','year','quarter','is_month_end','days_from_start','time_since_last']):
            date_feat = DataProfiler._date_features(dates, n)
            for k, v in date_feat.items():
                result[k] = v

        if 'snapshots' in feature_names or 'snapshot_vector' in feature_names:
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
            result['snapshots'] = snapshots
            result['snapshot_vector'] = np.concatenate([np.array(v) for v in snapshots.values()]).tolist() if snapshots else []

        if 'trend_snapshot' in feature_names or 'seasonal_snapshot' in feature_names:
            period = result.get('period', 12)
            multi_views = DataProfiler._compute_multi_view_snapshots(history, period)
            result['trend_snapshot'] = multi_views.get('trend_view', [])
            result['seasonal_snapshot'] = multi_views.get('seasonal_view', [])

        if 'period' in feature_names and not need_basic:
            result['period'] = DataProfiler._auto_period(history, freq, dates)

        if any('resample' in f for f in feature_names):
            multi_res = DataProfiler._multi_resolution_profile(history)
            for k, v in multi_res.items():
                result[k] = v

        return result

    @staticmethod
    def _statistical_profile(history, auto_period=True, freq=None, dates=None):
        n = len(history)
        if n < 10:
            return {'adf_pvalue':0.5, 'seasonal_strength':0.0, 'trend_strength':0.0,
                    'missing_rate':0.0, 'data_length':n, 'recent_volatility':0.0,
                    'period':7, 'local_slope':0.0, 'change_point_detected':False,
                    'acf_peak_lag':0, 'diff_adf_pvalue':0.5, 'sample_entropy':0.0,
                    'spectral_entropy':0.0, 'fft_peak_freq':0.0, 'acf_365':0.0}

        try:
            adf_p = adfuller(history, maxlag=min(12, n//2))[1]
        except:
            adf_p = 0.5

        period = DataProfiler._auto_period(history, freq, dates) if auto_period else 7

        if n >= 2*period:
            seasonal_diff = history[period:] - history[:-period]
            seas = max(0.0, 1 - np.std(seasonal_diff) / (np.std(history)+1e-8))
        else:
            seas = 0.0

        if adf_p > 0.95:
            seas = max(0.0, seas - 0.2)

        diffs = np.diff(history)
        trend = abs(np.mean(diffs>0)-0.5)*2 if len(diffs)>0 else 0.0
        recent_vol = np.std(history[-5:])/np.std(history) if n>=5 and np.std(history)>0 else 1.0

        if n >= 5:
            x = np.arange(5); y = history[-5:]
            slope, _, _, _, _ = scipy_stats.linregress(x, y)
        else:
            slope = 0.0

        if n >= 20:
            recent_var = np.var(history[-20:])
            total_var = np.var(history)
            change_point_detected = (total_var > 0 and recent_var / total_var > 3.0)
        else:
            change_point_detected = False

        if n >= 30:
            tau, _ = scipy_stats.kendalltau(np.arange(n), history)
            trend_strength = abs(tau)
        else:
            trend_strength = trend

        # ========== 新增特征计算 ==========
        # 1. ACF 峰值 lag
        acf_peak_lag = 0
        if n >= 20:
            try:
                acf_vals = acf(history, nlags=min(50, n//2))
                # 排除 lag=0，寻找第一个显著峰值 (acf > 0.2 且大于邻居)
                peaks = [i for i in range(2, len(acf_vals)-1) if acf_vals[i] > acf_vals[i-1] and acf_vals[i] > acf_vals[i+1] and acf_vals[i] > 0.2]
                if peaks:
                    acf_peak_lag = peaks[0]
                else:
                    # 取最大自相关对应的 lag
                    best_lag = np.argmax(acf_vals[1:]) + 1
                    if acf_vals[best_lag] > 0.2:
                        acf_peak_lag = best_lag
            except:
                pass

        # 2. 一阶差分后的 ADF p-value
        diff_adf_pvalue = 0.5
        if n > 5:
            try:
                diff_series = np.diff(history)
                diff_adf_pvalue = adfuller(diff_series, maxlag=min(12, len(diff_series)//2))[1]
            except:
                pass

        # 3. 样本熵 (复杂度)
        sample_entropy = 0.0
        if n >= 30:
            try:
                sample_entropy = DataProfiler._sample_entropy(history, m=2)
            except:
                pass

        # 4. 频谱熵
        spectral_entropy = 0.0
        if n >= 30:
            try:
                spectral_entropy = DataProfiler._spectral_entropy(history)
            except:
                pass

        # 5. FFT 峰值频率 (归一化)
        fft_peak_freq = 0.0
        if n >= 30:
            try:
                fft_vals = np.abs(fft(history))
                freqs = fftfreq(n, d=1)[:n//2]
                power = fft_vals[:n//2] ** 2
                if len(power) > 0 and np.max(power) > 0:
                    peak_idx = np.argmax(power)
                    fft_peak_freq = freqs[peak_idx] if freqs[peak_idx] > 0 else 0.0
            except:
                pass

        # 6. 年周期自相关 (365 lag)
        acf_365 = 0.0
        if n >= 730 and (freq and freq.lower() in ('d', 'daily')) or period == 365:
            try:
                acf_365 = np.corrcoef(history[365:], history[:-365])[0, 1]
            except:
                pass

        return {
            'adf_pvalue': adf_p, 'seasonal_strength': seas, 'trend_strength': trend_strength,
            'missing_rate': np.mean(np.isnan(history)), 'data_length': n,
            'recent_volatility': recent_vol, 'period': period,
            'local_slope': round(slope, 4),
            'change_point_detected': change_point_detected,
            'acf_peak_lag': acf_peak_lag,
            'diff_adf_pvalue': diff_adf_pvalue,
            'sample_entropy': sample_entropy,
            'spectral_entropy': spectral_entropy,
            'fft_peak_freq': fft_peak_freq,
            'acf_365': acf_365
        }

    @staticmethod
    def _auto_period(history, freq=None, dates=None):
        n = len(history)
        if n < 20:
            return 7

        # 1. 如果有日期数据，自动判断是否为日度数据（间隔1天）
        if dates is not None and len(dates) > 10:
            try:
                dts = pd.to_datetime(dates)
                diffs = (dts[1:11] - dts[:10]).days if len(dts) > 10 else (dts[1:] - dts[:-1]).days
                if len(diffs) > 0 and all(d == 1 for d in diffs[:10]):
                    if n >= 365:
                        return 365
                    else:
                        return 7
            except:
                pass

        # 2. 根据频率参数判断
        if freq:
            freq_lower = freq.lower()
            if freq_lower in ('d', 'daily'):
                if n >= 730:
                    try:
                        acf_365 = np.corrcoef(history[365:], history[:-365])[0, 1]
                        if acf_365 > 0.3:
                            return 365
                    except:
                        pass
                if n >= 14:
                    try:
                        acf_7 = np.corrcoef(history[7:], history[:-7])[0, 1]
                        if acf_7 > 0.3:
                            return 7
                    except:
                        pass
                return 365 if n >= 365 else 7
            elif freq_lower in ('m', 'monthly'):
                return 12

        # 3. 无频率信息时，基于自相关检测
        try:
            max_lag = min(n//2, 50)
            acf_vals = acf(history, nlags=max_lag)
            peaks = [i for i in range(2, len(acf_vals)-1)
                     if acf_vals[i] > acf_vals[i-1] and acf_vals[i] > acf_vals[i+1] and acf_vals[i] > 0.2]
            if peaks:
                return peaks[0]
            if n >= 365:
                try:
                    acf_365 = np.corrcoef(history[365:], history[:-365])[0, 1]
                    if acf_365 > 0.2:
                        return 365
                except:
                    pass
            best_lag = np.argmax(acf_vals[1:]) + 1
            if acf_vals[best_lag] > 0.2:
                return best_lag
        except:
            pass
        return 7

    @staticmethod
    def compute_dtw_distance(vec1, vec2):
        if len(vec1)==0 or len(vec2)==0:
            return float('inf')
        n1, n2 = len(vec1), len(vec2)
        max_len = 200
        if n1 > max_len:
            indices = np.linspace(0, n1-1, max_len, dtype=int)
            vec1 = np.array(vec1)[indices]
            n1 = max_len
        if n2 > max_len:
            indices = np.linspace(0, n2-1, max_len, dtype=int)
            vec2 = np.array(vec2)[indices]
            n2 = max_len
        idx1 = np.linspace(0, n1-1, min(n1,n2))
        idx2 = np.linspace(0, n2-1, min(n1,n2))
        v1 = np.interp(idx1, np.arange(n1), vec1)
        v2 = np.interp(idx2, np.arange(n2), vec2)
        return np.sqrt(np.mean((v1-v2)**2))