import numpy as np
import pandas as pd
from scipy import stats
from .base import BaseSkill

class CalendarSkill(BaseSkill):
    def __init__(self):
        super().__init__()
        self.name = "calendar"
        self.description = "自适应日历同期预测（支持月度和日度数据）"
        self.min_data_points = 13
        self.requires_full_history = True
        self.strength_tags = ["season", "calendar"]
        self.model_family = "lightweight"
        self.required_features = ["seasonal_strength", "data_length", "period", "has_dates"]
        self.decision_hint = (
            "利用历史同月/同日数据的加权统计或趋势外推，计算极快。"
            "当季节性>0.5时强烈推荐作为稳健基准，建议分配10%~30%权重。"
        )
        self.state_card = {
            "when_to_use": {
                "conditions": [
                    {"field": "seasonal_strength", "op": ">", "value": 0.5},
                    {"field": "data_length", "op": ">=", "value": 13}
                ],
                "logic": "AND"
            },
            "when_not_to_use": {
                "conditions": [
                    {"field": "seasonal_strength", "op": "<", "value": 0.2},
                    {"field": "data_length", "op": "<", "value": 13}
                ],
                "logic": "OR"
            },
            "visible_cues": ["强年度周期"],
            "verification_cue": "预测值与历史同期值一致",
            "fallback_skill": "seasonal_naive"
        }

    def execute(self, history: np.ndarray, horizon: int, **kwargs) -> np.ndarray:
        dates = kwargs.get('dates', None)
        period = kwargs.get('period', 12)
        freq = kwargs.get('freq', None)

        if dates is None or len(history) < period:
            return self._seasonal_naive(history, horizon, period)

        if len(dates) != len(history):
            return self._seasonal_naive(history, horizon, period)

        if not isinstance(dates, pd.Series):
            try:
                dates = pd.Series(pd.to_datetime(dates))
            except:
                return self._seasonal_naive(history, horizon, period)

        is_daily = False
        if freq and freq.lower() in ('d', 'daily'):
            is_daily = True
        elif len(dates) > 10:
            diffs = (dates.iloc[1:11] - dates.iloc[:10]).dt.days if len(dates) > 10 else (dates.diff().dt.days.dropna())
            if len(diffs) > 0 and all(d == 1 for d in diffs[:10]):
                is_daily = True

        if is_daily:
            return self._daily_calendar(history, dates, horizon)
        else:
            return self._monthly_calendar(history, dates, horizon)

    def _daily_calendar(self, history, date_series, horizon):
        last_date = date_series.iloc[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon, freq='D')
        preds = []
        for future_date in future_dates:
            target_month_day = (future_date.month, future_date.day)
            mask = (date_series.dt.month == target_month_day[0]) & (date_series.dt.day == target_month_day[1])
            if len(mask) != len(history):
                preds.append(history[-1])
                continue
            vals = history[mask.values]
            if len(vals) == 0:
                preds.append(history[-1] if len(history) > 0 else 0.0)
            else:
                years = date_series[mask].dt.year.values
                current_year = last_date.year
                weights = np.exp(-0.3 * (current_year - years))
                weights /= weights.sum()
                preds.append(np.dot(vals, weights))
        return np.array(preds)

    def _monthly_calendar(self, history, date_series, horizon):
        last_date = date_series.iloc[-1]
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=horizon, freq='MS')
        future_months = future_dates.month.values
        preds = []
        for target_month in future_months:
            mask = date_series.dt.month == target_month
            if len(mask) != len(history):
                preds.append(history[-1])
                continue
            vals = history[mask.values]
            years = date_series[mask].dt.year.values
            if len(vals) == 0:
                preds.append(history[-(12 - target_month % 12)])
                continue
            if len(vals) == 1:
                preds.append(vals[0])
                continue
            pred = self._adaptive_predict(vals, years)
            preds.append(pred)
        return np.array(preds)

    def _adaptive_predict(self, values, years):
        if len(values) < 2:
            return values[0]
        mean_val = np.mean(values)
        std_val = np.std(values)
        if std_val > 0:
            keep = np.abs(values - mean_val) <= 2 * std_val
            values = values[keep]
            years = years[keep]
        if len(values) < 2:
            return values[0]
        x = years - years[0]
        if np.std(x) == 0:
            current_year = years[-1]
            year_diff = current_year - years
            weights = np.exp(-0.3 * year_diff)
            weights /= weights.sum()
            return float(np.dot(values, weights))
        slope, intercept, _, p_value, _ = stats.linregress(x, values)
        if len(values) >= 3 and p_value < 0.15:
            last_year = years[-1]
            future_x = (last_year + 1) - years[0]
            pred = intercept + slope * future_x
            pred = np.clip(pred, np.min(values) * 0.8, np.max(values) * 1.2)
            return float(pred)
        else:
            current_year = years[-1]
            year_diff = current_year - years
            weights = np.exp(-0.3 * year_diff)
            weights /= weights.sum()
            return float(np.dot(values, weights))

    def _seasonal_naive(self, history, horizon, period):
        if len(history) < period:
            return np.full(horizon, np.mean(history[-min(5, len(history)):]))
        preds = [history[-(period - i % period)] for i in range(horizon)]
        return np.array(preds)