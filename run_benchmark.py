import argparse
import os
from datetime import datetime
import pandas as pd
from src.agents.llm_planner import LLMPlannerAgent
from src.skills.registry import SkillRegistry
from src.skills.naive import NaiveSkill
from src.skills.seasonal_naive import SeasonalNaiveSkill
from src.skills.prophet_skill import ProphetSkill
from src.skills.auto_arima import AutoARIMASkill
from src.skills.naive_drift import NaiveDriftSkill
from src.skills.residual_correction import ResidualCorrectionSkill
from src.skills.local_drift import LocalDriftSkill
from src.skills.ets import ETSSkill
from src.skills.theta import ThetaSkill
from src.skills.holt_winters import HoltWintersSkill
from src.skills.croston import CrostonSkill
from src.skills.tbats import TBATSSkill
from src.skills.calendar_skill import CalendarSkill
from src.skills.fourier_skill import FourierSkill
from src.skills.multi_seasonal_naive import MultiSeasonalNaiveSkill
from src.evaluation.fixed_origin_evaluator import FixedOriginEvaluator
from src.skills.detrender import DetrenderSkill
from src.skills.seasonal_extractor import SeasonalExtractorSkill
from src.skills.trend_forecaster import TrendForecasterSkill
from src.skills.seasonal_forecaster import SeasonalForecasterSkill
from src.skills.bias_corrector import BiasCorrectorSkill
from src.skills.progressive_adaptive_combiner import ProgressiveAdaptiveCombiner
from src.skills.stl_decompose_skill import STLDecomposeSkill
from src.skills.chunk_ensemble import ChunkEnsembleSkill
from src.skills.multi_resolution import MultiResolutionSkill
from src.skills.residual_correction_advanced import ResidualCorrectionAdvancedSkill
from src.skills.adaptive_weighted_ensemble import AdaptiveWeightedEnsemble
from src.skills.fft_filter import FFTFilterSkill

# 移除可能导致问题的技能（如 incremental_gbm 需要 lightgbm）
try:
    from src.skills.incremental_gbm import IncrementalGBMSkill
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

DATASETS = ['airline_passengers', 'gold_price', 'champagne_sales', 'sunspots', 'melbourne_temp']


def build_full_registry():
    registry = SkillRegistry()

    # 基础技能
    naive = NaiveSkill()
    seasonal_naive = SeasonalNaiveSkill(period=12)
    prophet = ProphetSkill()
    auto_arima = AutoARIMASkill()
    naive_drift = NaiveDriftSkill()
    residual_corr = ResidualCorrectionSkill(base_skill=auto_arima)
    local_drift = LocalDriftSkill(window=5)
    ets = ETSSkill()
    theta = ThetaSkill()
    hw = HoltWintersSkill(period=12)
    croston = CrostonSkill()
    tbats = TBATSSkill()
    calendar_skill = CalendarSkill()
    fourier = FourierSkill(period=12)
    multi_sea = MultiSeasonalNaiveSkill(period=12)

    # 分解/组合技能
    detrender = DetrenderSkill()
    seasonal_extractor = SeasonalExtractorSkill(period=12)
    trend_forecaster = TrendForecasterSkill()
    seasonal_forecaster = SeasonalForecasterSkill(period=12)
    bias_corrector = BiasCorrectorSkill()
    progressive_combiner = ProgressiveAdaptiveCombiner()
    stl = STLDecomposeSkill()

    # 长序列专用技能
    chunk_ensemble = ChunkEnsembleSkill()
    multi_res = MultiResolutionSkill()
    residual_adv = ResidualCorrectionAdvancedSkill()
    adaptive_ensemble = AdaptiveWeightedEnsemble(skills=[naive, seasonal_naive, hw, calendar_skill])
    fft_filter = FFTFilterSkill()
    incremental_gbm = IncrementalGBMSkill() if HAS_LGB else None

    all_skills = [
        naive, seasonal_naive, prophet, auto_arima,
        naive_drift, residual_corr, local_drift,
        ets, theta, hw, croston, tbats, calendar_skill, fourier, multi_sea,
        detrender, seasonal_extractor, trend_forecaster, seasonal_forecaster,
        bias_corrector, progressive_combiner, stl,
        chunk_ensemble, multi_res, residual_adv, adaptive_ensemble, fft_filter
    ]
    if incremental_gbm:
        all_skills.append(incremental_gbm)

    for s in all_skills:
        registry.register(s)

    return registry, all_skills


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--model', default='glm-4')
    parser.add_argument('--min_train_size', type=int, default=132)
    parser.add_argument('--horizon', type=int, default=12)
    parser.add_argument('--data_ratio', type=float, default=1.0)
    parser.add_argument('--no_skills', action='store_true')
    parser.add_argument('--skill_mode', choices=['branch', 'single', 'ensemble'], default='branch')
    parser.add_argument('--skill_name', type=str, default=None)
    parser.add_argument('--llm_call_interval', type=int, default=1)
    args = parser.parse_args()

    os.makedirs('storage/logs', exist_ok=True)
    datasets = [args.dataset] if args.dataset else DATASETS

    for ds in datasets:
        log_file = f'storage/logs/agent_{ds}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

        full_registry, all_skills = build_full_registry()
        use_skills = not args.no_skills

        if args.skill_mode == 'single':
            target = full_registry.get(args.skill_name)
            if not target:
                print(f"❌ 技能 '{args.skill_name}' 不存在")
                return
            registry = SkillRegistry()
            registry.register(target)
        elif args.skill_mode == 'ensemble':
            from src.skills.ensemble import EnsembleSkill
            registry = SkillRegistry()
            registry.register(EnsembleSkill(skills=all_skills))
        else:
            registry = full_registry

        agent = LLMPlannerAgent(
            model=args.model,
            skill_registry=registry,
            log_file=log_file,
            use_skills=use_skills,
            llm_call_interval=args.llm_call_interval
        )
        evaluator = FixedOriginEvaluator(
            agent,
            min_train_size=args.min_train_size,
            horizon=args.horizon,
            data_ratio=args.data_ratio
        )

        print(f"\n▶ 评估 {ds}")
        try:
            res = evaluator.evaluate(ds)
            evaluator.print_report(res)
            df = pd.DataFrame({'prediction': res.get('predictions', []),
                               'actual': res.get('actuals', [])})
            os.makedirs('storage', exist_ok=True)
            df.to_csv(f'storage/eval_{ds}.csv', index=False)
            print(f"📁 保存至 storage/eval_{ds}.csv")
        except Exception as e:
            print(f"❌ {ds} 失败: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()