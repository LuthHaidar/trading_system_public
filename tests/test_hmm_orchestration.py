from collections import deque

import numpy as np
import pandas as pd

from strategies.strategy_orchestration import StrategyOrchestrationStrategy


class _DummySubStrategy:
    def get_required_history(self):
        return 1

    def generate_signals(self, date, data, current_positions):
        _ = (date, data, current_positions)
        return {'AAA': 1.0}

    def get_signal_metadata(self):
        return {'AAA': {'signal_strength': 0.8}}


def _build_orchestration_strategy() -> StrategyOrchestrationStrategy:
    strategy = StrategyOrchestrationStrategy(
        config={
            '_all_strategies_config': {},
            'base_weights': {'dummy': 1.0},
            'benchmark_ticker': 'SPY',
            'performance_lookback': 20,
            'hmm_n_states': 3,
            'hmm_covariance_type': 'full',
        }
    )
    strategy.sub_strategies = {'dummy': _DummySubStrategy()}
    strategy.sub_strategy_names = ['dummy']
    strategy.performance_history = {'dummy': deque(maxlen=20)}
    strategy.base_weights = {'dummy': 1.0}
    return strategy


def test_hmm_orchestration_generates_non_empty_signals_on_long_series():
    strategy = _build_orchestration_strategy()

    idx = pd.date_range('2022-01-01', periods=500, freq='B')
    rng = np.random.default_rng(42)
    returns = rng.normal(0.0005, 0.012, len(idx))
    close = 100.0 * np.cumprod(1.0 + returns)

    data = {
        'SPY': pd.DataFrame({'Close': close}, index=idx),
        'AAA': pd.DataFrame({'Close': close * 0.95}, index=idx),
    }

    signals = strategy.generate_signals(idx[-1], data, current_positions={})

    assert signals
    assert 'AAA' in signals
    assert signals['AAA'] > 0


def test_hmm_orchestration_short_series_logs_fallback_warning():
    strategy = _build_orchestration_strategy()

    # Force HMM path to raise insufficient history so fallback is exercised
    # irrespective of local hmmlearn availability specifics.
    class _Detector:
        pass

    strategy.regime_detector = _Detector()

    idx = pd.date_range('2024-01-01', periods=50, freq='B')
    close = np.linspace(100.0, 104.0, len(idx))
    data = {'SPY': pd.DataFrame({'Close': close}, index=idx)}

    import risk.advanced_risk as ar

    original_hmm = ar.AdvancedRiskAnalytics.hmm_regime_scaler
    original_heuristic = ar.AdvancedRiskAnalytics.regime_based_scaler

    try:
        ar.AdvancedRiskAnalytics.hmm_regime_scaler = staticmethod(
            lambda market_returns, detector, bull_scale=1.1, bear_scale=0.7, high_vol_scale=0.8: (_ for _ in ()).throw(RuntimeError('insufficient history'))
        )
        ar.AdvancedRiskAnalytics.regime_based_scaler = staticmethod(
            lambda returns, vol_threshold=0.25, bear_return_threshold=0.0: {'regime': 'bull', 'scale': 1.1}
        )

        logger_name = strategy.logger.name if hasattr(strategy, 'logger') else 'strategies.strategy_orchestration'
        from unittest import TestCase
        tc = TestCase()
        with tc.assertLogs(logger_name, level='WARNING') as logs:
            condition = strategy._detect_market_condition(idx[-1], data)

        assert condition == 'bull'
        assert any('insufficient history' in msg for msg in logs.output)
    finally:
        ar.AdvancedRiskAnalytics.hmm_regime_scaler = original_hmm
        ar.AdvancedRiskAnalytics.regime_based_scaler = original_heuristic


def test_regime_based_scaler_never_returns_high_vol_label():
    import risk.advanced_risk as ar

    low_vol = pd.Series(np.full(80, 0.0005))
    high_vol = pd.Series(np.concatenate([np.zeros(60), np.array([0.2, -0.2] * 10)]))

    out_low = ar.AdvancedRiskAnalytics.regime_based_scaler(low_vol)
    out_high = ar.AdvancedRiskAnalytics.regime_based_scaler(high_vol, vol_threshold=0.01)

    assert out_low['regime'] in {'bull', 'neutral', 'bear'}
    assert out_high['regime'] in {'bull', 'neutral', 'bear'}
    assert out_high['regime'] != 'high_vol'


def test_orchestration_asserts_on_unrecognized_regime_label():
    strategy = _build_orchestration_strategy()
    strategy.regime_detector = None

    idx = pd.date_range('2024-01-01', periods=30, freq='B')
    close = np.linspace(100.0, 101.0, len(idx))
    data = {'SPY': pd.DataFrame({'Close': close}, index=idx)}

    import risk.advanced_risk as ar
    original_heuristic = ar.AdvancedRiskAnalytics.regime_based_scaler
    try:
        ar.AdvancedRiskAnalytics.regime_based_scaler = staticmethod(
            lambda returns, vol_threshold=0.25, bear_return_threshold=0.0: {'regime': 'unexpected', 'scale': 1.0}
        )
        from unittest import TestCase
        tc = TestCase()
        with tc.assertRaisesRegex(AssertionError, "Unrecognized regime label"):
            strategy._detect_market_condition(idx[-1], data)
    finally:
        ar.AdvancedRiskAnalytics.regime_based_scaler = original_heuristic
