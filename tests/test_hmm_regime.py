import numpy as np
import pandas as pd
import pytest

pytest.importorskip("hmmlearn")
from risk.hmm_regime import HMMRegimeDetector


def test_hmm_predict_and_proba_sum_to_one():
    rng = np.random.default_rng(123)
    low_vol = rng.normal(0.0008, 0.005, 160)
    high_vol = rng.normal(-0.0003, 0.02, 180)
    rets = np.concatenate([low_vol, high_vol])
    idx = pd.date_range('2020-01-01', periods=len(rets), freq='B')
    series = pd.Series(rets, index=idx)

    detector = HMMRegimeDetector(random_state=42)
    label = detector.predict(series)
    probs = detector.predict_proba(series)

    assert label in {'bull', 'neutral', 'bear'}
    assert pytest.approx(sum(probs.values()), rel=1e-6, abs=1e-6) == 1.0


def test_hmm_insufficient_history_raises():
    idx = pd.date_range('2023-01-01', periods=40, freq='B')
    series = pd.Series(np.linspace(-0.01, 0.01, len(idx)), index=idx)
    detector = HMMRegimeDetector(min_fit_observations=120)
    with pytest.raises(RuntimeError, match='insufficient history'):
        detector.predict(series)


def test_hmm_labels_consistent_for_same_input():
    rng = np.random.default_rng(7)
    rets = rng.normal(0.0005, 0.01, 320)
    idx = pd.date_range('2019-01-01', periods=len(rets), freq='B')
    series = pd.Series(rets, index=idx)

    detector = HMMRegimeDetector(random_state=42)
    first = detector.predict(series)
    second = detector.predict(series)
    assert first == second


def test_hmm_crash_segment_is_labeled_bear():
    rng = np.random.default_rng(99)
    calm1 = rng.normal(0.0008, 0.005, 150)
    crash = rng.normal(-0.0030, 0.030, 60)
    calm2 = rng.normal(0.0007, 0.005, 150)
    rets = np.concatenate([calm1, crash, calm2])
    idx = pd.date_range('2020-01-01', periods=len(rets), freq='B')
    series = pd.Series(rets, index=idx)

    detector = HMMRegimeDetector(random_state=42, n_states=3)
    labels = detector.decode_full_series(series)

    crash_idx = idx[len(calm1):len(calm1) + len(crash)]
    crash_labels = labels.reindex(crash_idx).dropna()
    assert not crash_labels.empty
    assert (crash_labels == 'bear').mean() >= 0.5


def test_hmm_refit_uses_elapsed_days_not_observation_delta():
    class _CountingDetector(HMMRegimeDetector):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.fit_calls = 0

        def fit(self, returns: pd.Series) -> None:
            self.fit_calls += 1
            return super().fit(returns)

    rng = np.random.default_rng(5)
    idx = pd.date_range('2024-01-01', periods=500, freq='h')
    series = pd.Series(rng.normal(0.00005, 0.002, len(idx)), index=idx)

    detector = _CountingDetector(
        n_states=3,
        min_fit_observations=120,
        refit_interval_days=10,
        min_state_mean_separation=0.0,
        random_state=42,
    )

    first = series.iloc[:260]      # ~11 days
    second = series.iloc[:360]     # many new observations, only ~4 more days
    third = series.iloc[:500]      # advance past 10-day threshold from initial fit

    _ = detector.predict(first)
    calls_after_first = detector.fit_calls
    _ = detector.predict(second)
    calls_after_second = detector.fit_calls
    _ = detector.predict(third)
    calls_after_third = detector.fit_calls

    assert calls_after_first >= 1
    assert calls_after_second == calls_after_first
    assert calls_after_third > calls_after_second


def test_decode_point_in_time_marks_pre_history_as_unknown():
    rng = np.random.default_rng(12)
    idx = pd.date_range('2022-01-01', periods=260, freq='B')
    series = pd.Series(rng.normal(0.0004, 0.01, len(idx)), index=idx)
    detector = HMMRegimeDetector(random_state=42, min_fit_observations=120)
    pit = detector.decode_point_in_time(series, min_history=120)

    assert (pit.iloc[:119] == 'unknown').all()
    assert pit.iloc[-1] in {'bull', 'neutral', 'bear', 'unknown'}


def test_hmm_non_separated_states_fall_back_to_neutral():
    idx = pd.date_range('2021-01-01', periods=300, freq='B')
    rng = np.random.default_rng(1234)
    series = pd.Series(rng.normal(0.0005, 1e-6, len(idx)), index=idx)
    detector = HMMRegimeDetector(random_state=42, min_fit_observations=120, min_state_mean_separation=5.0)

    label = detector.predict(series)
    probs = detector.predict_proba(series)

    assert label == 'neutral'
    assert probs.get('neutral', 0.0) == pytest.approx(1.0)
