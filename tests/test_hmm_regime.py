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
