from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

try:
    from hmmlearn.hmm import GaussianHMM
except ImportError as exc:  # pragma: no cover - exercised in runtime environments without dependency
    raise ImportError(
        "hmmlearn is required for HMMRegimeDetector. Install with `pip install hmmlearn>=0.3`."
    ) from exc


@dataclass
class HMMModelState:
    model: GaussianHMM
    state_map: Dict[int, str]
    fitted_observations: int


class HMMRegimeDetector:
    """Hidden Markov Model based market regime detector."""

    def __init__(
        self,
        n_states: int = 3,
        covariance_type: str = 'full',
        min_fit_observations: int = 120,
        refit_interval: int = 21,
        random_state: int = 42,
    ):
        self.n_states = int(n_states)
        self.covariance_type = str(covariance_type)
        self.min_fit_observations = int(min_fit_observations)
        self.refit_interval = int(refit_interval)
        self.random_state = int(random_state)
        self._state: Optional[HMMModelState] = None

    def is_fitted(self) -> bool:
        return self._state is not None

    @staticmethod
    def _build_observations(returns: pd.Series) -> pd.DataFrame:
        series = pd.Series(returns).dropna().astype(float)
        obs = pd.DataFrame(index=series.index)
        obs['daily_return'] = series
        obs['rolling_20d_vol'] = series.rolling(20).std()
        obs['rolling_60d_return'] = series.rolling(60).sum()

        z_window = 60
        for col in ['daily_return', 'rolling_20d_vol', 'rolling_60d_return']:
            mean = obs[col].rolling(z_window).mean()
            std = obs[col].rolling(z_window).std().replace(0.0, np.nan)
            obs[col] = (obs[col] - mean) / std

        return obs.dropna()

    def fit(self, returns: pd.Series) -> None:
        obs = self._build_observations(returns)
        if len(obs) < self.min_fit_observations:
            raise RuntimeError('insufficient history')

        model = GaussianHMM(
            n_components=self.n_states,
            covariance_type=self.covariance_type,
            random_state=self.random_state,
            n_iter=200,
        )
        model.fit(obs.values)

        hidden = model.predict(obs.values)
        obs_with_state = obs.copy()
        obs_with_state['state'] = hidden
        state_means = obs_with_state.groupby('state')['daily_return'].mean().sort_values()

        state_order = list(state_means.index)
        labels = ['bear', 'neutral', 'bull']
        if self.n_states != 3:
            labels = [f'state_{i}' for i in range(self.n_states)]
            if self.n_states >= 2:
                labels[0] = 'bear'
                labels[-1] = 'bull'
            if self.n_states >= 3:
                labels[len(labels) // 2] = 'neutral'

        state_map = {int(state): labels[idx] for idx, state in enumerate(state_order)}
        self._state = HMMModelState(model=model, state_map=state_map, fitted_observations=len(obs))

    def _ensure_fitted(self, returns: pd.Series) -> pd.DataFrame:
        obs = self._build_observations(returns)
        if len(obs) < self.min_fit_observations:
            raise RuntimeError('insufficient history')

        if self._state is None:
            self.fit(returns)
            obs = self._build_observations(returns)
            return obs

        obs_delta = max(0, len(obs) - int(self._state.fitted_observations))
        if obs_delta >= self.refit_interval:
            self.fit(returns)
            obs = self._build_observations(returns)
        return obs

    def predict(self, returns: pd.Series) -> str:
        obs = self._ensure_fitted(returns)
        hidden = int(self._state.model.predict(obs.values)[-1])
        return self._state.state_map.get(hidden, 'neutral')

    def predict_proba(self, returns: pd.Series) -> Dict[str, float]:
        obs = self._ensure_fitted(returns)
        probs = self._state.model.predict_proba(obs.values)[-1]
        mapped = {label: 0.0 for label in set(self._state.state_map.values())}
        for idx, prob in enumerate(probs):
            label = self._state.state_map.get(int(idx), f'state_{idx}')
            mapped[label] = float(prob)

        total = float(sum(mapped.values()))
        if total > 0:
            mapped = {k: float(v / total) for k, v in mapped.items()}
        return mapped

    def decode_full_series(self, returns: pd.Series) -> pd.Series:
        """Fit on full series and decode all labels in one pass (retrospective analytics)."""
        self.fit(returns)
        obs = self._build_observations(returns)
        hidden = self._state.model.predict(obs.values)
        labels = [self._state.state_map.get(int(state), 'neutral') for state in hidden]
        return pd.Series(labels, index=obs.index, dtype='object')
