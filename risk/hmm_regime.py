from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

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
    last_fit_date: Optional[pd.Timestamp] = None


class HMMRegimeDetector:
    """Hidden Markov Model based market regime detector."""

    def __init__(
        self,
        n_states: int = 3,
        covariance_type: str = 'full',
        min_fit_observations: int = 120,
        refit_interval_days: int = 21,
        refit_interval: Optional[int] = None,
        min_state_mean_separation: float = 0.5,
        random_state: int = 42,
    ):
        self.n_states = int(n_states)
        self.covariance_type = str(covariance_type)
        self.min_fit_observations = int(min_fit_observations)
        # `refit_interval` kept for backward compatibility; prefer `refit_interval_days`.
        if refit_interval is not None:
            refit_interval_days = refit_interval
        self.refit_interval_days = int(refit_interval_days)
        self.min_state_mean_separation = float(min_state_mean_separation)
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
        raw_returns = pd.Series(returns).dropna().astype(float).reindex(obs_with_state.index)
        raw_vol = raw_returns.rolling(20).std().reindex(obs_with_state.index)
        labeled = pd.DataFrame(
            {
                'state': hidden,
                'raw_return': raw_returns.values,
                'raw_vol': raw_vol.values,
            },
            index=obs_with_state.index,
        ).dropna()
        state_return_means = labeled.groupby('state')['raw_return'].mean()
        state_vol_means = labeled.groupby('state')['raw_vol'].mean()
        states_present = [int(s) for s in state_return_means.index.tolist()]

        state_map: Dict[int, str] = {}
        if len(states_present) == 1:
            state_map[states_present[0]] = 'neutral'
        elif self.n_states == 2 or len(states_present) == 2:
            bull_state = int(state_return_means.idxmax())
            for s in states_present:
                state_map[int(s)] = 'bull' if int(s) == bull_state else 'bear'
        else:
            bull_state = int(state_return_means.idxmax())
            state_map[bull_state] = 'bull'

            remaining = [s for s in states_present if s != bull_state]
            negative_remaining = [s for s in remaining if float(state_return_means.get(s, 0.0)) < 0.0]
            bear_pool = negative_remaining or remaining
            bear_state = int(max(bear_pool, key=lambda s: float(state_vol_means.get(s, -np.inf))))
            state_map[bear_state] = 'bear'

            for s in remaining:
                if s != bear_state:
                    state_map[int(s)] = 'neutral'

        sep_vals = []
        ordered_returns = [float(state_return_means[s]) for s in states_present]
        for i in range(len(ordered_returns)):
            for j in range(i + 1, len(ordered_returns)):
                sep_vals.append(abs(ordered_returns[i] - ordered_returns[j]))
        max_sep = max(sep_vals) if sep_vals else 0.0

        within_state_vars = []
        for state_id in states_present:
            state_raw = labeled[labeled['state'] == state_id]['raw_return']
            if len(state_raw) > 1:
                within_state_vars.append(float(state_raw.var()))
        pooled_std = float(np.sqrt(np.mean(within_state_vars))) if within_state_vars else 0.0
        threshold = float(self.min_state_mean_separation * pooled_std)
        if len(states_present) > 1 and max_sep < threshold:
            logger.warning(
                "HMM states are not well-separated (max_sep=%.4f, threshold=%.4f); labels may be unreliable",
                max_sep,
                threshold,
            )
            self._state = None
            return

        last_fit_date = pd.Timestamp(pd.Series(returns).dropna().index[-1])
        self._state = HMMModelState(
            model=model,
            state_map=state_map,
            fitted_observations=len(obs),
            last_fit_date=last_fit_date,
        )

    def _ensure_fitted(self, returns: pd.Series) -> pd.DataFrame:
        obs = self._build_observations(returns)
        if len(obs) < self.min_fit_observations:
            raise RuntimeError('insufficient history')

        if self._state is None:
            self.fit(returns)
            if self._state is None:
                raise RuntimeError('hmm states not well-separated')
            obs = self._build_observations(returns)
            return obs

        current_date = pd.Timestamp(pd.Series(returns).dropna().index[-1])
        last_fit_date = pd.Timestamp(self._state.last_fit_date) if self._state.last_fit_date is not None else current_date
        elapsed_days = max(0, int((current_date - last_fit_date).days))
        if elapsed_days >= self.refit_interval_days:
            self.fit(returns)
            if self._state is None:
                raise RuntimeError('hmm states not well-separated')
            obs = self._build_observations(returns)
        return obs

    def predict(self, returns: pd.Series) -> str:
        try:
            obs = self._ensure_fitted(returns)
            hidden = int(self._state.model.predict(obs.values)[-1])
            return self._state.state_map.get(hidden, 'neutral')
        except RuntimeError as exc:
            if 'not well-separated' in str(exc):
                return 'neutral'
            raise

    def predict_proba(self, returns: pd.Series) -> Dict[str, float]:
        try:
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
        except RuntimeError as exc:
            if 'not well-separated' in str(exc):
                return {'neutral': 1.0}
            raise

    def decode_full_series(self, returns: pd.Series) -> pd.Series:
        """Fit on full series and decode all labels in one pass (retrospective analytics)."""
        self.fit(returns)
        obs = self._build_observations(returns)
        if self._state is None:
            return pd.Series(['neutral'] * len(obs), index=obs.index, dtype='object')
        hidden = self._state.model.predict(obs.values)
        labels = [self._state.state_map.get(int(state), 'neutral') for state in hidden]
        return pd.Series(labels, index=obs.index, dtype='object')

    def decode_point_in_time(self, returns: pd.Series, min_history: int = 120) -> pd.Series:
        """Decode labels sequentially using only data up to each timestamp (PIT)."""
        series = pd.Series(returns).dropna().astype(float)
        labels: Dict[pd.Timestamp, str] = {}
        for dt in series.index:
            hist = series.loc[:dt]
            if len(hist) < int(min_history):
                labels[pd.Timestamp(dt)] = 'unknown'
                continue
            try:
                labels[pd.Timestamp(dt)] = str(self.predict(hist))
            except RuntimeError:
                labels[pd.Timestamp(dt)] = 'unknown'
        return pd.Series(labels, index=series.index, dtype='object')
