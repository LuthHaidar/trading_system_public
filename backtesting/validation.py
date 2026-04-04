import itertools
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class WindowSplit:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


class ValidationSuite:
    """Research validation helpers for robust strategy evaluation."""

    @staticmethod
    def benchmark_comparison(strategy_returns: pd.Series,
                             benchmark_returns: pd.Series,
                             risk_free_rate: float = 0.02) -> Dict[str, float]:
        aligned = pd.concat([strategy_returns, benchmark_returns], axis=1).dropna()
        if len(aligned) < 2:
            return {
                'alpha': 0.0,
                'beta': 0.0,
                'tracking_error': 0.0,
                'information_ratio': 0.0,
                'active_return': 0.0,
            }

        strategy = aligned.iloc[:, 0]
        benchmark = aligned.iloc[:, 1]
        # Use pandas covariance/variance defaults (sample statistics, ddof=1)
        # so beta remains consistent with the primary performance metrics module.
        cov = strategy.cov(benchmark)
        var_b = benchmark.var()
        beta = cov / var_b if var_b > 0 else 0.0

        daily_rf = risk_free_rate / 252
        alpha_daily = (strategy.mean() - daily_rf) - beta * (benchmark.mean() - daily_rf)
        active = strategy - benchmark
        tracking_error = active.std() * np.sqrt(252)
        information_ratio = (active.mean() * np.sqrt(252) / active.std()) if active.std() > 0 else 0.0
        active_return = float(active.mean() * 252)

        return {
            'alpha': float(alpha_daily * 252),
            'beta': float(beta),
            'tracking_error': float(tracking_error),
            'information_ratio': float(information_ratio),
            'active_return': active_return,
        }

    @staticmethod
    def train_oos_split(series: pd.Series, split_ratio: float = 0.7) -> Tuple[pd.Series, pd.Series]:
        split_idx = max(1, int(len(series) * split_ratio))
        return series.iloc[:split_idx], series.iloc[split_idx:]

    @staticmethod
    def time_series_cv(index: pd.DatetimeIndex,
                       train_size: int,
                       test_size: int,
                       step_size: int,
                       anchored: bool = False) -> List[WindowSplit]:
        splits = []
        start = train_size
        while start + test_size <= len(index):
            train_start = index[0] if anchored else index[start - train_size]
            train_end = index[start - 1]
            test_start = index[start]
            test_end = index[start + test_size - 1]
            splits.append(WindowSplit(train_start, train_end, test_start, test_end))
            start += step_size
        return splits

    @staticmethod
    def rolling_parameter_optimization(
        index: pd.DatetimeIndex,
        param_grid: Dict[str, Sequence],
        objective_fn: Callable[[WindowSplit, Dict], float],
        train_size: int,
        test_size: int,
        step_size: int,
        anchored: bool = False,
    ) -> List[Dict]:
        splits = ValidationSuite.time_series_cv(index, train_size, test_size, step_size, anchored)
        keys = list(param_grid.keys())
        combinations = [dict(zip(keys, vals)) for vals in itertools.product(*[param_grid[k] for k in keys])]

        results = []
        for split in splits:
            best = {'score': -np.inf, 'params': None}
            for params in combinations:
                score = objective_fn(split, params)
                if score > best['score']:
                    best = {'score': float(score), 'params': params}
            results.append({
                'split': split,
                'best_score': best['score'],
                'best_params': best['params'],
            })
        return results

    @staticmethod
    def parameter_stability(optimization_results: List[Dict]) -> Dict[str, float]:
        if not optimization_results:
            return {'stability': 0.0}

        keys = list(optimization_results[0]['best_params'].keys())
        per_key = {}
        for key in keys:
            values = [res['best_params'][key] for res in optimization_results if res['best_params'] is not None]
            uniq = len(set(values))
            per_key[f'{key}_switch_ratio'] = (uniq - 1) / max(1, len(values) - 1)

        avg_switch = np.mean(list(per_key.values())) if per_key else 0.0
        per_key['stability'] = float(1.0 - avg_switch)
        return per_key

    @staticmethod
    def degradation_report(train_returns: pd.Series, oos_returns: pd.Series) -> Dict[str, float]:
        train_sharpe = ValidationSuite._sharpe(train_returns)
        oos_sharpe = ValidationSuite._sharpe(oos_returns)
        train_cagr = ValidationSuite._cagr_from_returns(train_returns)
        oos_cagr = ValidationSuite._cagr_from_returns(oos_returns)
        return {
            'train_sharpe': train_sharpe,
            'oos_sharpe': oos_sharpe,
            'sharpe_drift': oos_sharpe - train_sharpe,
            'train_cagr': train_cagr,
            'oos_cagr': oos_cagr,
            'cagr_drift': oos_cagr - train_cagr,
        }

    @staticmethod
    def monte_carlo_simulation(
        returns: pd.Series,
        n_sims: int = 200,
        horizon_days: int = 252,
        block_size: int = 5,
        random_state: int = 42,
        return_paths: bool = False,
    ) -> Dict[str, float]:
        if returns.empty:
            result = {'p05_return': 0.0, 'p50_return': 0.0, 'p95_return': 0.0}
            if return_paths:
                result['paths'] = np.empty((0, 0), dtype=np.float32)
            return result

        rng = np.random.default_rng(random_state)
        samples = returns.values
        n_obs = len(samples)
        block = max(1, int(block_size))
        paths = np.ones((n_sims, horizon_days + 1), dtype=np.float64) if return_paths else None

        if block <= 1 or n_obs <= 1:
            # Fallback to iid bootstrap for very short series.
            iid = rng.choice(samples, size=(n_sims, horizon_days), replace=True)
            path_growth = np.cumprod(1 + iid, axis=1)
            terminal = path_growth[:, -1] - 1
            if return_paths:
                paths[:, 1:] = path_growth
                paths = paths / paths[:, [0]]
            result = {
                'p05_return': float(np.percentile(terminal, 5)),
                'p50_return': float(np.percentile(terminal, 50)),
                'p95_return': float(np.percentile(terminal, 95)),
            }
            if return_paths:
                result['paths'] = paths.astype(np.float32)
            return result

        block = min(block, n_obs)
        n_blocks = int(np.ceil(horizon_days / block))
        terminal = np.zeros(n_sims, dtype=float)

        max_start = n_obs - block
        for i in range(n_sims):
            path_chunks = []
            for _ in range(n_blocks):
                start = rng.integers(0, max_start + 1) if max_start > 0 else 0
                path_chunks.append(samples[start:start + block])
            scenario = np.concatenate(path_chunks)[:horizon_days]
            path = np.cumprod(1 + scenario)
            terminal[i] = path[-1] - 1
            if return_paths:
                paths[i, 1:] = path

        result = {
            'p05_return': float(np.percentile(terminal, 5)),
            'p50_return': float(np.percentile(terminal, 50)),
            'p95_return': float(np.percentile(terminal, 95)),
        }
        if return_paths:
            paths = paths / paths[:, [0]]
            result['paths'] = paths.astype(np.float32)
        return result

    @staticmethod
    def monte_carlo_path_stats(paths: np.ndarray, ruin_threshold: float = 0.5) -> Dict[str, float]:
        if paths is None or len(paths) == 0:
            return {
                'avg_drawdown': 0.0,
                'p05_drawdown': 0.0,
                'p95_drawdown': 0.0,
                'avg_sharpe': 0.0,
                'p05_sharpe': 0.0,
                'p95_sharpe': 0.0,
                'avg_volatility': 0.0,
                'p_ruin': 0.0,
            }

        paths = np.asarray(paths, dtype=float)
        if paths.ndim != 2 or paths.shape[1] < 2:
            raise ValueError('paths must be a 2D array with shape (n_sims, horizon_days+1)')

        running_max = np.maximum.accumulate(paths, axis=1)
        drawdowns = np.where(running_max > 0, (running_max - paths) / running_max, 0.0)
        max_drawdowns = np.max(drawdowns, axis=1)

        daily_returns = np.diff(paths, axis=1) / np.where(paths[:, :-1] == 0, np.nan, paths[:, :-1])
        std_daily = np.nanstd(daily_returns, axis=1)
        vol = std_daily * np.sqrt(252)
        mean_daily = np.nanmean(daily_returns, axis=1)
        sharpe = np.zeros_like(mean_daily, dtype=float)
        np.divide(mean_daily * np.sqrt(252), std_daily, out=sharpe, where=std_daily > 0)
        sharpe = np.nan_to_num(sharpe, nan=0.0, posinf=0.0, neginf=0.0)

        p_ruin = float(np.mean(np.min(paths, axis=1) < float(ruin_threshold)))

        return {
            'avg_drawdown': float(np.mean(max_drawdowns)),
            'p05_drawdown': float(np.percentile(max_drawdowns, 5)),
            'p95_drawdown': float(np.percentile(max_drawdowns, 95)),
            'avg_sharpe': float(np.mean(sharpe)),
            'p05_sharpe': float(np.percentile(sharpe, 5)),
            'p95_sharpe': float(np.percentile(sharpe, 95)),
            'avg_volatility': float(np.mean(vol)),
            'p_ruin': p_ruin,
        }

    @staticmethod
    def _extract_trade_returns(trades) -> np.ndarray:
        values = []
        for trade in trades or []:
            if isinstance(trade, dict):
                if 'pnl' in trade:
                    values.append(float(trade['pnl']))
                elif 'return' in trade:
                    values.append(float(trade['return']))
            else:
                pnl = getattr(trade, 'pnl', None)
                ret = getattr(trade, 'return_', None)
                if pnl is not None:
                    values.append(float(pnl))
                elif ret is not None:
                    values.append(float(ret))
        return np.asarray(values, dtype=float)

    @staticmethod
    def monte_carlo_trade_simulation(
        trades,
        n_sims: int = 200,
        mode: str = 'shuffle',
        ruin_threshold: float = 0.5,
        random_state: int = 42,
        return_paths: bool = False,
    ) -> Dict[str, float]:
        trade_returns = ValidationSuite._extract_trade_returns(trades)
        if trade_returns.size == 0:
            result = {'p05_return': 0.0, 'p50_return': 0.0, 'p95_return': 0.0, 'trade_count': 0}
            if return_paths:
                result['paths'] = np.empty((0, 0), dtype=np.float32)
            return result

        rng = np.random.default_rng(random_state)
        n = int(trade_returns.size)
        terminal = np.zeros(n_sims, dtype=float)
        paths = np.ones((n_sims, n + 1), dtype=np.float64) if return_paths else None
        clipped = np.clip(trade_returns, -0.99, None)

        for i in range(n_sims):
            if mode == 'resample':
                scenario = rng.choice(clipped, size=n, replace=True)
            else:
                scenario = rng.permutation(clipped)
            path = np.cumprod(1.0 + scenario)
            terminal[i] = float(path[-1] - 1.0)
            if return_paths:
                paths[i, 1:] = path

        result = {
            'p05_return': float(np.percentile(terminal, 5)),
            'p50_return': float(np.percentile(terminal, 50)),
            'p95_return': float(np.percentile(terminal, 95)),
            'trade_count': n,
        }
        if return_paths:
            result['paths'] = paths.astype(np.float32)
            result['path_stats'] = ValidationSuite.monte_carlo_trade_path_stats(
                result['paths'],
                ruin_threshold=ruin_threshold,
            )
        return result

    @staticmethod
    def monte_carlo_trade_path_stats(paths: np.ndarray, ruin_threshold: float = 0.5) -> Dict[str, float]:
        return ValidationSuite.monte_carlo_path_stats(paths=paths, ruin_threshold=ruin_threshold)

    @staticmethod
    def sensitivity_analysis(base_params: Dict,
                             perturbations: Dict[str, Sequence],
                             evaluator: Callable[[Dict], float]) -> pd.DataFrame:
        keys = list(perturbations.keys())
        rows = []
        for vals in itertools.product(*[perturbations[k] for k in keys]):
            candidate = dict(base_params)
            candidate.update(dict(zip(keys, vals)))
            score = evaluator(candidate)
            row = dict(candidate)
            row['score'] = float(score)
            rows.append(row)
        return pd.DataFrame(rows).sort_values('score', ascending=False).reset_index(drop=True)

    @staticmethod
    def multi_universe_summary(universe_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        sharpe_vals = [v.get('sharpe_ratio', 0.0) for v in universe_metrics.values()]
        return_vals = [v.get('total_return', 0.0) for v in universe_metrics.values()]
        return {
            'universes': len(universe_metrics),
            'avg_sharpe': float(np.mean(sharpe_vals)) if sharpe_vals else 0.0,
            'sharpe_dispersion': float(np.std(sharpe_vals)) if sharpe_vals else 0.0,
            'avg_total_return': float(np.mean(return_vals)) if return_vals else 0.0,
        }

    @staticmethod
    def _sharpe(returns: pd.Series) -> float:
        std = returns.std()
        if len(returns) < 2 or std == 0:
            return 0.0
        return float(np.sqrt(252) * returns.mean() / std)

    @staticmethod
    def _cagr_from_returns(returns: pd.Series) -> float:
        if returns.empty:
            return 0.0
        periods = len(returns)
        total = float(np.prod(1 + returns.values))
        years = periods / 252
        if years <= 0 or total <= 0:
            return 0.0
        return float(total ** (1 / years) - 1)
