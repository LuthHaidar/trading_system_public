import numpy as np
import pandas as pd

from risk.optimizer import PortfolioOptimizer


def _build_price_data(returns_map: dict[str, np.ndarray]) -> dict[str, pd.DataFrame]:
    dates = pd.bdate_range("2021-01-01", periods=len(next(iter(returns_map.values()))))
    data: dict[str, pd.DataFrame] = {}
    for ticker, rets in returns_map.items():
        close = 100.0 * np.cumprod(1.0 + rets)
        data[ticker] = pd.DataFrame({"Close": close}, index=dates)
    return data


def _signals_for(data: dict[str, pd.DataFrame]) -> dict[str, float]:
    return {ticker: 1.0 for ticker in data}


def test_max_sharpe_returns_unit_sum_weights() -> None:
    rng = np.random.default_rng(7)
    returns_map = {
        "AAA": rng.normal(0.0010, 0.0100, 320),
        "BBB": rng.normal(0.0008, 0.0120, 320),
        "CCC": rng.normal(0.0006, 0.0090, 320),
    }
    data = _build_price_data(returns_map)
    optimizer = PortfolioOptimizer(
        {"covariance_estimator": "sample", "optimizer_max_weight": 1.0}
    )

    weights = optimizer.optimize(_signals_for(data), data, method="max_sharpe")

    assert weights
    assert abs(sum(weights.values()) - 1.0) < 1e-6


def test_min_variance_dominated_by_lowest_vol_asset() -> None:
    rng = np.random.default_rng(11)
    returns_map = {
        "LOW": rng.normal(0.0004, 0.0010, 320),
        "MID": rng.normal(0.0007, 0.0120, 320),
        "HIGH": rng.normal(0.0009, 0.0250, 320),
    }
    data = _build_price_data(returns_map)
    optimizer = PortfolioOptimizer(
        {"covariance_estimator": "sample", "optimizer_max_weight": 1.0}
    )

    weights = optimizer.optimize(_signals_for(data), data, method="min_variance")

    assert weights["LOW"] > 0.80


def test_risk_parity_equal_risk_contribution_diagonal() -> None:
    cov = pd.DataFrame(
        np.diag([0.04, 0.09, 0.16]),
        index=["A", "B", "C"],
        columns=["A", "B", "C"],
    )
    optimizer = PortfolioOptimizer(
        {"covariance_estimator": "sample", "optimizer_max_weight": 1.0}
    )

    weights = optimizer._risk_parity_optimize(cov)

    inv_vol = np.array([1 / 0.2, 1 / 0.3, 1 / 0.4])
    expected = inv_vol / inv_vol.sum()
    assert np.allclose(weights, expected, atol=1e-3)


def test_optimizer_respects_max_weight_constraint() -> None:
    rng = np.random.default_rng(17)
    returns_map = {
        "AAA": rng.normal(0.0012, 0.0100, 320),
        "BBB": rng.normal(0.0008, 0.0120, 320),
        "CCC": rng.normal(0.0006, 0.0130, 320),
        "DDD": rng.normal(0.0007, 0.0110, 320),
    }
    data = _build_price_data(returns_map)
    optimizer = PortfolioOptimizer(
        {"covariance_estimator": "sample", "optimizer_max_weight": 0.30}
    )

    weights = optimizer.optimize(_signals_for(data), data, method="max_sharpe")

    assert max(weights.values()) <= 0.3000001


def test_optimizer_respects_min_weight_constraint() -> None:
    rng = np.random.default_rng(23)
    returns_map = {
        "AAA": rng.normal(0.0011, 0.0110, 320),
        "BBB": rng.normal(0.0010, 0.0100, 320),
        "CCC": rng.normal(0.0009, 0.0090, 320),
    }
    data = _build_price_data(returns_map)
    optimizer = PortfolioOptimizer(
        {
            "covariance_estimator": "sample",
            "optimizer_min_weight": 0.10,
            "optimizer_max_weight": 0.80,
        }
    )

    weights = optimizer.optimize(_signals_for(data), data, method="min_variance")

    assert min(weights.values()) >= 0.099999


def test_covariance_ledoit_wolf_positive_definite() -> None:
    rng = np.random.default_rng(29)
    returns_df = pd.DataFrame(
        {
            "A": rng.normal(0.0, 0.0100, 280),
            "B": rng.normal(0.0, 0.0120, 280),
            "C": rng.normal(0.0, 0.0090, 280),
        }
    )
    optimizer = PortfolioOptimizer({"covariance_estimator": "ledoit_wolf"})

    cov = optimizer._estimate_covariance(returns_df)
    eigvals = np.linalg.eigvalsh(cov.values)

    assert np.all(eigvals > 0)


def test_covariance_exponential_recent_bias() -> None:
    rng = np.random.default_rng(31)
    quiet = rng.normal(0.0004, 0.0050, 220)
    noisy = rng.normal(0.0004, 0.0300, 80)
    series = np.concatenate([quiet, noisy])
    returns_df = pd.DataFrame({"A": series, "B": series * 0.8 + rng.normal(0.0, 0.002, len(series))})

    sample = PortfolioOptimizer(
        {"covariance_estimator": "sample", "covariance_jitter": 0.0}
    )._estimate_covariance(returns_df)
    ew = PortfolioOptimizer(
        {
            "covariance_estimator": "exponential",
            "covariance_ew_halflife": 10,
            "covariance_jitter": 0.0,
        }
    )._estimate_covariance(returns_df)

    assert float(ew.loc["A", "A"]) > float(sample.loc["A", "A"])


def test_covariance_exponential_shrinkage_full_shrink() -> None:
    rng = np.random.default_rng(37)
    returns_df = pd.DataFrame(
        {
            "A": rng.normal(0.0, 0.0100, 300),
            "B": rng.normal(0.0, 0.0120, 300),
            "C": rng.normal(0.0, 0.0090, 300),
        }
    )
    optimizer = PortfolioOptimizer(
        {
            "covariance_estimator": "exponential_shrinkage",
            "covariance_shrinkage": 1.0,
            "covariance_jitter": 0.0,
        }
    )

    cov = optimizer._estimate_covariance(returns_df)
    off_diag = cov.values - np.diag(np.diag(cov.values))

    assert np.allclose(off_diag, 0.0, atol=1e-12)


def test_optimizer_falls_back_on_infeasible_bounds(caplog) -> None:
    rng = np.random.default_rng(41)
    returns_map = {
        "AAA": rng.normal(0.0010, 0.0100, 320),
        "BBB": rng.normal(0.0008, 0.0110, 320),
        "CCC": rng.normal(0.0007, 0.0120, 320),
    }
    data = _build_price_data(returns_map)
    optimizer = PortfolioOptimizer(
        {
            "covariance_estimator": "sample",
            "optimizer_min_weight": 0.50,
            "optimizer_max_weight": 0.60,
        }
    )

    with caplog.at_level("WARNING"):
        weights = optimizer.optimize(_signals_for(data), data, method="min_variance")

    assert "Infeasible optimizer bounds" in caplog.text
    assert all(abs(weights[t] - (1.0 / 3.0)) < 1e-5 for t in ["AAA", "BBB", "CCC"])


def test_unknown_estimator_warns_and_uses_sample(caplog) -> None:
    rng = np.random.default_rng(43)
    returns_df = pd.DataFrame(
        {
            "A": rng.normal(0.0, 0.0100, 260),
            "B": rng.normal(0.0, 0.0110, 260),
            "C": rng.normal(0.0, 0.0120, 260),
        }
    )
    optimizer = PortfolioOptimizer(
        {"covariance_estimator": "nonexistent", "covariance_jitter": 0.0}
    )

    with caplog.at_level("WARNING"):
        cov = optimizer._estimate_covariance(returns_df)

    assert "Unknown covariance_estimator 'nonexistent'; using sample covariance" in caplog.text
    assert np.allclose(cov.values, (returns_df.cov() * 252).values)
