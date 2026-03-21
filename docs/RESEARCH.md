# Research Utilities

This document covers non-production research helpers under `research/`.

## `research/efficient_frontier.py`

### Purpose
`calculate_efficient_frontier(...)` builds a set of efficient-frontier points from historical close-price data using the same covariance/optimization machinery as `risk.optimizer.PortfolioOptimizer`.

This module is intended for **analysis and experimentation**, not direct live-order routing.

### Function signature

```python
calculate_efficient_frontier(
    data: Dict[str, pd.DataFrame],
    config: Optional[Dict] = None,
    n_points: int = 50,
) -> pd.DataFrame
```

### Inputs
- `data`: mapping of ticker -> DataFrame. At minimum, each DataFrame must include a `Close` column indexed by datetime.
- `config`: optional optimizer config (for example: `optimizer_min_weight`, `optimizer_max_weight`, covariance estimator settings).
- `n_points`: number of target-return points to sample between the minimum and maximum annualized mean return across assets.

### Output
Returns a DataFrame with columns:
- `return`: annualized portfolio return for the frontier point.
- `volatility`: annualized portfolio volatility.
- `sharpe`: `(return - risk_free_rate) / volatility` (0.0 when volatility is non-positive).

If there is insufficient return data, an empty DataFrame with these columns is returned.

### Method overview
1. Prepares returns from the provided price data.
2. Estimates covariance using `PortfolioOptimizer` settings.
3. Sweeps target returns on an evenly spaced grid.
4. Solves a minimum-variance portfolio for each target return.
5. Computes return/volatility/Sharpe per successful solution.

### Example

```python
import pandas as pd
from research.efficient_frontier import calculate_efficient_frontier

# data = {"SPY": spy_df, "QQQ": qqq_df, ...}
frontier = calculate_efficient_frontier(
    data=data,
    config={"optimizer_min_weight": 0.0, "optimizer_max_weight": 1.0},
    n_points=40,
)

print(frontier.head())
```

### Notes and limitations
- The helper currently returns frontier summary points only; it does not return per-point weights.
- Optimization feasibility depends on bounds and target-return constraints.
- Because this is a research helper, callers should treat output as analytical input and apply additional validation/visualization as needed.
