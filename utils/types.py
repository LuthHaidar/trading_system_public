"""Project-wide type aliases."""

from __future__ import annotations

from typing import Dict, TypeAlias

import pandas as pd

WeightMap: TypeAlias = Dict[str, float]
TickerData: TypeAlias = Dict[str, pd.DataFrame]
