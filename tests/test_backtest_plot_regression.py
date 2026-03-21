import tempfile
import unittest
from pathlib import Path

import pandas as pd

from backtest import plot_calendar_heatmap


class BacktestPlotRegressionTests(unittest.TestCase):
    def test_plot_calendar_heatmap_writes_png(self):
        idx = pd.to_datetime([
            '2023-01-31', '2023-02-28', '2023-03-31',
            '2024-01-31', '2024-02-29',
        ])
        monthly = pd.Series([0.01, -0.02, 0.03, 0.04, -0.01], index=idx)

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'calendar_heatmap.png'
            plot_calendar_heatmap(monthly, str(out))
            self.assertTrue(out.exists())
            self.assertGreater(out.stat().st_size, 0)


if __name__ == '__main__':
    unittest.main()
