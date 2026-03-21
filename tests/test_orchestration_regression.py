import unittest
from collections import deque

import pandas as pd

from strategies.strategy_orchestration import StrategyOrchestrationStrategy


class OrchestrationRegressionTests(unittest.TestCase):
    def test_realized_return_attribution_updates_strategy_histories(self):
        strategy = StrategyOrchestrationStrategy(
            config={
                '_all_strategies_config': {},
                'base_weights': {},
                'performance_lookback': 5,
            }
        )
        strategy.performance_history = {
            's1': deque(maxlen=5),
            's2': deque(maxlen=5),
        }
        strategy.last_strategy_weights = {'s1': 0.7, 's2': 0.3}

        strategy.on_realized_portfolio_return(
            date=pd.Timestamp('2024-01-03'),
            portfolio_return=0.10,
            context={'engine': 'backtest'},
        )

        self.assertAlmostEqual(strategy.performance_history['s1'][-1], 0.07, places=8)
        self.assertAlmostEqual(strategy.performance_history['s2'][-1], 0.03, places=8)


if __name__ == '__main__':
    unittest.main()
