import os
import sqlite3
import tempfile
import unittest
import zlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

from utils.data_platform import AuditStore, SQLiteMetricsStore


class DataPlatformRegressionTests(unittest.TestCase):
    def test_state_snapshots_are_compressed_and_legacy_rows_are_migrated(self):
        fd, raw_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        db_path = Path(raw_path)
        try:
            with sqlite3.connect(db_path) as conn:
                conn.execute(
                    """
                    CREATE TABLE state_snapshots (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        ts TEXT NOT NULL,
                        strategy TEXT,
                        run_id TEXT,
                        source TEXT NOT NULL,
                        state_json TEXT NOT NULL
                    )
                    """
                )
                conn.execute(
                    """
                    INSERT INTO state_snapshots (ts, strategy, run_id, source, state_json)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    ("2024-01-01T00:00:00+00:00", "demo", "legacy", "unit", '{"positions":{"AAA":1}}'),
                )

            store = AuditStore(str(db_path))
            legacy_state = store.load_latest_state(source="unit", strategy="demo")
            self.assertIsNotNone(legacy_state)
            self.assertEqual(legacy_state["positions"]["AAA"], 1)

            store.save_state_snapshot(
                state={"positions": {"BBB": 2}},
                source="unit",
                strategy="demo",
                run_id="new",
            )
            latest = store.load_latest_state(source="unit", strategy="demo")
            self.assertEqual(latest["positions"]["BBB"], 2)

            with sqlite3.connect(db_path) as conn:
                cols = [row[1] for row in conn.execute("PRAGMA table_info(state_snapshots)").fetchall()]
                blob = conn.execute("SELECT state_blob FROM state_snapshots ORDER BY id DESC LIMIT 1").fetchone()[0]
                version = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
            self.assertIn("state_blob", cols)
            self.assertNotIn("state_json", cols)
            self.assertGreater(len(blob), 0)
            self.assertEqual(version, 3)
            self.assertIn("BBB", zlib.decompress(blob).decode("utf-8"))
        finally:
            try:
                os.remove(db_path)
            except OSError:
                pass

    def test_prune_audit_events_removes_old_rows(self):
        fd, raw_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        db_path = Path(raw_path)
        try:
            store = AuditStore(str(db_path))
            now = datetime.now(timezone.utc)
            store.log_event("old", "unit", {"age": "old"}, ts=now - timedelta(days=200))
            store.log_event("new", "unit", {"age": "new"}, ts=now - timedelta(days=2))
            deleted = store.prune_audit_events(older_than_days=90)
            self.assertEqual(deleted, 1)
            with sqlite3.connect(db_path) as conn:
                remaining = conn.execute("SELECT COUNT(*) FROM audit_events").fetchone()[0]
            self.assertEqual(remaining, 1)
        finally:
            try:
                os.remove(db_path)
            except OSError:
                pass

    def test_audit_store_migrates_legacy_backtest_runs_schema(self):
        fd, raw_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        db_path = Path(raw_path)
        try:
            with sqlite3.connect(db_path) as conn:
                conn.execute(
                    """
                    CREATE TABLE backtest_runs (
                        run_id TEXT PRIMARY KEY,
                        ts TEXT NOT NULL,
                        strategy TEXT NOT NULL,
                        source TEXT NOT NULL DEFAULT 'backtest',
                        start_date TEXT,
                        end_date TEXT,
                        tickers_json TEXT NOT NULL,
                        benchmark_label TEXT,
                        primary_total_return REAL,
                        primary_cagr REAL,
                        primary_sharpe REAL,
                        primary_max_drawdown REAL,
                        total_return REAL,
                        cagr REAL,
                        volatility REAL,
                        sharpe_ratio REAL,
                        rolling_sharpe_63 REAL,
                        rolling_sharpe_252 REAL,
                        sortino_ratio REAL,
                        calmar_ratio REAL,
                        max_drawdown REAL,
                        max_drawdown_duration_days REAL,
                        max_time_to_recovery_days REAL,
                        benchmark_alpha REAL,
                        benchmark_beta REAL,
                        benchmark_active_return REAL,
                        benchmark_tracking_error REAL,
                        benchmark_information_ratio REAL,
                        spy_alpha REAL,
                        spy_beta REAL,
                        spy_active_return REAL,
                        spy_tracking_error REAL,
                        spy_information_ratio REAL,
                        win_rate REAL,
                        profit_factor REAL,
                        turnover_ratio REAL,
                        total_cost REAL,
                        cost_pct REAL,
                        realized_pnl REAL,
                        unrealized_pnl REAL,
                        initial_equity REAL,
                        final_equity REAL,
                        metrics_json TEXT NOT NULL,
                        degradation_json TEXT,
                        cost_sensitivity_json TEXT
                    )
                    """
                )
                conn.execute(
                    """
                    INSERT INTO backtest_runs (
                        run_id, ts, strategy, source, tickers_json, total_return, sharpe_ratio, metrics_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    ("legacy_run", "2024-01-01T00:00:00+00:00", "demo", "backtest", '["AAA"]', 0.12, 0.8, "{}"),
                )

            store = AuditStore(str(db_path))
            compared = store.compare_backtest_runs(["legacy_run"])
            self.assertEqual(len(compared), 1)
            self.assertAlmostEqual(float(compared.iloc[0]["total_return"]), 0.12, places=8)

            with sqlite3.connect(db_path) as conn:
                cols = [row[1] for row in conn.execute("PRAGMA table_info(backtest_runs)").fetchall()]
                version = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]

            self.assertNotIn("primary_total_return", cols)
            self.assertNotIn("primary_cagr", cols)
            self.assertNotIn("primary_sharpe", cols)
            self.assertNotIn("primary_max_drawdown", cols)
            self.assertNotIn("metrics_json", cols)
            self.assertEqual(version, 3)
        finally:
            try:
                os.remove(db_path)
            except OSError:
                pass

    def test_audit_store_uses_wal_and_utc_default_timestamps(self):
        fd, raw_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        db_path = Path(raw_path)
        try:
            store = AuditStore(str(db_path))
            store.log_event(event_type='heartbeat', source='unit', details={'ok': True})

            with sqlite3.connect(db_path) as conn:
                journal_mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
                ts = conn.execute("SELECT ts FROM audit_events ORDER BY id DESC LIMIT 1").fetchone()[0]

            self.assertEqual(str(journal_mode).lower(), 'wal')
            self.assertIn('+00:00', ts)
        finally:
            try:
                os.remove(db_path)
            except OSError:
                pass

    def test_metrics_store_query_supports_time_window_and_tag_filtering(self):
        fd, raw_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        db_path = Path(raw_path)
        try:
            store = SQLiteMetricsStore(str(db_path))
            t1 = datetime(2024, 1, 1, tzinfo=timezone.utc)
            t2 = datetime(2024, 1, 2, tzinfo=timezone.utc)
            t3 = datetime(2024, 1, 3, tzinfo=timezone.utc)
            store.write_metric('backtest.sharpe_ratio', 0.1, t1, tags={'strategy': 'mr'})
            store.write_metric('backtest.sharpe_ratio', 0.2, t2, tags={'strategy': 'mom'})
            store.write_metric('backtest.sharpe_ratio', 0.3, t3, tags={'strategy': 'mom'})
            store.write_metric('backtest.total_return', 0.4, t3, tags={'strategy': 'mom'})

            df = store.query_metrics(
                name='backtest.sharpe_ratio',
                start=t2,
                end=t3,
                tags={'strategy': 'mom'},
                limit=10,
            )

            self.assertEqual(len(df), 2)
            self.assertEqual(df.iloc[0]['value'], 0.2)
            self.assertEqual(df.iloc[1]['value'], 0.3)
        finally:
            try:
                os.remove(db_path)
            except OSError:
                pass

    def test_metrics_store_migrates_legacy_tags_json_to_metric_tags(self):
        fd, raw_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        db_path = Path(raw_path)
        try:
            with sqlite3.connect(db_path) as conn:
                conn.execute(
                    """
                    CREATE TABLE metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        ts TEXT NOT NULL,
                        name TEXT NOT NULL,
                        value REAL NOT NULL,
                        tags_json TEXT
                    )
                    """
                )
                conn.execute(
                    "INSERT INTO metrics (ts, name, value, tags_json) VALUES (?, ?, ?, ?)",
                    ("2024-01-01T00:00:00+00:00", "backtest.sharpe_ratio", 0.5, '{"strategy":"mom","region":"US"}'),
                )

            store = SQLiteMetricsStore(str(db_path))
            filtered = store.query_metrics("backtest.sharpe_ratio", tags={"strategy": "mom"})
            self.assertEqual(len(filtered), 1)
            self.assertEqual(float(filtered.iloc[0]["value"]), 0.5)

            with sqlite3.connect(db_path) as conn:
                metrics_cols = [row[1] for row in conn.execute("PRAGMA table_info(metrics)").fetchall()]
                tag_count = conn.execute("SELECT COUNT(*) FROM metric_tags").fetchone()[0]
                version = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]

            self.assertNotIn("tags_json", metrics_cols)
            self.assertEqual(tag_count, 2)
            self.assertEqual(version, 2)
        finally:
            try:
                os.remove(db_path)
            except OSError:
                pass

    def test_backtest_run_can_be_recorded_and_compared(self):
        fd, raw_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        db_path = Path(raw_path)
        try:
            store = AuditStore(str(db_path))

            metrics = {
                'total_return': 0.12,
                'cagr': 0.10,
                'volatility': 0.15,
                'sharpe_ratio': 0.8,
                'rolling_sharpe_63': 0.9,
                'rolling_sharpe_252': 0.7,
                'sortino_ratio': 1.1,
                'calmar_ratio': 0.6,
                'max_drawdown': -0.2,
                'max_drawdown_duration_days': 80,
                'max_time_to_recovery_days': 42,
                'benchmark_alpha': 0.01,
                'benchmark_beta': 0.9,
                'benchmark_active_return': -0.02,
                'benchmark_tracking_error': 0.1,
                'benchmark_information_ratio': -0.2,
                'spy_alpha': -0.01,
                'spy_beta': 1.0,
                'spy_active_return': -0.03,
                'spy_tracking_error': 0.11,
                'spy_information_ratio': -0.3,
                'win_rate': 0.55,
                'profit_factor': 1.2,
                'turnover_ratio': 5.0,
                'total_cost': 10.0,
                'cost_pct': 0.01,
                'realized_pnl': 100.0,
                'unrealized_pnl': 15.0,
                'initial_equity': 1000.0,
                'final_equity': 1115.0,
            }
            benchmark = {
                'available': True,
                'ticker': 'EqualWeight(AAA,BBB)',
                'metrics': {
                    'total_return': 0.15,
                    'cagr': 0.12,
                    'sharpe_ratio': 0.85,
                    'max_drawdown': -0.18,
                },
            }

            store.record_backtest_run(
                run_id='run_a',
                strategy='demo',
                start_date='2024-01-01',
                end_date='2024-12-31',
                tickers=['AAA', 'BBB'],
                metrics=metrics,
                benchmark_analysis=benchmark,
                degradation_analysis={'available': True},
                cost_sensitivity={'base': {'estimated_total_cost': 10.0}},
            )

            listed = store.query_backtest_runs(strategy='demo', limit=5)
            self.assertEqual(len(listed), 1)
            self.assertEqual(listed.iloc[0]['run_id'], 'run_a')

            compared = store.compare_backtest_runs(['run_a'])
            self.assertEqual(len(compared), 1)
            self.assertAlmostEqual(float(compared.iloc[0]['sharpe_ratio']), 0.8, places=8)
        finally:
            try:
                os.remove(db_path)
            except OSError:
                pass

    def test_trade_decisions_can_be_recorded_and_queried(self):
        fd, raw_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        db_path = Path(raw_path)
        try:
            store = AuditStore(str(db_path))
            store.record_trade_decisions(
                decisions=[
                    {
                        'ts': '2024-01-02T00:00:00',
                        'signal_ts': '2024-01-01T00:00:00',
                        'ticker': 'AAA',
                        'decision': 'ENTRY',
                        'decision_reason': 'strategy_rebalance',
                        'current_weight': 0.0,
                        'target_weight': 0.2,
                        'signal_strength': 0.2,
                        'weight_delta': 0.2,
                        'confidence': 0.2,
                        'current_shares': 0,
                        'target_shares': 10,
                        'executed_shares': 10,
                        'run_id': 'run_journal',
                    }
                ],
                strategy='demo',
                run_id='run_journal',
            )
            df = store.query_trade_decisions(run_id='run_journal', limit=5)
            self.assertEqual(len(df), 1)
            self.assertEqual(df.iloc[0]['decision'], 'ENTRY')
            self.assertAlmostEqual(float(df.iloc[0]['weight_delta']), 0.2, places=8)
        finally:
            try:
                os.remove(db_path)
            except OSError:
                pass


if __name__ == '__main__':
    unittest.main()
