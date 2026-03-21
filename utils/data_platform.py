"""Phase 2 data platform: audit persistence, trade journal, and metrics storage adapters."""

from __future__ import annotations

import json
import sqlite3
import zlib
from dataclasses import asdict, is_dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

class AuditStore:
    """SQLite-backed audit trail for trades, equity snapshots, and state checkpoints."""

    _BACKTEST_RUNS_COLUMNS: List[str] = [
        "run_id",
        "ts",
        "strategy",
        "source",
        "start_date",
        "end_date",
        "tickers_json",
        "benchmark_label",
        "total_return",
        "cagr",
        "volatility",
        "sharpe_ratio",
        "rolling_sharpe_63",
        "rolling_sharpe_252",
        "sortino_ratio",
        "calmar_ratio",
        "max_drawdown",
        "max_drawdown_duration_days",
        "max_time_to_recovery_days",
        "benchmark_alpha",
        "benchmark_beta",
        "benchmark_active_return",
        "benchmark_tracking_error",
        "benchmark_information_ratio",
        "spy_alpha",
        "spy_beta",
        "spy_active_return",
        "spy_tracking_error",
        "spy_information_ratio",
        "win_rate",
        "profit_factor",
        "turnover_ratio",
        "total_cost",
        "cost_pct",
        "realized_pnl",
        "unrealized_pnl",
        "initial_equity",
        "final_equity",
        "degradation_json",
        "cost_sensitivity_json",
    ]

    def __init__(self, db_path: str = "state/trading_audit.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _initialize_schema(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._run_migrations(conn)

    def _run_migrations(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER NOT NULL,
                applied_at TEXT NOT NULL,
                description TEXT
            )
            """
        )
        row = conn.execute("SELECT MAX(version) AS version FROM schema_version").fetchone()
        current_version = int(row["version"]) if row and row["version"] is not None else 0
        migrations = [
            (1, "create audit schema baseline", self._migrate_v1),
            (2, "drop legacy backtest primary_* and metrics_json columns", self._migrate_v2),
            (3, "compress state snapshots into state_blob", self._migrate_v3),
        ]
        for version, description, fn in migrations:
            if version <= current_version:
                continue
            fn(conn)
            conn.execute(
                "INSERT INTO schema_version (version, applied_at, description) VALUES (?, ?, ?)",
                (version, datetime.now(timezone.utc).isoformat(), description),
            )

    def _migrate_v1(self, conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS audit_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                event_type TEXT NOT NULL,
                source TEXT NOT NULL,
                strategy TEXT,
                run_id TEXT,
                details_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                strategy TEXT NOT NULL,
                run_id TEXT,
                ticker TEXT NOT NULL,
                action TEXT NOT NULL,
                shares REAL NOT NULL,
                price REAL NOT NULL,
                value REAL NOT NULL,
                commission REAL NOT NULL,
                slippage REAL NOT NULL,
                fx_cost REAL NOT NULL,
                currency TEXT NOT NULL,
                signal_ts TEXT,
                decision TEXT,
                decision_reason TEXT,
                signal_strength REAL,
                signal_confidence REAL,
                weight_delta REAL,
                current_weight REAL,
                target_weight REAL,
                current_shares REAL,
                target_shares REAL,
                executed_shares REAL,
                source TEXT NOT NULL DEFAULT 'backtest'
            );

            CREATE TABLE IF NOT EXISTS equity_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                strategy TEXT NOT NULL,
                run_id TEXT,
                equity REAL NOT NULL,
                cash REAL,
                realized_equity REAL,
                unrealized_equity REAL,
                source TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS state_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                strategy TEXT,
                run_id TEXT,
                source TEXT NOT NULL,
                state_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS trade_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                signal_ts TEXT,
                strategy TEXT NOT NULL,
                run_id TEXT,
                ticker TEXT NOT NULL,
                decision TEXT NOT NULL,
                decision_reason TEXT,
                current_weight REAL,
                target_weight REAL,
                signal_strength REAL,
                weight_delta REAL,
                confidence REAL,
                current_shares REAL,
                target_shares REAL,
                executed_shares REAL,
                source TEXT NOT NULL DEFAULT 'backtest'
            );

            CREATE TABLE IF NOT EXISTS backtest_runs (
                run_id TEXT PRIMARY KEY,
                ts TEXT NOT NULL,
                strategy TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'backtest',
                start_date TEXT,
                end_date TEXT,
                tickers_json TEXT NOT NULL,
                benchmark_label TEXT,
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
                degradation_json TEXT,
                cost_sensitivity_json TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_trades_ts ON trades(ts);
            CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy);
            CREATE INDEX IF NOT EXISTS idx_equity_strategy_ts ON equity_snapshots(strategy, ts);
            CREATE INDEX IF NOT EXISTS idx_events_source_ts ON audit_events(source, ts);
            """
        )
        self._ensure_column(conn, "audit_events", "run_id", "TEXT")
        self._ensure_column(conn, "trades", "run_id", "TEXT")
        self._ensure_column(conn, "trades", "signal_ts", "TEXT")
        self._ensure_column(conn, "trades", "decision", "TEXT")
        self._ensure_column(conn, "trades", "decision_reason", "TEXT")
        self._ensure_column(conn, "trades", "signal_strength", "REAL")
        self._ensure_column(conn, "trades", "signal_confidence", "REAL")
        self._ensure_column(conn, "trades", "weight_delta", "REAL")
        self._ensure_column(conn, "trades", "current_weight", "REAL")
        self._ensure_column(conn, "trades", "target_weight", "REAL")
        self._ensure_column(conn, "trades", "current_shares", "REAL")
        self._ensure_column(conn, "trades", "target_shares", "REAL")
        self._ensure_column(conn, "trades", "executed_shares", "REAL")
        self._ensure_column(conn, "trade_decisions", "weight_delta", "REAL")
        self._ensure_column(conn, "equity_snapshots", "run_id", "TEXT")
        self._ensure_column(conn, "equity_snapshots", "realized_equity", "REAL")
        self._ensure_column(conn, "equity_snapshots", "unrealized_equity", "REAL")
        self._ensure_column(conn, "state_snapshots", "run_id", "TEXT")
        self._ensure_column(conn, "backtest_runs", "degradation_json", "TEXT")
        self._ensure_column(conn, "backtest_runs", "cost_sensitivity_json", "TEXT")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_run_id ON trades(run_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_equity_run_id_ts ON equity_snapshots(run_id, ts)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_events_run_id_ts ON audit_events(run_id, ts)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_trade_decisions_run_id_ts ON trade_decisions(run_id, ts)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_backtest_runs_strategy_ts ON backtest_runs(strategy, ts)")

    def _migrate_v2(self, conn: sqlite3.Connection) -> None:
        table_exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'backtest_runs'"
        ).fetchone()
        if not table_exists:
            return

        existing = {row["name"] for row in conn.execute("PRAGMA table_info(backtest_runs)").fetchall()}
        legacy_columns = {
            "primary_total_return",
            "primary_cagr",
            "primary_sharpe",
            "primary_max_drawdown",
            "metrics_json",
        }
        if not (existing & legacy_columns):
            return

        conn.execute("DROP TABLE IF EXISTS backtest_runs_new")
        conn.execute(
            """
            CREATE TABLE backtest_runs_new (
                run_id TEXT PRIMARY KEY,
                ts TEXT NOT NULL,
                strategy TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'backtest',
                start_date TEXT,
                end_date TEXT,
                tickers_json TEXT NOT NULL,
                benchmark_label TEXT,
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
                degradation_json TEXT,
                cost_sensitivity_json TEXT
            )
            """
        )

        select_terms = []
        for column in self._BACKTEST_RUNS_COLUMNS:
            if column in existing:
                select_terms.append(column)
            else:
                if column in {"degradation_json", "cost_sensitivity_json"}:
                    select_terms.append(f"'{{}}' AS {column}")
                else:
                    select_terms.append(f"NULL AS {column}")

        conn.execute(
            f"""
            INSERT INTO backtest_runs_new ({", ".join(self._BACKTEST_RUNS_COLUMNS)})
            SELECT {", ".join(select_terms)}
            FROM backtest_runs
            """
        )
        conn.execute("DROP TABLE backtest_runs")
        conn.execute("ALTER TABLE backtest_runs_new RENAME TO backtest_runs")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_backtest_runs_strategy_ts ON backtest_runs(strategy, ts)")

    def _migrate_v3(self, conn: sqlite3.Connection) -> None:
        table_exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'state_snapshots'"
        ).fetchone()
        if not table_exists:
            return

        existing = {row["name"] for row in conn.execute("PRAGMA table_info(state_snapshots)").fetchall()}
        if "state_blob" in existing and "state_json" not in existing:
            return

        conn.execute("DROP TABLE IF EXISTS state_snapshots_new")
        conn.execute(
            """
            CREATE TABLE state_snapshots_new (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                strategy TEXT,
                run_id TEXT,
                source TEXT NOT NULL,
                state_blob BLOB NOT NULL
            )
            """
        )
        if "state_json" in existing:
            rows = conn.execute(
                "SELECT id, ts, strategy, run_id, source, state_json FROM state_snapshots"
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, ts, strategy, run_id, source, state_blob FROM state_snapshots"
            ).fetchall()
        payload_rows: List[tuple[Any, ...]] = []
        for row in rows:
            if "state_json" in existing:
                raw_state = row["state_json"] or "{}"
                compressed = zlib.compress(raw_state.encode("utf-8"))
            else:
                blob = row["state_blob"]
                compressed = bytes(blob) if blob is not None else zlib.compress(b"{}")
            payload_rows.append(
                (
                    int(row["id"]),
                    row["ts"],
                    row["strategy"],
                    row["run_id"],
                    row["source"],
                    compressed,
                )
            )
        if payload_rows:
            conn.executemany(
                """
                INSERT INTO state_snapshots_new (id, ts, strategy, run_id, source, state_blob)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                payload_rows,
            )
        conn.execute("DROP TABLE state_snapshots")
        conn.execute("ALTER TABLE state_snapshots_new RENAME TO state_snapshots")

    @staticmethod
    def _ensure_column(conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
        existing = {row['name'] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        if column in existing:
            return
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def log_event(
        self,
        event_type: str,
        source: str,
        details: Dict[str, Any],
        strategy: Optional[str] = None,
        ts: Optional[datetime] = None,
        run_id: Optional[str] = None,
    ) -> None:
        ts_value = (ts or datetime.now(timezone.utc)).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO audit_events (ts, event_type, source, strategy, run_id, details_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (ts_value, event_type, source, strategy, run_id, json.dumps(details, default=str)),
            )

    def record_trades(self, trades: Iterable[Any], strategy: str, source: str = "backtest",
                      run_id: Optional[str] = None) -> int:
        rows = []
        for trade in trades:
            payload = asdict(trade) if is_dataclass(trade) else dict(trade)
            ts = payload.get("date") or payload.get("ts") or datetime.now(timezone.utc)
            if isinstance(ts, (datetime, pd.Timestamp)):
                ts = ts.isoformat()
            signal_ts = payload.get("signal_date") or payload.get("signal_ts")
            if isinstance(signal_ts, (datetime, pd.Timestamp)):
                signal_ts = signal_ts.isoformat()
            rows.append(
                (
                    ts,
                    strategy,
                    payload.get("run_id", run_id),
                    payload["ticker"],
                    payload["action"],
                    float(payload["shares"]),
                    float(payload["price"]),
                    float(payload["value"]),
                    float(payload.get("commission", 0.0)),
                    float(payload.get("slippage", 0.0)),
                    float(payload.get("fx_cost", 0.0)),
                    payload.get("currency", "USD"),
                    signal_ts,
                    payload.get("decision"),
                    payload.get("decision_reason"),
                    float(payload["signal_strength"]) if payload.get("signal_strength") is not None else None,
                    float(payload["signal_confidence"]) if payload.get("signal_confidence") is not None else None,
                    float(payload["weight_delta"]) if payload.get("weight_delta") is not None else None,
                    float(payload["current_weight"]) if payload.get("current_weight") is not None else None,
                    float(payload["target_weight"]) if payload.get("target_weight") is not None else None,
                    float(payload["current_shares"]) if payload.get("current_shares") is not None else None,
                    float(payload["target_shares"]) if payload.get("target_shares") is not None else None,
                    float(payload["executed_shares"]) if payload.get("executed_shares") is not None else None,
                    source,
                )
            )

        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO trades (
                    ts, strategy, run_id, ticker, action, shares, price, value,
                    commission, slippage, fx_cost, currency, signal_ts, decision, decision_reason,
                    signal_strength, signal_confidence, weight_delta, current_weight, target_weight,
                    current_shares, target_shares, executed_shares, source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
        return len(rows)

    def record_equity_curve(self, equity_curve: pd.Series, strategy: str, source: str = "backtest",
                            cash_curve: Optional[pd.Series] = None,
                            realized_equity_curve: Optional[pd.Series] = None,
                            unrealized_equity_curve: Optional[pd.Series] = None,
                            run_id: Optional[str] = None) -> int:
        if equity_curve is None or equity_curve.empty:
            return 0

        rows = []
        for ts, equity in equity_curve.items():
            stamp = ts.isoformat() if isinstance(ts, (datetime, pd.Timestamp)) else str(ts)
            cash_val = None
            realized_val = None
            unrealized_val = None
            if cash_curve is not None and ts in cash_curve.index:
                cash_val = float(cash_curve.loc[ts])
            if realized_equity_curve is not None and ts in realized_equity_curve.index:
                realized_val = float(realized_equity_curve.loc[ts])
            if unrealized_equity_curve is not None and ts in unrealized_equity_curve.index:
                unrealized_val = float(unrealized_equity_curve.loc[ts])
            rows.append((stamp, strategy, run_id, float(equity), cash_val, realized_val, unrealized_val, source))

        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO equity_snapshots (ts, strategy, run_id, equity, cash, realized_equity, unrealized_equity, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
        return len(rows)

    def save_state_snapshot(self, state: Dict[str, Any], source: str,
                            strategy: Optional[str] = None, run_id: Optional[str] = None) -> None:
        state_blob = zlib.compress(json.dumps(state, default=str).encode("utf-8"))
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO state_snapshots (ts, strategy, run_id, source, state_blob)
                VALUES (?, ?, ?, ?, ?)
                """,
                (datetime.now(timezone.utc).isoformat(), strategy, run_id, source, state_blob),
            )

    def load_latest_state(self, source: str, strategy: Optional[str] = None) -> Optional[Dict[str, Any]]:
        query = """
            SELECT state_blob
            FROM state_snapshots
            WHERE source = ?
        """
        params: List[Any] = [source]
        if strategy:
            query += " AND strategy = ?"
            params.append(strategy)
        query += " ORDER BY ts DESC LIMIT 1"

        with self._connect() as conn:
            row = conn.execute(query, params).fetchone()

        if not row:
            return None
        try:
            raw = zlib.decompress(row["state_blob"]).decode("utf-8")
        except (TypeError, zlib.error, UnicodeDecodeError):
            raw = row["state_blob"].decode("utf-8") if isinstance(row["state_blob"], bytes) else "{}"
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    def prune_audit_events(self, older_than_days: int = 90) -> int:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=older_than_days)).isoformat()
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM audit_events WHERE ts < ?", (cutoff,))
            return cursor.rowcount

    def query_trade_journal(self, ticker: Optional[str] = None, strategy: Optional[str] = None,
                            run_id: Optional[str] = None, limit: int = 200) -> pd.DataFrame:
        clauses = []
        params: List[Any] = []
        if ticker:
            clauses.append("ticker = ?")
            params.append(ticker)
        if strategy:
            clauses.append("strategy = ?")
            params.append(strategy)
        if run_id:
            clauses.append("run_id = ?")
            params.append(run_id)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"SELECT * FROM trades {where} ORDER BY ts DESC LIMIT ?"
        params.append(limit)

        with self._connect() as conn:
            df = pd.read_sql_query(query, conn, params=params)
        return df

    def record_trade_decisions(self, decisions: Iterable[Dict[str, Any]], strategy: str,
                               source: str = "backtest", run_id: Optional[str] = None) -> int:
        rows = []
        for decision in decisions:
            ts = decision.get("ts") or datetime.now(timezone.utc)
            if isinstance(ts, (datetime, pd.Timestamp)):
                ts = ts.isoformat()
            signal_ts = decision.get("signal_ts")
            if isinstance(signal_ts, (datetime, pd.Timestamp)):
                signal_ts = signal_ts.isoformat()
            rows.append(
                (
                    ts,
                    signal_ts,
                    strategy,
                    decision.get("run_id", run_id),
                    str(decision.get("ticker", "")),
                    str(decision.get("decision", "HOLD")),
                    decision.get("decision_reason"),
                    float(decision["current_weight"]) if decision.get("current_weight") is not None else None,
                    float(decision["target_weight"]) if decision.get("target_weight") is not None else None,
                    float(decision["signal_strength"]) if decision.get("signal_strength") is not None else None,
                    float(decision["weight_delta"]) if decision.get("weight_delta") is not None else None,
                    float(decision["confidence"]) if decision.get("confidence") is not None else None,
                    float(decision["current_shares"]) if decision.get("current_shares") is not None else None,
                    float(decision["target_shares"]) if decision.get("target_shares") is not None else None,
                    float(decision["executed_shares"]) if decision.get("executed_shares") is not None else None,
                    source,
                )
            )

        if not rows:
            return 0

        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO trade_decisions (
                    ts, signal_ts, strategy, run_id, ticker, decision, decision_reason,
                    current_weight, target_weight, signal_strength, weight_delta, confidence,
                    current_shares, target_shares, executed_shares, source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
        return len(rows)

    def query_trade_decisions(self, run_id: Optional[str] = None, strategy: Optional[str] = None,
                              ticker: Optional[str] = None, limit: int = 200) -> pd.DataFrame:
        clauses = []
        params: List[Any] = []
        if run_id:
            clauses.append("run_id = ?")
            params.append(run_id)
        if strategy:
            clauses.append("strategy = ?")
            params.append(strategy)
        if ticker:
            clauses.append("ticker = ?")
            params.append(ticker)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"SELECT * FROM trade_decisions {where} ORDER BY ts DESC LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            return pd.read_sql_query(query, conn, params=params)

    def record_backtest_run(self,
                            run_id: str,
                            strategy: str,
                            start_date: str,
                            end_date: str,
                            tickers: List[str],
                            metrics: Dict[str, Any],
                            benchmark_analysis: Optional[Dict[str, Any]] = None,
                            degradation_analysis: Optional[Dict[str, Any]] = None,
                            cost_sensitivity: Optional[Dict[str, Any]] = None,
                            source: str = "backtest",
                            ts: Optional[datetime] = None) -> None:
        benchmark = benchmark_analysis or {}

        payload = (
            run_id,
            (ts or datetime.now(timezone.utc)).isoformat(),
            strategy,
            source,
            start_date,
            end_date,
            json.dumps(tickers, default=str),
            benchmark.get('ticker'),
            metrics.get('total_return'),
            metrics.get('cagr'),
            metrics.get('volatility'),
            metrics.get('sharpe_ratio'),
            metrics.get('rolling_sharpe_63'),
            metrics.get('rolling_sharpe_252'),
            metrics.get('sortino_ratio'),
            metrics.get('calmar_ratio'),
            metrics.get('max_drawdown'),
            metrics.get('max_drawdown_duration_days'),
            metrics.get('max_time_to_recovery_days'),
            metrics.get('benchmark_alpha'),
            metrics.get('benchmark_beta'),
            metrics.get('benchmark_active_return'),
            metrics.get('benchmark_tracking_error'),
            metrics.get('benchmark_information_ratio'),
            metrics.get('spy_alpha'),
            metrics.get('spy_beta'),
            metrics.get('spy_active_return'),
            metrics.get('spy_tracking_error'),
            metrics.get('spy_information_ratio'),
            metrics.get('win_rate'),
            metrics.get('profit_factor'),
            metrics.get('turnover_ratio'),
            metrics.get('total_cost'),
            metrics.get('cost_pct'),
            metrics.get('realized_pnl'),
            metrics.get('unrealized_pnl'),
            metrics.get('initial_equity', metrics.get('initial_capital')),
            metrics.get('final_equity', metrics.get('final_capital')),
            json.dumps(degradation_analysis or {}, default=str),
            json.dumps(cost_sensitivity or {}, default=str),
        )

        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO backtest_runs (
                    run_id, ts, strategy, source, start_date, end_date, tickers_json, benchmark_label,
                    total_return, cagr, volatility, sharpe_ratio, rolling_sharpe_63, rolling_sharpe_252,
                    sortino_ratio, calmar_ratio, max_drawdown, max_drawdown_duration_days, max_time_to_recovery_days,
                    benchmark_alpha, benchmark_beta, benchmark_active_return, benchmark_tracking_error, benchmark_information_ratio,
                    spy_alpha, spy_beta, spy_active_return, spy_tracking_error, spy_information_ratio,
                    win_rate, profit_factor, turnover_ratio, total_cost, cost_pct, realized_pnl, unrealized_pnl,
                    initial_equity, final_equity, degradation_json, cost_sensitivity_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                payload,
            )

    def query_backtest_runs(self, strategy: Optional[str] = None, limit: int = 50) -> pd.DataFrame:
        clauses = []
        params: List[Any] = []
        if strategy:
            clauses.append("strategy = ?")
            params.append(strategy)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"""
            SELECT run_id, ts, strategy, start_date, end_date, benchmark_label,
                   total_return, cagr, sharpe_ratio, rolling_sharpe_63, rolling_sharpe_252,
                   max_drawdown, max_time_to_recovery_days, benchmark_alpha, spy_alpha,
                   benchmark_active_return, spy_active_return,
                   win_rate, profit_factor, turnover_ratio, total_cost, cost_pct,
                   realized_pnl, unrealized_pnl, initial_equity, final_equity
            FROM backtest_runs
            {where}
            ORDER BY ts DESC
            LIMIT ?
        """
        params.append(limit)
        with self._connect() as conn:
            return pd.read_sql_query(query, conn, params=params)

    def compare_backtest_runs(self, run_ids: Iterable[str]) -> pd.DataFrame:
        ids = [str(rid) for rid in run_ids if str(rid).strip()]
        if not ids:
            return pd.DataFrame()
        placeholders = ",".join(["?"] * len(ids))
        query = f"""
            SELECT run_id, ts, strategy, start_date, end_date, benchmark_label,
                   total_return, cagr, volatility, sharpe_ratio, rolling_sharpe_63, rolling_sharpe_252,
                   sortino_ratio, calmar_ratio, max_drawdown, max_drawdown_duration_days, max_time_to_recovery_days,
                   benchmark_alpha, benchmark_active_return, benchmark_information_ratio,
                   spy_alpha, spy_active_return, spy_information_ratio,
                   win_rate, profit_factor, turnover_ratio, total_cost, cost_pct, realized_pnl, unrealized_pnl,
                   initial_equity, final_equity
            FROM backtest_runs
            WHERE run_id IN ({placeholders})
            ORDER BY ts DESC
        """
        with self._connect() as conn:
            return pd.read_sql_query(query, conn, params=ids)

    def get_strategy_comparison(self) -> pd.DataFrame:
        with self._connect() as conn:
            return pd.read_sql_query(
                """
                SELECT strategy,
                       COUNT(*) AS total_trades,
                       SUM(CASE WHEN action = 'BUY' THEN 1 ELSE 0 END) AS buys,
                       SUM(CASE WHEN action = 'SELL' THEN 1 ELSE 0 END) AS sells,
                       SUM(commission + slippage + fx_cost) AS transaction_cost,
                       AVG(value) AS avg_notional
                FROM trades
                GROUP BY strategy
                ORDER BY total_trades DESC
                """,
                conn,
            )


class MetricsStoreAdapter:
    """Interface for a time-series metrics sink."""

    def write_metric(self, name: str, value: float, ts: datetime, tags: Optional[Dict[str, str]] = None) -> None:
        raise NotImplementedError

    def query_metrics(
        self,
        name: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        tags: Optional[Dict[str, str]] = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        raise NotImplementedError


class SQLiteMetricsStore(MetricsStoreAdapter):
    """Built-in time-series store for metrics when external TSDB is unavailable."""

    def __init__(self, db_path: str = "state/metrics.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._run_migrations(conn)

    def _run_migrations(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER NOT NULL,
                applied_at TEXT NOT NULL,
                description TEXT
            )
            """
        )
        row = conn.execute("SELECT MAX(version) AS version FROM schema_version").fetchone()
        current_version = int(row["version"]) if row and row["version"] is not None else 0
        migrations = [
            (1, "create metrics schema baseline", self._migrate_v1),
            (2, "normalize metric tags into metric_tags table", self._migrate_v2),
        ]
        for version, description, fn in migrations:
            if version <= current_version:
                continue
            fn(conn)
            conn.execute(
                "INSERT INTO schema_version (version, applied_at, description) VALUES (?, ?, ?)",
                (version, datetime.now(timezone.utc).isoformat(), description),
            )

    @staticmethod
    def _migrate_v1(conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                name TEXT NOT NULL,
                value REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS metric_tags (
                metric_id INTEGER NOT NULL REFERENCES metrics(id),
                key TEXT NOT NULL,
                value TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_metrics_name_ts ON metrics(name, ts);
            CREATE INDEX IF NOT EXISTS idx_metric_tags_kv ON metric_tags(key, value);
            CREATE INDEX IF NOT EXISTS idx_metric_tags_metric_id ON metric_tags(metric_id);
            """
        )

    def _migrate_v2(self, conn: sqlite3.Connection) -> None:
        metrics_exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'metrics'"
        ).fetchone()
        if not metrics_exists:
            return

        existing = {row["name"] for row in conn.execute("PRAGMA table_info(metrics)").fetchall()}
        if "tags_json" not in existing:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS metric_tags (
                    metric_id INTEGER NOT NULL REFERENCES metrics(id),
                    key TEXT NOT NULL,
                    value TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_metric_tags_kv ON metric_tags(key, value)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_metric_tags_metric_id ON metric_tags(metric_id)")
            return

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS metric_tags (
                metric_id INTEGER NOT NULL REFERENCES metrics(id),
                key TEXT NOT NULL,
                value TEXT NOT NULL
            )
            """
        )
        rows = conn.execute("SELECT id, tags_json FROM metrics").fetchall()
        tag_rows: List[tuple[int, str, str]] = []
        for row in rows:
            raw_tags = row["tags_json"]
            if not raw_tags:
                continue
            try:
                parsed = json.loads(raw_tags)
            except (TypeError, json.JSONDecodeError):
                continue
            if not isinstance(parsed, dict):
                continue
            for key, value in parsed.items():
                tag_rows.append((int(row["id"]), str(key), str(value)))
        if tag_rows:
            conn.executemany(
                "INSERT INTO metric_tags (metric_id, key, value) VALUES (?, ?, ?)",
                tag_rows,
            )

        conn.execute("DROP TABLE IF EXISTS metrics_new")
        conn.execute(
            """
            CREATE TABLE metrics_new (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                name TEXT NOT NULL,
                value REAL NOT NULL
            )
            """
        )
        conn.execute("INSERT INTO metrics_new (id, ts, name, value) SELECT id, ts, name, value FROM metrics")
        conn.execute("DROP TABLE metrics")
        conn.execute("ALTER TABLE metrics_new RENAME TO metrics")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_name_ts ON metrics(name, ts)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_metric_tags_kv ON metric_tags(key, value)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_metric_tags_metric_id ON metric_tags(metric_id)")

    def write_metric(self, name: str, value: float, ts: datetime, tags: Optional[Dict[str, str]] = None) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "INSERT INTO metrics (ts, name, value) VALUES (?, ?, ?)",
                (ts.isoformat(), name, value),
            )
            metric_id = int(cursor.lastrowid)
            tag_rows = [(metric_id, str(k), str(v)) for k, v in (tags or {}).items()]
            if tag_rows:
                conn.executemany(
                    "INSERT INTO metric_tags (metric_id, key, value) VALUES (?, ?, ?)",
                    tag_rows,
                )

    def query_metrics(
        self,
        name: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        tags: Optional[Dict[str, str]] = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        clauses = ["m.name = ?"]
        params: List[Any] = [name]

        if start:
            clauses.append("m.ts >= ?")
            params.append(start.isoformat())
        if end:
            clauses.append("m.ts <= ?")
            params.append(end.isoformat())

        if tags:
            for key, value in tags.items():
                clauses.append(
                    """
                    EXISTS (
                        SELECT 1
                        FROM metric_tags mt
                        WHERE mt.metric_id = m.id AND mt.key = ? AND mt.value = ?
                    )
                    """
                )
                params.extend([str(key), str(value)])

        where = "WHERE " + " AND ".join(clauses)
        query = f"SELECT m.id, m.ts, m.name, m.value FROM metrics m {where} ORDER BY m.ts ASC LIMIT ?"
        params.append(limit)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            df = pd.read_sql_query(query, conn, params=params)
            if df.empty:
                df["tags_json"] = pd.Series(dtype="object")
                return df.drop(columns=["id"], errors="ignore")
            metric_ids = [int(x) for x in df["id"].tolist()]
            placeholders = ",".join(["?"] * len(metric_ids))
            tag_query = (
                "SELECT metric_id, key, value FROM metric_tags "
                f"WHERE metric_id IN ({placeholders}) ORDER BY metric_id ASC"
            )
            tag_rows = conn.execute(tag_query, metric_ids).fetchall()

        tags_by_id: Dict[int, Dict[str, str]] = {}
        for row in tag_rows:
            metric_id = int(row["metric_id"])
            if metric_id not in tags_by_id:
                tags_by_id[metric_id] = {}
            tags_by_id[metric_id][str(row["key"])] = str(row["value"])
        df["tags_json"] = df["id"].map(lambda metric_id: json.dumps(tags_by_id.get(int(metric_id), {})))
        df = df.drop(columns=["id"]).reset_index(drop=True)

        return df


def create_metrics_store(config: Dict[str, Any]) -> MetricsStoreAdapter:
    return SQLiteMetricsStore(config.get("sqlite_path", "state/metrics.db"))
