"""Maintenance jobs for derived database artifacts."""
from __future__ import annotations
import sqlite3
from datetime import date, datetime, timedelta
from typing import List, Dict

from analysis.jobs.analysis_background_tasks import SpilloverConfig, build_spillover_last_90d, ensure_spillover_table

WINDOW_DAYS = 90


def _today() -> date:
    return datetime.utcnow().date()


def _window_start(end: date) -> date:
    return end - timedelta(days=WINDOW_DAYS)


def ensure_indices(conn: sqlite3.Connection) -> None:
    conn.executescript("""
    -- Helpful for daily ATM aggregation from options_quotes
    CREATE INDEX IF NOT EXISTS idx_optq_atm ON options_quotes(is_atm, asof_date, ticker);
    CREATE INDEX IF NOT EXISTS idx_optq_tkr_dt ON options_quotes(ticker, asof_date);
    CREATE INDEX IF NOT EXISTS idx_px_tkr_dt ON underlying_prices(ticker, asof_date);
    """)
    conn.commit()


def list_all_distinct_tickers(conn: sqlite3.Connection) -> List[str]:
    q = """
    SELECT ticker FROM (
        SELECT DISTINCT ticker FROM options_quotes
        UNION
        SELECT DISTINCT ticker FROM underlying_prices
    )
    ORDER BY 1
    """
    return [r[0] for r in conn.execute(q)]


def prune_derived_outside_window(conn: sqlite3.Connection, keep_from: date) -> int:
    cur = conn.execute("DELETE FROM iv_spillover WHERE asof_date < ?", (keep_from.isoformat(),))
    conn.commit()
    return cur.rowcount


def maintain_last_90_days(db_path: str = "data/iv_data.db") -> Dict[str, int]:
    """
    Idempotent maintenance:
      - ensure tables & indices
      - detect any new tickers by appearance in raw tables (no separate universe needed)
      - recompute spillovers for trailing 90d (corr, beta, xgb)
      - prune derived outside window
    """
    with sqlite3.connect(db_path) as conn:
        ensure_spillover_table(conn)
        ensure_indices(conn)

        # New ticker “detection” is implicit: any ticker present in raw tables
        # is included in ATM IV panel => included in spillover build.

        stats = build_spillover_last_90d(SpilloverConfig(db_path=db_path))
        kept_from = _window_start(_today())
        pruned = prune_derived_outside_window(conn, kept_from)
        stats["pruned_iv_spillover_rows"] = pruned
        return stats
