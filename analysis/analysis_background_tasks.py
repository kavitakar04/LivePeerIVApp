# analysis/analysis_background_tasks.py
from __future__ import annotations
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Iterable, List, Dict, Tuple, Optional

import numpy as np
import pandas as pd

from analysis.settings import (
    DEFAULT_BACKGROUND_MIN_SAMPLES,
    DEFAULT_BACKGROUND_RECOMPUTE_TAIL_DAYS,
    DEFAULT_BACKGROUND_SPILLOVER_HORIZONS,
    DEFAULT_BACKGROUND_WINDOW_DAYS,
)

HORIZONS = DEFAULT_BACKGROUND_SPILLOVER_HORIZONS
WINDOW_DAYS = DEFAULT_BACKGROUND_WINDOW_DAYS


@dataclass(frozen=True)
class SpilloverConfig:
    db_path: str = "data/iv_data.db"
    horizons: Tuple[int, ...] = HORIZONS
    recompute_tail_days: int = DEFAULT_BACKGROUND_RECOMPUTE_TAIL_DAYS
    min_samples: int = DEFAULT_BACKGROUND_MIN_SAMPLES


def _today() -> date:
    return datetime.utcnow().date()


def _window_start(end: date) -> date:
    return end - timedelta(days=WINDOW_DAYS)


def ensure_spillover_table(conn: sqlite3.Connection) -> None:
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS iv_spillover (
      asof_date DATE NOT NULL,
      target     TEXT NOT NULL,
      peer       TEXT NOT NULL,
      horizon_d  INTEGER NOT NULL,
      method     TEXT NOT NULL,
      value      REAL,
      sample_n   INTEGER,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      UNIQUE(asof_date, target, peer, horizon_d, method) ON CONFLICT REPLACE
    );
    CREATE INDEX IF NOT EXISTS idx_ivspill_dt ON iv_spillover(asof_date);
    CREATE INDEX IF NOT EXISTS idx_ivspill_target ON iv_spillover(target, asof_date);
    CREATE INDEX IF NOT EXISTS idx_ivspill_peer ON iv_spillover(peer, asof_date);
    """)
    conn.commit()


def distinct_tickers_in_db(conn: sqlite3.Connection) -> List[str]:
    q = """
    SELECT ticker FROM (
        SELECT DISTINCT ticker FROM options_quotes
        UNION
        SELECT DISTINCT ticker FROM underlying_prices
        UNION
        SELECT DISTINCT ticker FROM ticker_interest_rates
    )
    ORDER BY 1
    """
    return [r[0] for r in conn.execute(q)]


def daily_atm_iv_panel(conn: sqlite3.Connection, start: date, end: date) -> pd.DataFrame:
    """
    Returns DataFrame: columns [asof_date, ticker, atm_iv]
    atm_iv = median iv where is_atm=1 for each asof_date,ticker.
    """
    # SQLite has no MEDIAN by default; fallback to percentile_cont via pandas aggregation.
    df = pd.read_sql_query(
        """
        SELECT asof_date, ticker, iv
        FROM options_quotes
        WHERE is_atm = 1
          AND asof_date BETWEEN ? AND ?
          AND iv IS NOT NULL
        """,
        conn,
        params=(start, end),
        parse_dates=["asof_date"],
    )
    if df.empty:
        return pd.DataFrame(columns=["asof_date", "ticker", "atm_iv"])
    atm = df.groupby(["asof_date", "ticker"])["iv"].median().rename("atm_iv").reset_index()
    return atm


def compute_iv_returns(atm_df: pd.DataFrame) -> pd.DataFrame:
    """
    Wide daily panel of log returns: index asof_date, columns tickers.
    """
    if atm_df.empty:
        return pd.DataFrame()
    pivot = atm_df.pivot(index="asof_date", columns="ticker", values="atm_iv").sort_index()
    # Guard against non-positive IVs
    pivot = pivot.replace([np.inf, -np.inf], np.nan)
    pivot = pivot.where(pivot > 0.0)
    ivret = np.log(pivot).diff()
    return ivret


def _shift_series(s: pd.Series, d: int) -> pd.Series:
    return s.shift(-d)  # peer responses “d days ahead” relative to target t


def _pairwise_corr_beta(
    ivret: pd.DataFrame, horizons: Iterable[int]
) -> List[Tuple[pd.Timestamp, str, str, int, float, int, float, int]]:
    """
    For each date,ticker pair and horizon:
      - corr: corr(target_ret_t, peer_ret_{t..t+d})
      - beta: OLS peer_ret_{t..t+d} ~ target_ret_t  (slope)
    Returns rows for bulk insert into iv_spillover.
    """
    out_corr, out_beta = [], []
    if ivret.empty:
        return out_corr + out_beta

    ivret.index
    tickers = list(ivret.columns)

    for h in horizons:
        shifted = ivret.apply(lambda s: _shift_series(s, h))
        # We want metrics anchored at date t (current row) where both are present
        for i, tgt in enumerate(tickers):
            x = ivret[tgt]
            if x.isna().all():
                continue
            for j, peer in enumerate(tickers):
                if peer == tgt:
                    continue
                y = shifted[peer]
                pair = pd.concat([x, y], axis=1, keys=["x", "y"]).dropna()
                n = len(pair)
                if n < 20:
                    continue
                # corr
                cval = float(pair["x"].corr(pair["y"]))
                out_corr.append((None, tgt, peer, h, cval, n, None, None))
                # beta: y = a + b x
                vx = pair["x"].values
                vy = pair["y"].values
                denom = (vx**2).sum()
                if denom <= 0 or np.isnan(denom):
                    continue
                b = float(np.dot(vx, vy) / denom)
                out_beta.append((None, tgt, peer, h, b, n, None, None))

    return out_corr + out_beta


def _xgb_feature_importance(
    ivret: pd.DataFrame, horizons: Iterable[int]
) -> List[Tuple[pd.Timestamp, str, str, int, float, int, float, int]]:
    """
    Optional light XGB path: for each target & horizon, predict future target return
    from contemporaneous peers (and optionally lags), record feature gain per peer.
    Keeps it simple and fast; if xgboost unavailable, skip gracefully.
    """
    try:
        from xgboost import XGBRegressor
    except Exception:
        return []

    if ivret.empty:
        return []

    out = []
    tickers = list(ivret.columns)

    for tgt in tickers:
        # Simple design matrix: peers at t explaining target at t+h
        X_base = ivret.drop(columns=[tgt]).copy()
        for h in horizons:
            y = ivret[tgt].shift(-h)
            df = pd.concat([X_base, y.rename("y")], axis=1).dropna()
            if len(df) < 200 or X_base.shape[1] == 0:
                continue
            X = df.drop(columns=["y"]).values
            Y = df["y"].values
            model = XGBRegressor(
                n_estimators=200,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=1,
            )
            model.fit(X, Y)
            gains = getattr(model, "feature_importances_", None)
            if gains is None:
                continue
            peers = list(X_base.columns)
            n = len(df)
            for peer, g in zip(peers, gains):
                out.append((None, tgt, peer, h, float(g), n, None, None))
    return out


def _bulk_upsert_spillover(
    conn: sqlite3.Connection,
    rows: List[Tuple[Optional[pd.Timestamp], str, str, int, float, int, Optional[float], Optional[int]]],
    method: str,
) -> int:
    if not rows:
        return 0
    # Anchor on actual dates present in ivret index; since rows hold None for date,
    # we write rows across the window by using the latest date as of insert time.
    # Simpler: store by the *window end* date (today) to mark the measurement date.
    today = _today().isoformat()
    data = [(today, r[1], r[2], r[3], method, r[4], r[5]) for r in rows]
    conn.executemany(
        "INSERT INTO iv_spillover(asof_date,target,peer,horizon_d,method,value,sample_n) "
        "VALUES (?,?,?,?,?,?,?) "
        "ON CONFLICT(asof_date,target,peer,horizon_d,method) DO UPDATE SET "
        "value=excluded.value, sample_n=excluded.sample_n, created_at=CURRENT_TIMESTAMP",
        data,
    )
    conn.commit()
    return len(data)


def build_spillover_last_90d(cfg: SpilloverConfig) -> Dict[str, int]:
    """
    Main entry: compute corr, beta, and xgb over the rolling 90-day window.
    Recomputes the last `recompute_tail_days` to stabilize estimates.
    """
    with sqlite3.connect(cfg.db_path) as conn:
        ensure_spillover_table(conn)
        end = _today()
        start = _window_start(end)
        # Pull last 90d ATM IV and compute returns
        atm = daily_atm_iv_panel(conn, start, end)
        ivret = compute_iv_returns(atm)

        # Restrict to recent tail for recompute (but correlations use full 90d history)
        # We log the measurement under `asof_date = today`.
        out_stats = {"corr": 0, "beta": 0, "xgb": 0}

        # Corr + Beta
        rows = _pairwise_corr_beta(ivret, cfg.horizons)
        out_stats["corr"] = _bulk_upsert_spillover(conn, rows=[r for r in rows if r[6] is None], method="corr")
        out_stats["beta"] = _bulk_upsert_spillover(conn, rows=[r for r in rows if r[6] is None], method="beta")

        # XGB (optional; skip if xgboost missing)
        xgb_rows = _xgb_feature_importance(ivret, cfg.horizons)
        if xgb_rows:
            out_stats["xgb"] = _bulk_upsert_spillover(conn, xgb_rows, method="xgb")

        return out_stats
