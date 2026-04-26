#!/usr/bin/env python3
"""Precompute cache entries for IVCorrelation.

Given a JSON file describing a list of tasks, compute the corresponding
artifacts and store them in the ``calc_cache`` SQLite table.  This allows
warming the cache from the command line ahead of GUI usage.

Example tasks file::

    [
      {"kind": "smile", "ticker": "AAPL", "asof": "2024-01-10", "T_days": 30, "model": "svi"},
      {"kind": "corr", "tickers": ["AAPL", "MSFT"], "asof": "2024-01-10"},
      {"kind": "spill", "tickers": ["AAPL", "MSFT"], "threshold": 0.1}
    ]

Run as::

    python scripts/warm_cache.py tasks.json --db-path data/calculations.db
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable, List, Dict

import pandas as pd

# Make project imports work when executed as a script
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.cache_io import compute_or_load
from analysis.smile_data_service import prepare_smile_data, get_smile_slice
from analysis.correlation_view import corr_by_expiry_rank
from analysis.settings import DEFAULT_ATM_BAND, DEFAULT_MAX_EXPIRIES
from analysis.spillover.vol_spillover import run_spillover, load_iv_data


def _warm_smile(task: Dict[str, Any], db_path: str) -> None:
    ticker = task["ticker"].upper()
    asof = task["asof"]
    T_days = float(task.get("T_days", 30))
    model = task.get("model", "svi")
    ci = float(task.get("ci", 68.0))
    overlay_synth = bool(task.get("overlay_synth", False))
    peers = task.get("peers")
    weights = task.get("weights")
    overlay_peers = bool(task.get("overlay_peers", False))
    max_expiries = int(task.get("max_expiries", DEFAULT_MAX_EXPIRIES))

    payload = {
        "ticker": ticker,
        "asof": pd.to_datetime(asof).floor("min").isoformat(),
        "model": model,
        "params": weights,
        "T_days": T_days,
    }

    def _builder() -> Any:
        return prepare_smile_data(
            target=ticker,
            asof=asof,
            T_days=T_days,
            model=model,
            ci=ci,
            overlay_synth=overlay_synth,
            peers=peers,
            weights=weights,
            overlay_peers=overlay_peers,
            max_expiries=max_expiries,
        )

    compute_or_load("smile", payload, _builder, db_path)


def _warm_corr(task: Dict[str, Any], db_path: str) -> None:
    tickers = [t.upper() for t in task["tickers"]]
    asof = task["asof"]
    max_expiries = int(task.get("max_expiries", DEFAULT_MAX_EXPIRIES))
    atm_band = float(task.get("atm_band", DEFAULT_ATM_BAND))

    payload = {
        "tickers": sorted(tickers),
        "asof": pd.to_datetime(asof).floor("min").isoformat(),
        "max_expiries": max_expiries,
        "atm_band": atm_band,
    }

    def _builder() -> Any:
        return corr_by_expiry_rank(
            get_slice=get_smile_slice,
            tickers=tickers,
            asof=asof,
            max_expiries=max_expiries,
            atm_band=atm_band,
        )

    compute_or_load("corr", payload, _builder, db_path)


def _warm_spill(task: Dict[str, Any], db_path: str) -> None:
    tickers = task.get("tickers")
    threshold = float(task.get("threshold", 0.10))
    lookback = int(task.get("lookback", 60))
    top_k = int(task.get("top_k", 3))
    horizons = task.get("horizons", (1, 3, 5))
    path = task.get("path", "data/iv_data.parquet")
    df = load_iv_data(path)

    payload = {
        "tickers": sorted([t.upper() for t in tickers]) if tickers else None,
        "threshold": threshold,
        "lookback": lookback,
        "top_k": top_k,
        "horizons": tuple(horizons),
        "asof": df["date"].max().floor("min").isoformat() if not df.empty else None,
    }

    def _builder() -> Any:
        return run_spillover(
            df,
            tickers=tickers,
            threshold=threshold,
            lookback=lookback,
            top_k=top_k,
            horizons=horizons,
        )

    compute_or_load("spill", payload, _builder, db_path)


def main() -> None:
    p = argparse.ArgumentParser(description="Warm calc_cache entries")
    p.add_argument("tasks", help="JSON file describing tasks")
    p.add_argument("--db-path", default="data/calculations.db", help="Path to cache DB")
    args = p.parse_args()

    with open(args.tasks) as fh:
        tasks = json.load(fh)

    for task in tasks:
        kind = task.get("kind")
        if kind == "smile":
            _warm_smile(task, args.db_path)
        elif kind == "corr":
            _warm_corr(task, args.db_path)
        elif kind == "spill":
            _warm_spill(task, args.db_path)
        else:
            print(f"Unknown task kind: {kind}")


if __name__ == "__main__":
    main()
