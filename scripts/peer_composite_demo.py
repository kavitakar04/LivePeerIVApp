#!/usr/bin/env python3
"""
Command-line peer-composite demo.

Examples:
  python scripts/peer_composite_demo.py --target SPY --peers QQQ IWM --weight-mode corr
  python scripts/peer_composite_demo.py --target SPY --peers QQQ IWM --export-dir out/peer_spy --no-show

Optional ingestion (if you want fresh data):
  python scripts/peer_composite_demo.py --ingest --tickers SPY QQQ IWM

"""

from __future__ import annotations
import argparse
import pandas as pd
import sys
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.peer_composite_service import PeerCompositeConfig, PeerCompositeBuilder
from display.plotting.peer_composite_viewer import show_peer_composite
from analysis.analysis_pipeline import ingest_and_process, available_dates
from analysis.settings import (
    DEFAULT_PILLAR_DAYS,
    DEFAULT_PILLAR_TOLERANCE_DAYS,
    DEFAULT_RV_LOOKBACK_DAYS,
)

def parse_args():
    p = argparse.ArgumentParser(description="Peer-composite surface demo")
    p.add_argument("--target", required=True, help="Target ticker")
    p.add_argument("--peers", nargs="+", required=True, help="Peer tickers")
    p.add_argument("--weight-mode", choices=["corr", "pca", "cosine", "equal", "custom"], default="corr")
    p.add_argument("--custom-weights", nargs="+", help="Ticker=weight pairs if weight-mode=custom")
    p.add_argument("--pillar-days", nargs="+", type=int, default=list(DEFAULT_PILLAR_DAYS[:4]))
    p.add_argument("--tenors", nargs="+", type=int, default=None, help="Override default tenors (days)")
    p.add_argument("--tolerance-days", type=float, default=DEFAULT_PILLAR_TOLERANCE_DAYS)
    p.add_argument("--lookback", type=int, default=DEFAULT_RV_LOOKBACK_DAYS)
    p.add_argument("--no-show", action="store_true", help="Do not display matplotlib viewer")
    p.add_argument("--export-dir", help="If provided, export artifacts to this directory")
    p.add_argument("--ingest", action="store_true", help="Ingest data before running")
    p.add_argument("--tickers", nargs="+", help="Tickers to ingest (default target+peers)")
    p.add_argument("--strict-date-intersection", action="store_true", help="Only keep dates where all peers have surfaces (DISABLED - always False)")
    return p.parse_args()


def parse_custom_weights(pairs):
    out = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Bad custom weight pair: {pair}")
        t, v = pair.split("=", 1)
        out[t.upper()] = float(v)
    return out


def main():
    args = parse_args()

    if args.ingest:
        ingest_list = args.tickers or [args.target] + args.peers
        print(f"Ingesting data for: {ingest_list}")
        ingest_and_process(ingest_list, max_expiries=6)

    # Basic validation: ensure we have at least one date for target
    dates = available_dates(ticker=args.target, most_recent_only=False)
    if not dates:
        print(f"No data available for {args.target}; aborting.")
        return

    custom_weights = None
    if args.weight_mode == "custom":
        if not args.custom_weights:
            raise SystemExit("Must supply --custom-weights ticker=weight ... for custom mode.")
        custom_weights = parse_custom_weights(args.custom_weights)

    cfg = PeerCompositeConfig(
        target=args.target.upper(),
        peers=tuple(t.upper() for t in args.peers),
        tenors=tuple(args.tenors) if args.tenors else None or (),
        tolerance_days=args.tolerance_days,
        lookback=args.lookback,
        weight_mode=args.weight_mode,
        strict_date_intersection=False,  # Always False - disabled filtering
    )

    # Fill missing tenors from default if not specified
    if not cfg.tenors:
        from analysis.peer_composite_builder import DEFAULT_TENORS
        cfg.tenors = DEFAULT_TENORS

    builder = PeerCompositeBuilder(cfg)
    artifacts = builder.build_all(custom_weights=custom_weights)

    print("\n=== Weights ===")
    print(artifacts.weights)

    print("\n=== Meta ===")
    for k, v in artifacts.meta.items():
        print(f"{k}: {v}")

    # Recent RV summary
    if not artifacts.rv_metrics.empty:
        last_rv = (
            artifacts.rv_metrics.sort_values("asof_date")
            .groupby("pillar_days")
            .tail(1)
            .reset_index(drop=True)
        )
        print("\n=== Latest RV per Pillar ===")
        print(last_rv)

    if args.export_dir:
        print(f"\nExporting artifacts to {args.export_dir}")
        builder.export(artifacts, args.export_dir)

    if not args.no_show:
        show_peer_composite(artifacts, target=cfg.target)


if __name__ == "__main__":
    main()
