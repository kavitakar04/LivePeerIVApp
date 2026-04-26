from __future__ import annotations
from typing import Iterable
import logging

from .db_utils import get_conn, ensure_initialized, insert_quotes
from .data_downloader import download_raw_option_data, get_available_expiries
from .data_pipeline import enrich_quotes
from .interest_rates import STANDARD_RISK_FREE_RATE, STANDARD_DIVIDEND_YIELD
from analysis.settings import DEFAULT_UNDERLYING_LOOKBACK_DAYS

logger = logging.getLogger(__name__)

LAST_COVERAGE_REPORT: list[dict] = []


def _limit_expiries(expiries: Iterable[str], max_expiries: int) -> list[str]:
    ordered = sorted(dict.fromkeys(str(expiry) for expiry in expiries))
    if max_expiries and max_expiries > 0:
        return ordered[: int(max_expiries)]
    return ordered


def build_comparison_expiry_plan(tickers: Iterable[str], max_expiries: int) -> tuple[dict[str, list[str]], list[str]]:
    """Return provider expiry lists plus the shared comparison expiry request set."""
    tickers_up = [str(t).upper().strip() for t in tickers if str(t).strip()]
    provider_expiries: dict[str, list[str]] = {}
    for ticker in tickers_up:
        try:
            provider_expiries[ticker] = [str(expiry) for expiry in get_available_expiries(ticker)]
        except Exception as exc:
            logger.warning("expiry list fetch failed ticker=%s reason=%s", ticker, exc)
            provider_expiries[ticker] = []

    if len(tickers_up) <= 1:
        ticker = tickers_up[0] if tickers_up else ""
        return provider_expiries, _limit_expiries(provider_expiries.get(ticker, []), max_expiries)

    expiry_sets = [set(provider_expiries.get(ticker, [])) for ticker in tickers_up]
    shared = set.intersection(*expiry_sets) if expiry_sets else set()
    return provider_expiries, _limit_expiries(shared, max_expiries)


def get_last_coverage_report() -> list[dict]:
    return [dict(row) for row in LAST_COVERAGE_REPORT]


def save_for_tickers(
    tickers: Iterable[str],
    max_expiries: int = 8,
    r: float = STANDARD_RISK_FREE_RATE,
    q: float = STANDARD_DIVIDEND_YIELD,
    underlying_lookback_days: int = DEFAULT_UNDERLYING_LOOKBACK_DAYS,
) -> int:
    tickers = [str(t).upper().strip() for t in tickers if str(t).strip()]
    conn = get_conn()
    ensure_initialized(conn)
    total = 0
    provider_expiries, requested_expiries = build_comparison_expiry_plan(tickers, max_expiries)
    coverage_report: list[dict] = []
    for t in tickers:
        download_kwargs = {"max_expiries": max_expiries}
        if len(tickers) > 1:
            download_kwargs["expiries"] = requested_expiries
        raw = download_raw_option_data(t, **download_kwargs)
        fetched_expiries = sorted(raw["expiry"].astype(str).unique().tolist()) if raw is not None and not raw.empty and "expiry" in raw else []
        stored_expiries: list[str] = []
        inserted = 0
        if raw is None or raw.empty:
            print(f"No raw rows for {t}")
        else:
            enriched = enrich_quotes(
                raw,
                r=r,
                q=q,
                underlying_lookback_days=underlying_lookback_days,
            )
            if enriched is None or enriched.empty:
                enriched = None
            if enriched is None:
                print(f"No enriched rows for {t}")
            else:
                if enriched.columns.duplicated().any():
                    enriched = enriched.loc[:, ~enriched.columns.duplicated()].copy()
                stored_expiries = sorted(enriched["expiry"].astype(str).unique().tolist()) if "expiry" in enriched else []
                inserted = insert_quotes(conn, enriched.to_dict(orient="records"))
                total += inserted
                print(f"Inserted {t}: {len(enriched)} rows")

        missing_shared = sorted(set(requested_expiries) - set(stored_expiries))
        report_row = {
            "ticker": t,
            "provider_expiries": provider_expiries.get(t, []),
            "requested_expiries": list(requested_expiries),
            "fetched_expiries": fetched_expiries,
            "stored_expiries": stored_expiries,
            "missing_shared_expiries": missing_shared,
            "inserted_rows": inserted,
        }
        coverage_report.append(report_row)
        logger.info(
            "ingestion coverage ticker=%s provider_expiries=%s requested_expiries=%s "
            "fetched_expiries=%s stored_expiries=%s missing_shared_expiries=%s inserted_rows=%s",
            t,
            len(report_row["provider_expiries"]),
            report_row["requested_expiries"],
            report_row["fetched_expiries"],
            report_row["stored_expiries"],
            report_row["missing_shared_expiries"],
            inserted,
        )

    LAST_COVERAGE_REPORT.clear()
    LAST_COVERAGE_REPORT.extend(coverage_report)
    if coverage_report:
        print("Coverage report:")
        print("ticker | provider_expiries | requested_expiries | fetched_expiries | stored_expiries | missing_shared_expiries")
        for row in coverage_report:
            print(
                f"{row['ticker']} | "
                f"{len(row['provider_expiries'])} | "
                f"{','.join(row['requested_expiries']) or '-'} | "
                f"{','.join(row['fetched_expiries']) or '-'} | "
                f"{','.join(row['stored_expiries']) or '-'} | "
                f"{','.join(row['missing_shared_expiries']) or '-'}"
            )

    return total
