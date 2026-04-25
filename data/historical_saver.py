from __future__ import annotations
from typing import Iterable

from .db_utils import get_conn, ensure_initialized, insert_quotes
from .data_downloader import download_raw_option_data
from .data_pipeline import enrich_quotes
from .interest_rates import STANDARD_RISK_FREE_RATE, STANDARD_DIVIDEND_YIELD


def save_for_tickers(
    tickers: Iterable[str],
    max_expiries: int = 8,
    r: float = STANDARD_RISK_FREE_RATE,
    q: float = STANDARD_DIVIDEND_YIELD,
) -> int:
    conn = get_conn()
    ensure_initialized(conn)
    total = 0
    for t in tickers:
        raw = download_raw_option_data(t, max_expiries=max_expiries)
        if raw is None or raw.empty:
            print(f"No raw rows for {t}")
            continue
        enriched = enrich_quotes(raw, r=r, q=q)
        if enriched is None or enriched.empty:
            print(f"No enriched rows for {t}")
            continue

        if enriched.columns.duplicated().any():
            enriched = enriched.loc[:, ~enriched.columns.duplicated()].copy()

        total += insert_quotes(conn, enriched.to_dict(orient="records"))
        print(f"Inserted {t}: {len(enriched)} rows")
    return total
