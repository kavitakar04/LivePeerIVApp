import os, sys
import pandas as pd
from unittest.mock import patch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.historical_saver import get_last_coverage_report, save_for_tickers


def test_duplicate_columns_are_dropped():
    records_holder = {}

    def fake_download_raw_option_data(ticker, max_expiries=8):
        # Minimal non-empty frame so we exercise the rest of the pipeline
        return pd.DataFrame({"x": [1]})

    def fake_enrich_quotes(raw, r=0.0, q=0.0, underlying_lookback_days=365):
        # Create DataFrame with duplicate column names
        return pd.DataFrame([[1, 2]], columns=["a", "a"])

    def fake_insert_quotes(conn, records):
        records_holder["records"] = records
        return len(records)

    with patch("data.historical_saver.get_conn", lambda: object()), \
         patch("data.historical_saver.ensure_initialized", lambda conn: None), \
         patch("data.historical_saver.download_raw_option_data", fake_download_raw_option_data), \
         patch("data.historical_saver.enrich_quotes", fake_enrich_quotes), \
         patch("data.historical_saver.insert_quotes", fake_insert_quotes):
        total = save_for_tickers(["TST"])
        assert total == 1

    # After conversion to records only a single key should remain
    assert list(records_holder["records"][0].keys()) == ["a"]


def test_comparison_ingestion_fetches_shared_native_expiries_for_all_tickers():
    calls = {}
    provider = {
        "SPXL": ["2026-05-01", "2026-05-15", "2026-06-18", "2026-08-21", "2026-11-20"],
        "UPRO": ["2026-05-01", "2026-05-15", "2026-06-18", "2026-08-21", "2026-11-20"],
        "URTY": ["2026-05-15", "2026-06-18", "2026-08-21", "2026-11-20"],
    }

    def fake_get_available_expiries(ticker):
        return provider[ticker]

    def fake_download_raw_option_data(ticker, max_expiries=8, expiries=None):
        calls[ticker] = list(expiries or [])
        rows = []
        for expiry in expiries or []:
            rows.append({"ticker": ticker, "expiry": expiry, "asof_date": "2026-04-26"})
        return pd.DataFrame(rows)

    def fake_enrich_quotes(raw, r=0.0, q=0.0, underlying_lookback_days=365):
        return raw.copy()

    def fake_insert_quotes(conn, records):
        return len(records)

    with patch("data.historical_saver.get_conn", lambda: object()), \
         patch("data.historical_saver.ensure_initialized", lambda conn: None), \
         patch("data.historical_saver.get_available_expiries", fake_get_available_expiries), \
         patch("data.historical_saver.download_raw_option_data", fake_download_raw_option_data), \
         patch("data.historical_saver.enrich_quotes", fake_enrich_quotes), \
         patch("data.historical_saver.insert_quotes", fake_insert_quotes):
        total = save_for_tickers(["SPXL", "UPRO", "URTY"], max_expiries=10)

    expected = ["2026-05-15", "2026-06-18", "2026-08-21", "2026-11-20"]
    assert calls == {"SPXL": expected, "UPRO": expected, "URTY": expected}
    assert total == 12
    report = get_last_coverage_report()
    assert [row["ticker"] for row in report] == ["SPXL", "UPRO", "URTY"]
    assert all(row["requested_expiries"] == expected for row in report)
    assert all(row["missing_shared_expiries"] == [] for row in report)


def test_comparison_coverage_report_marks_missing_stored_shared_expiries():
    provider = {
        "AAA": ["2026-05-15", "2026-06-18"],
        "BBB": ["2026-05-15", "2026-06-18"],
    }

    def fake_download_raw_option_data(ticker, max_expiries=8, expiries=None):
        fetched = ["2026-05-15"] if ticker == "BBB" else list(expiries or [])
        return pd.DataFrame({"ticker": ticker, "expiry": fetched, "asof_date": "2026-04-26"})

    with patch("data.historical_saver.get_conn", lambda: object()), \
         patch("data.historical_saver.ensure_initialized", lambda conn: None), \
         patch("data.historical_saver.get_available_expiries", lambda ticker: provider[ticker]), \
         patch("data.historical_saver.download_raw_option_data", fake_download_raw_option_data), \
         patch("data.historical_saver.enrich_quotes", lambda raw, r=0.0, q=0.0, underlying_lookback_days=365: raw.copy()), \
         patch("data.historical_saver.insert_quotes", lambda conn, records: len(records)):
        save_for_tickers(["AAA", "BBB"], max_expiries=10)

    report = {row["ticker"]: row for row in get_last_coverage_report()}
    assert report["AAA"]["missing_shared_expiries"] == []
    assert report["BBB"]["missing_shared_expiries"] == ["2026-06-18"]


def test_underlying_lookback_days_are_forwarded_to_enrichment():
    seen = {}

    def fake_download_raw_option_data(ticker, max_expiries=8):
        return pd.DataFrame({"ticker": [ticker], "expiry": ["2026-05-15"], "asof_date": ["2026-04-26"]})

    def fake_enrich_quotes(raw, r=0.0, q=0.0, underlying_lookback_days=365):
        seen["underlying_lookback_days"] = underlying_lookback_days
        return raw.copy()

    with patch("data.historical_saver.get_conn", lambda: object()), \
         patch("data.historical_saver.ensure_initialized", lambda conn: None), \
         patch("data.historical_saver.download_raw_option_data", fake_download_raw_option_data), \
         patch("data.historical_saver.enrich_quotes", fake_enrich_quotes), \
         patch("data.historical_saver.insert_quotes", lambda conn, records: len(records)):
        save_for_tickers(["AAA"], underlying_lookback_days=800)

    assert seen["underlying_lookback_days"] == 800
