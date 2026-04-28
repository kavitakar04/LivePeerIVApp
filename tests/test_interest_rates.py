import sqlite3

import pandas as pd
import pytest

import data.interest_rates as interest_rates
from data.db_schema import init_db
from data.greeks import compute_all_greeks_df


def _memory_rates_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    return conn


def test_ticker_interest_rate_normalizes_imported_percent_to_decimal(monkeypatch):
    conn = _memory_rates_conn()
    monkeypatch.setattr(interest_rates, "get_conn", lambda: conn)
    conn.execute(
        """
        INSERT INTO ticker_interest_rates(
            ticker, rate_date, rate_value, source_file, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        ("AAA", "2026-01-01", 4.08, "ML_test.csv", "now", "now"),
    )
    conn.commit()

    assert interest_rates.get_ticker_interest_rate("AAA") == pytest.approx(0.0408)


def test_greeks_use_default_rate_without_extra_percent_conversion(monkeypatch):
    conn = _memory_rates_conn()
    monkeypatch.setattr(interest_rates, "get_conn", lambda: conn)
    raw = pd.DataFrame(
        {
            "ticker": ["NO_RATE"],
            "S": [100.0],
            "K": [100.0],
            "T": [0.5],
            "sigma": [0.25],
            "call_put": ["C"],
        }
    )

    out = compute_all_greeks_df(raw, use_ticker_rates=True)

    assert out.loc[0, "r"] == pytest.approx(interest_rates.DEFAULT_INTEREST_RATE)


def test_greeks_use_ticker_decimal_rate(monkeypatch):
    conn = _memory_rates_conn()
    monkeypatch.setattr(interest_rates, "get_conn", lambda: conn)
    conn.execute(
        """
        INSERT INTO ticker_interest_rates(
            ticker, rate_date, rate_value, source_file, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        ("AAA", "2026-01-01", 7.5, "ML_test.csv", "now", "now"),
    )
    conn.commit()
    raw = pd.DataFrame(
        {
            "ticker": ["AAA"],
            "S": [100.0],
            "K": [100.0],
            "T": [0.5],
            "sigma": [0.25],
            "call_put": ["C"],
        }
    )

    out = compute_all_greeks_df(raw, use_ticker_rates=True)

    assert out.loc[0, "r"] == pytest.approx(0.075)
