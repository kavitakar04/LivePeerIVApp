import sqlite3
from datetime import datetime, timedelta

from data import data_pipeline


def _conn_with_underlying_rows(row_count: int, earliest_days_ago: int = 900) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE underlying_prices(asof_date TEXT, ticker TEXT, close REAL)")
    start = datetime.now() - timedelta(days=earliest_days_ago)
    if row_count <= 1:
        dates = [datetime.now()]
    else:
        step = earliest_days_ago / float(row_count - 1)
        dates = [start + timedelta(days=i * step) for i in range(row_count)]
    conn.executemany(
        "INSERT INTO underlying_prices(asof_date, ticker, close) VALUES (?, ?, ?)",
        [(d.strftime("%Y-%m-%d"), "AAA", 100.0 + i) for i, d in enumerate(dates)],
    )
    conn.commit()
    return conn


def test_history_period_for_requested_underlying_days():
    assert data_pipeline._history_period_for_lookback(30) == "1mo"
    assert data_pipeline._history_period_for_lookback(365) == "1y"
    assert data_pipeline._history_period_for_lookback(800) == "5y"


def test_underlying_coverage_uses_requested_lookback_without_500_day_cap(monkeypatch):
    conn = _conn_with_underlying_rows(row_count=600, earliest_days_ago=900)
    calls = []

    monkeypatch.setattr(data_pipeline, "get_conn", lambda: conn)
    monkeypatch.setattr(
        data_pipeline,
        "update_underlying_prices",
        lambda tickers, period="1y": calls.append((set(tickers), period)) or 123,
    )

    updated = data_pipeline.check_and_update_underlying_prices({"AAA"}, lookback_days=800)

    assert updated == 123
    assert calls == [({"AAA"}, "5y")]
