from __future__ import annotations
import sqlite3

SCHEMA_SQL = r"""
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS options_quotes (
    asof_date      TEXT    NOT NULL,
    ticker         TEXT    NOT NULL,
    expiry         TEXT    NOT NULL,
    strike         REAL    NOT NULL,
    call_put       TEXT    NOT NULL CHECK(call_put IN ('C','P')),
    iv             REAL,
    spot           REAL,
    ttm_years      REAL,
    moneyness      REAL,
    log_moneyness  REAL,
    delta          REAL,
    is_atm         INTEGER,
    volume         REAL,
    open_interest  REAL,
    bid            REAL,
    ask            REAL,
    mid            REAL,
    r              REAL,   -- NEW
    q              REAL,   -- NEW
    price          REAL,   -- NEW
    gamma          REAL,   -- NEW
    vega           REAL,   -- NEW
    theta          REAL,   -- NEW
    rho            REAL,   -- NEW
    d1             REAL,   -- NEW
    d2             REAL,   -- NEW
    vendor         TEXT    DEFAULT 'yfinance',
    PRIMARY KEY (asof_date, ticker, expiry, strike, call_put, vendor)
);

CREATE INDEX IF NOT EXISTS idx_quotes_ticker_date ON options_quotes(ticker, asof_date);
CREATE INDEX IF NOT EXISTS idx_quotes_expiry ON options_quotes(expiry);

CREATE TABLE IF NOT EXISTS underlying_prices (
    asof_date  TEXT NOT NULL,
    ticker     TEXT NOT NULL,
    close      REAL NOT NULL,
    PRIMARY KEY (asof_date, ticker)
);

CREATE TABLE IF NOT EXISTS ticker_groups (
    group_name     TEXT    PRIMARY KEY,
    target_ticker  TEXT    NOT NULL,
    peer_tickers   TEXT    NOT NULL,  -- JSON array of peer tickers
    description    TEXT,
    created_at     TEXT    NOT NULL,
    updated_at     TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ticker_groups_target ON ticker_groups(target_ticker);

CREATE TABLE IF NOT EXISTS interest_rates (
    rate_id        TEXT    PRIMARY KEY,
    rate_value     REAL    NOT NULL,
    description    TEXT,
    is_default     INTEGER DEFAULT 0,
    created_at     TEXT    NOT NULL,
    updated_at     TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS ticker_interest_rates (
    ticker         TEXT    NOT NULL,
    rate_date      TEXT    NOT NULL,
    rate_value     REAL    NOT NULL,
    fee            REAL,
    adjusted_float REAL,
    lender_count   INTEGER,
    borrow_status  TEXT,
    source_file    TEXT,
    created_at     TEXT    NOT NULL,
    updated_at     TEXT    NOT NULL,
    PRIMARY KEY (ticker, rate_date)
);

CREATE INDEX IF NOT EXISTS idx_interest_rates_default ON interest_rates(is_default);
CREATE INDEX IF NOT EXISTS idx_ticker_rates_ticker ON ticker_interest_rates(ticker);
CREATE INDEX IF NOT EXISTS idx_ticker_rates_date ON ticker_interest_rates(rate_date);
"""

MIGRATIONS = [
    ("r", "ALTER TABLE options_quotes ADD COLUMN r REAL"),
    ("q", "ALTER TABLE options_quotes ADD COLUMN q REAL"),
    ("price", "ALTER TABLE options_quotes ADD COLUMN price REAL"),
    ("gamma", "ALTER TABLE options_quotes ADD COLUMN gamma REAL"),
    ("vega", "ALTER TABLE options_quotes ADD COLUMN vega REAL"),
    ("theta", "ALTER TABLE options_quotes ADD COLUMN theta REAL"),
    ("rho", "ALTER TABLE options_quotes ADD COLUMN rho REAL"),
    ("d1", "ALTER TABLE options_quotes ADD COLUMN d1 REAL"),
    ("d2", "ALTER TABLE options_quotes ADD COLUMN d2 REAL"),
    ("open_interest", "ALTER TABLE options_quotes ADD COLUMN open_interest REAL"),
]


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_SQL)
    # Apply additive migrations if columns missing
    existing = {row[1] for row in conn.execute("PRAGMA table_info(options_quotes)")}
    for col, sql in MIGRATIONS:
        if col not in existing:
            try:
                conn.execute(sql)
            except sqlite3.OperationalError:
                pass
    conn.commit()
