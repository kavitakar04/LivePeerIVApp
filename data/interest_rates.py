# data/interest_rates.py
from __future__ import annotations
import os
from typing import List, Optional, Tuple
from datetime import datetime

from data.db_utils import get_conn, check_db_health

"""Central definitions for interest rate settings.

This module provides standard fallback values for risk-free and
 dividend rates to keep the rest of the codebase tidy.
"""

STANDARD_RISK_FREE_RATE = 0.0408  # 4.08%
DEFAULT_INTEREST_RATE = 0.0408  # 4.08%
STANDARD_DIVIDEND_YIELD = 0.0


def _rate_to_decimal(rate_value: float | int | None) -> Optional[float]:
    """Normalize stored rate values to decimal units.

    Global rates are stored as decimals (0.0408), while imported ML borrow
    rates are stored as percentage values (4.08). Accept both so Greek
    calculations always receive consistent decimal units.
    """
    if rate_value is None:
        return None
    rate = float(rate_value)
    return rate / 100.0 if abs(rate) > 1.0 else rate


def get_default_dividend_yield() -> float:
    """Get the default dividend yield value."""
    return STANDARD_DIVIDEND_YIELD


def create_default_interest_rates() -> None:
    """Create the default interest rate (4.08%) if no rates exist."""
    conn = get_conn()

    # Check if any rates exist
    existing = conn.execute("SELECT COUNT(*) FROM interest_rates").fetchone()[0]

    if existing == 0:
        # Create default rate
        now = datetime.now().isoformat()
        conn.execute(
            """
            INSERT INTO interest_rates (rate_id, rate_value, description, is_default, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            ("default", DEFAULT_INTEREST_RATE, "Default interest rate (4.08%)", 1, now, now),
        )
        conn.commit()
        check_db_health(conn)


def save_interest_rate(rate_id: str, rate_value: float, description: str = "", is_default: bool = False) -> None:
    """Save or update an interest rate."""
    conn = get_conn()
    now = datetime.now().isoformat()

    # If setting as default, first unset all other defaults
    if is_default:
        conn.execute("UPDATE interest_rates SET is_default = 0")

    # Insert or replace the rate
    conn.execute(
        """
        INSERT OR REPLACE INTO interest_rates
        (rate_id, rate_value, description, is_default, created_at, updated_at)
        VALUES (?, ?, ?, ?, COALESCE((SELECT created_at FROM interest_rates WHERE rate_id = ?), ?), ?)
    """,
        (rate_id, rate_value, description, int(is_default), rate_id, now, now),
    )
    conn.commit()
    check_db_health(conn)


def load_interest_rate(rate_id: str) -> Optional[Tuple[float, str, bool]]:
    """Load an interest rate by ID. Returns (rate_value, description, is_default) or None."""
    conn = get_conn()
    row = conn.execute(
        """
        SELECT rate_value, description, is_default
        FROM interest_rates
        WHERE rate_id = ?
    """,
        (rate_id,),
    ).fetchone()

    if row:
        return (float(row[0]), str(row[1] or ""), bool(row[2]))
    return None


def get_default_interest_rate() -> float:
    """Get the default interest rate value."""
    conn = get_conn()
    row = conn.execute("""
        SELECT rate_value
        FROM interest_rates
        WHERE is_default = 1
        ORDER BY updated_at DESC
        LIMIT 1
    """).fetchone()

    if row:
        return float(row[0])

    # If no default found, create and return the hardcoded default
    create_default_interest_rates()
    return DEFAULT_INTEREST_RATE


def list_interest_rates() -> List[Tuple[str, float, str, bool]]:
    """List all interest rates. Returns [(rate_id, rate_value, description, is_default), ...]"""
    conn = get_conn()
    rows = conn.execute("""
        SELECT rate_id, rate_value, description, is_default
        FROM interest_rates
        ORDER BY is_default DESC, rate_id ASC
    """).fetchall()

    return [(str(row[0]), float(row[1]), str(row[2] or ""), bool(row[3])) for row in rows]


def delete_interest_rate(rate_id: str) -> bool:
    """Delete an interest rate. Returns True if deleted, False if not found or is default."""
    conn = get_conn()

    # Check if it's the default rate
    row = conn.execute("SELECT is_default FROM interest_rates WHERE rate_id = ?", (rate_id,)).fetchone()
    if not row:
        return False

    if row[0]:  # is_default = 1
        # Don't allow deleting default rate
        return False

    conn.execute("DELETE FROM interest_rates WHERE rate_id = ?", (rate_id,))
    conn.commit()
    check_db_health(conn)
    return True


def set_default_interest_rate(rate_id: str) -> bool:
    """Set a specific interest rate as the default. Returns True if successful."""
    conn = get_conn()

    # Check if the rate exists
    exists = conn.execute("SELECT 1 FROM interest_rates WHERE rate_id = ?", (rate_id,)).fetchone()
    if not exists:
        return False

    # Unset all defaults, then set this one
    conn.execute("UPDATE interest_rates SET is_default = 0")
    conn.execute(
        "UPDATE interest_rates SET is_default = 1, updated_at = ? WHERE rate_id = ?",
        (datetime.now().isoformat(), rate_id),
    )
    conn.commit()
    check_db_health(conn)
    return True


def get_interest_rate_names() -> List[str]:
    """Get list of all interest rate IDs for dropdown menus."""
    conn = get_conn()
    rows = conn.execute("SELECT rate_id FROM interest_rates ORDER BY is_default DESC, rate_id ASC").fetchall()
    return [str(row[0]) for row in rows]


# New ticker-specific interest rate functions


def save_ticker_interest_rates(ticker_rates: List[dict], source_file: str) -> int:
    """Save ticker-specific interest rates from ML data.

    ticker_rates: List of dicts with keys: ticker, rate_date, rate_value,
        fee, adjusted_float, lender_count, borrow_status
    source_file: Name of the source file (e.g., 'ML_aug08.csv')

    Returns: Number of records inserted/updated
    """
    conn = get_conn()
    now = datetime.now().isoformat()

    rows = []
    for rate_data in ticker_rates:
        rows.append(
            (
                rate_data["ticker"],
                rate_data["rate_date"],
                float(rate_data["rate_value"]),
                rate_data.get("fee"),
                rate_data.get("adjusted_float"),
                rate_data.get("lender_count"),
                rate_data.get("borrow_status"),
                source_file,
                now,
                now,
            )
        )

    conn.executemany(
        """
        INSERT OR REPLACE INTO ticker_interest_rates
        (ticker, rate_date, rate_value, fee, adjusted_float, lender_count,
         borrow_status, source_file, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        rows,
    )
    conn.commit()
    check_db_health(conn)
    return len(rows)


def get_ticker_interest_rate(ticker: str, rate_date: Optional[str] = None) -> Optional[float]:
    """Get the interest rate for a specific ticker in decimal units.

    If rate_date is provided, gets the rate for that specific date.
    If rate_date is None, gets the most recent rate for the ticker.
    Falls back to the default global rate if no ticker-specific rate is found.
    """
    conn = get_conn()

    if rate_date:
        # Get rate for specific date
        row = conn.execute(
            """
            SELECT rate_value FROM ticker_interest_rates
            WHERE ticker = ? AND rate_date = ?
        """,
            (ticker, rate_date),
        ).fetchone()
    else:
        # Get most recent rate for ticker
        row = conn.execute(
            """
            SELECT rate_value FROM ticker_interest_rates
            WHERE ticker = ?
            ORDER BY rate_date DESC
            LIMIT 1
        """,
            (ticker,),
        ).fetchone()

    if row:
        return _rate_to_decimal(row[0])

    # Fallback to default global rate
    return _rate_to_decimal(get_default_interest_rate())


def get_most_recent_ticker_rates_date() -> Optional[str]:
    """Get the most recent rate_date across all tickers."""
    conn = get_conn()
    row = conn.execute("SELECT MAX(rate_date) FROM ticker_interest_rates").fetchone()
    return row[0] if row and row[0] else None


def list_tickers_with_rates(rate_date: Optional[str] = None) -> List[Tuple[str, float, Optional[str]]]:
    """List all tickers that have interest rate data.

    Returns: [(ticker, rate_value, borrow_status), ...]
    """
    conn = get_conn()

    if rate_date:
        sql = """
            SELECT ticker, rate_value, borrow_status
            FROM ticker_interest_rates
            WHERE rate_date = ?
            ORDER BY ticker
        """
        params = [rate_date]
    else:
        # Get most recent rate for each ticker
        sql = """
            SELECT t1.ticker, t1.rate_value, t1.borrow_status
            FROM ticker_interest_rates t1
            INNER JOIN (
                SELECT ticker, MAX(rate_date) as max_date
                FROM ticker_interest_rates
                GROUP BY ticker
            ) t2 ON t1.ticker = t2.ticker AND t1.rate_date = t2.max_date
            ORDER BY t1.ticker
        """
        params = []

    rows = conn.execute(sql, params).fetchall()
    return [(str(row[0]), float(row[1]), str(row[2]) if row[2] else None) for row in rows]


def get_ticker_rate_history(ticker: str) -> List[Tuple[str, float, Optional[str]]]:
    """Get the rate history for a specific ticker.

    Returns: [(rate_date, rate_value, borrow_status), ...] ordered by date descending
    """
    conn = get_conn()
    rows = conn.execute(
        """
        SELECT rate_date, rate_value, borrow_status
        FROM ticker_interest_rates
        WHERE ticker = ?
        ORDER BY rate_date DESC
    """,
        (ticker,),
    ).fetchall()

    return [(str(row[0]), float(row[1]), str(row[2]) if row[2] else None) for row in rows]


# ML File Processing Functions


def find_most_recent_ml_file(ml_dir: str = "data/ML_Rates") -> Optional[Tuple[str, str]]:
    """Find the most recent ML_rate CSV file.

    Returns: (file_path, rate_date) or (None, None) if no files found
    """
    import glob

    pattern = os.path.join(ml_dir, "ML_*.csv")
    ml_files = glob.glob(pattern)

    if not ml_files:
        return None, None

    # Parse dates from filenames and find the most recent
    file_dates = []
    for file in ml_files:
        filename = os.path.basename(file)
        date_str = filename.replace("ML_", "").replace(".csv", "")
        try:
            # Parse date like "aug08" -> 2025-08-08
            month_map = {
                "jan": "01",
                "feb": "02",
                "mar": "03",
                "apr": "04",
                "may": "05",
                "jun": "06",
                "jul": "07",
                "aug": "08",
                "sep": "09",
                "oct": "10",
                "nov": "11",
                "dec": "12",
            }
            month = date_str[:3].lower()
            day = date_str[3:].zfill(2)
            if month in month_map:
                date_obj = datetime.strptime(f"2025-{month_map[month]}-{day}", "%Y-%m-%d")
                file_dates.append((date_obj, file, f"2025-{month_map[month]}-{day}"))
        except (ValueError, KeyError):
            continue

    if not file_dates:
        return None, None

    # Sort by date and get the most recent
    file_dates.sort(key=lambda x: x[0])
    most_recent_file = file_dates[-1][1]
    most_recent_date = file_dates[-1][2]
    return most_recent_file, most_recent_date


def parse_ml_file(file_path: str) -> List[dict]:
    """Parse ML CSV file and extract ticker-specific rates.

    Returns: List of dicts with ticker rate information
    """
    import pandas as pd

    try:
        # Read the CSV file
        df = pd.read_csv(file_path)

        # Filter to only the first row for each ticker (the main data row)
        # The subsequent rows for each ticker appear to be historical dates
        ticker_rates = []

        for ticker in df["Stock"].unique():
            if pd.isna(ticker):
                continue

            ticker_data = df[df["Stock"] == ticker]

            # Get the first row which should have the current rate data
            first_row = ticker_data.iloc[0]

            # Skip if Rate is not a number (likely a date string)
            try:
                rate_value = float(first_row["Rate"])
                # Skip if rate is NaN
                if pd.isna(rate_value):
                    continue
            except (ValueError, TypeError):
                continue

            rate_info = {
                "ticker": str(ticker).strip(),
                "rate_value": rate_value,
                "fee": first_row.get("Fee") if pd.notna(first_row.get("Fee")) else None,
                "adjusted_float": (
                    first_row.get("Adjusted Float") if pd.notna(first_row.get("Adjusted Float")) else None
                ),
                "lender_count": first_row.get("Lender Count") if pd.notna(first_row.get("Lender Count")) else None,
                "borrow_status": first_row.get("Borrow Status") if pd.notna(first_row.get("Borrow Status")) else None,
            }

            ticker_rates.append(rate_info)

        return ticker_rates

    except Exception as e:
        print(f"Error parsing ML file: {e}")
        return []


def update_ticker_rates_from_ml(ml_dir: str = "data/ML_Rates", interactive: bool = True) -> bool:
    """Update the ticker-specific rates database with ML data.

    Args:
        ml_dir: Directory containing ML_*.csv files
        interactive: If True, prompts user for confirmation before saving

    Returns: True if rates were updated, False otherwise
    """
    from .db_schema import init_db
    import pandas as pd

    # Initialize database to ensure tables exist
    conn = get_conn()
    init_db(conn)
    conn.close()

    # Find the most recent ML file
    ml_file, rate_date = find_most_recent_ml_file(ml_dir)
    if not ml_file or not rate_date:
        if interactive:
            print("No ML_rate files found!")
        return False

    # Parse the file
    ticker_rates = parse_ml_file(ml_file)
    if not ticker_rates:
        if interactive:
            print("No ticker rates found in the file!")
        return False

    # Add the rate_date to each ticker rate
    for rate_info in ticker_rates:
        rate_info["rate_date"] = rate_date

    if interactive:
        print("=== Updating Ticker-Specific Interest Rates Database ===")
        print(f"Most recent ML_rate file: {ml_file} (date: {rate_date})")
        print(f"Parsing {ml_file}...")
        print(f"Found rates for {len(ticker_rates)} tickers")

        # Show some statistics
        rates = [r["rate_value"] for r in ticker_rates if not pd.isna(r["rate_value"])]
        if rates:
            print("\nRate Statistics:")
            print(f"  Total tickers: {len(ticker_rates)}")
            print(f"  Rate range: {min(rates):.4f} to {max(rates):.4f}")
            print(f"  Mean rate: {sum(rates)/len(rates):.4f}")

        # Show sample of ticker rates
        print("\nSample ticker rates:")
        for i, rate_info in enumerate(ticker_rates[:10]):
            print(f"  {rate_info['ticker']}: {rate_info['rate_value']:.4f}% ({rate_info.get('borrow_status', 'N/A')})")

        if len(ticker_rates) > 10:
            print(f"  ... and {len(ticker_rates) - 10} more")

        # Ask for confirmation
        response = input(f"\nSave these {len(ticker_rates)} ticker rates to database? (y/N): ").strip().lower()
        if response not in ["y", "yes"]:
            print("Cancelled.")
            return False

    # Save to database
    filename = os.path.basename(ml_file)
    records_saved = save_ticker_interest_rates(ticker_rates, filename)

    if interactive:
        print(f"✓ Saved {records_saved} ticker-specific rates for date {rate_date}")
        print(f"\nDatabase updated. Recent ticker rates date: {get_most_recent_ticker_rates_date()}")

        # Show some examples of the saved data
        print("\nSample ticker rates from database:")
        sample_tickers = list_tickers_with_rates(rate_date)[:5]
        for ticker, rate, status in sample_tickers:
            print(f"  {ticker}: {rate:.4f}% ({status or 'N/A'})")

    return True
