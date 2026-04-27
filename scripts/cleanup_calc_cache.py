"""Remove expired rows from the calc_cache table.

This script can be scheduled via cron to keep the cache table small.
Example cron entry to run nightly at 2am:
    0 2 * * * /usr/bin/python path/to/cleanup_calc_cache.py --db-path=/path/db.db
"""

from __future__ import annotations
import argparse
import sqlite3

SQL = "DELETE FROM calc_cache WHERE expires_at < strftime('%s','now')"


def cleanup(db_path: str) -> int:
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(SQL)
        conn.commit()
        return cur.rowcount


def main() -> None:
    parser = argparse.ArgumentParser(description="Cleanup expired calc_cache rows")
    parser.add_argument("--db-path", default="data/iv_data.db", help="Path to SQLite database")
    args = parser.parse_args()
    removed = cleanup(args.db_path)
    print(f"Removed {removed} expired rows from calc_cache")


if __name__ == "__main__":
    main()
