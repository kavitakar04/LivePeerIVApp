#!/usr/bin/env python3
"""Repair known option quote quality issues in the local SQLite database."""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.quote_quality import STORE_MAX_MONEYNESS, STORE_MIN_MONEYNESS

DEFAULT_DB = ROOT / "data" / "iv_data.db"


def repair(db_path: Path, *, backup: bool = True, clear_cache: bool = True) -> dict[str, int | str]:
    if not db_path.exists():
        raise FileNotFoundError(db_path)

    backup_path = ""
    if backup:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup_file = db_path.with_suffix(db_path.suffix + f".bak-{stamp}")
        shutil.copy2(db_path, backup_file)
        backup_path = str(backup_file)

    stats: dict[str, int | str] = {"backup_path": backup_path}
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("BEGIN")
        try:
            cur = conn.execute(
                """
                DELETE FROM options_quotes
                WHERE moneyness IS NULL
                   OR moneyness <= ?
                   OR moneyness >= ?
                   OR iv IS NULL
                   OR iv <= 0.01
                   OR iv >= 2.0
                   OR spot IS NULL
                   OR spot <= 0
                   OR strike IS NULL
                   OR strike <= 0
                   OR ttm_years IS NULL
                   OR ttm_years <= 0
                """,
                (STORE_MIN_MONEYNESS, STORE_MAX_MONEYNESS),
            )
            stats["deleted_out_of_bounds"] = int(cur.rowcount)

            cur = conn.execute("""
                DELETE FROM options_quotes
                WHERE bid IS NOT NULL
                  AND ask IS NOT NULL
                  AND bid > ask
                """)
            stats["deleted_crossed_markets"] = int(cur.rowcount)

            cur = conn.execute("""
                UPDATE options_quotes
                SET mid = (bid + ask) / 2.0
                WHERE bid IS NOT NULL
                  AND ask IS NOT NULL
                  AND bid <= ask
                  AND (mid IS NULL OR mid < bid OR mid > ask)
                """)
            stats["recomputed_mid"] = int(cur.rowcount)

            if clear_cache:
                cur = conn.execute("DELETE FROM calc_cache")
                stats["deleted_calc_cache"] = int(cur.rowcount)

            conn.commit()
        except Exception:
            conn.rollback()
            raise

        status = conn.execute("PRAGMA quick_check").fetchone()
        stats["quick_check"] = status[0] if status else "unknown"
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Repair known options quote quality issues.")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB)
    parser.add_argument("--no-backup", action="store_true", help="Do not create a timestamped DB backup")
    parser.add_argument("--keep-cache", action="store_true", help="Do not clear calc_cache after repairing raw rows")
    args = parser.parse_args()

    stats = repair(args.db_path, backup=not args.no_backup, clear_cache=not args.keep_cache)
    for key, value in stats.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
