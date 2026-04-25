#!/usr/bin/env python3
"""Audit raw market data, derived artifacts, and computation caches.

This script is intentionally read-only. It produces a Markdown report plus a
JSON payload so data quality checks can be repeated after ingestion or model
changes.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DB = ROOT / "data" / "iv_data.db"
DEFAULT_CALC_DB = ROOT / "data" / "calculations.db"
DEFAULT_REPORT = ROOT / "reports" / "data_quality_audit.md"
DEFAULT_JSON = ROOT / "reports" / "data_quality_audit.json"


def _connect(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(path)


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    return bool(row)


def _scalar(conn: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()) -> Any:
    row = conn.execute(sql, params).fetchone()
    return row[0] if row else None


def _df(conn: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()) -> pd.DataFrame:
    return pd.read_sql_query(sql, conn, params=params)


def _fmt_int(value: Any) -> str:
    try:
        return f"{int(value):,}"
    except Exception:
        return str(value)


def _pct(num: float, den: float) -> str:
    if not den:
        return "n/a"
    return f"{num / den:.1%}"


def _records(df: pd.DataFrame, limit: int | None = None) -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []
    out = df.head(limit).copy() if limit else df.copy()
    return json.loads(out.to_json(orient="records", date_format="iso"))


def audit_sqlite(path: Path) -> dict[str, Any]:
    item: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else 0,
        "quick_check": None,
        "tables": [],
    }
    if not path.exists():
        item["issues"] = ["database file missing"]
        return item

    conn = _connect(path)
    try:
        item["quick_check"] = _scalar(conn, "PRAGMA quick_check")
        tables = _df(
            conn,
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name",
        )["name"].tolist()
        item["tables"] = tables
        item["table_counts"] = {
            table: int(_scalar(conn, f"SELECT COUNT(*) FROM {table}") or 0)
            for table in tables
        }
        item["schemas"] = {
            table: _records(_df(conn, f"PRAGMA table_info({table})"))
            for table in tables
        }
    finally:
        conn.close()
    return item


def audit_options_db(path: Path) -> dict[str, Any]:
    db = audit_sqlite(path)
    if not db.get("exists"):
        return db

    conn = _connect(path)
    try:
        if not _table_exists(conn, "options_quotes"):
            db.setdefault("issues", []).append("options_quotes table missing")
            return db

        total = int(_scalar(conn, "SELECT COUNT(*) FROM options_quotes") or 0)
        db["options_summary"] = {
            "rows": total,
            "tickers": int(_scalar(conn, "SELECT COUNT(DISTINCT ticker) FROM options_quotes") or 0),
            "asof_dates": int(_scalar(conn, "SELECT COUNT(DISTINCT asof_date) FROM options_quotes") or 0),
            "min_asof": _scalar(conn, "SELECT MIN(asof_date) FROM options_quotes"),
            "max_asof": _scalar(conn, "SELECT MAX(asof_date) FROM options_quotes"),
            "vendors": _records(_df(conn, "SELECT vendor, COUNT(*) AS rows FROM options_quotes GROUP BY vendor ORDER BY rows DESC")),
        }

        db["nulls"] = _records(
            _df(
                conn,
                """
                SELECT
                  SUM(iv IS NULL) AS iv_null,
                  SUM(spot IS NULL) AS spot_null,
                  SUM(ttm_years IS NULL) AS ttm_null,
                  SUM(moneyness IS NULL) AS moneyness_null,
                  SUM(bid IS NULL) AS bid_null,
                  SUM(ask IS NULL) AS ask_null,
                  SUM(open_interest IS NULL) AS open_interest_null,
                  SUM(delta IS NULL) AS delta_null
                FROM options_quotes
                """,
            )
        )[0]

        db["bad_counts"] = _records(
            _df(
                conn,
                """
                SELECT
                  SUM(iv IS NOT NULL AND (iv <= 0 OR iv > 5)) AS bad_iv,
                  SUM(spot IS NOT NULL AND spot <= 0) AS bad_spot,
                  SUM(strike <= 0) AS bad_strike,
                  SUM(ttm_years IS NOT NULL AND ttm_years <= 0) AS bad_ttm,
                  SUM(moneyness IS NOT NULL AND (moneyness <= 0 OR moneyness > 5)) AS bad_moneyness,
                  SUM(bid IS NOT NULL AND bid < 0) AS bad_bid,
                  SUM(ask IS NOT NULL AND ask < 0) AS bad_ask,
                  SUM(bid IS NOT NULL AND ask IS NOT NULL AND bid > ask) AS crossed_market,
                  SUM(mid IS NOT NULL AND bid IS NOT NULL AND ask IS NOT NULL AND (mid < bid OR mid > ask)) AS mid_outside_market,
                  SUM(volume IS NOT NULL AND volume < 0) AS bad_volume,
                  SUM(open_interest IS NOT NULL AND open_interest < 0) AS bad_open_interest
                FROM options_quotes
                """,
            )
        )[0]

        db["ticker_coverage"] = _records(
            _df(
                conn,
                """
                SELECT ticker,
                       COUNT(*) AS rows,
                       COUNT(DISTINCT asof_date) AS asof_dates,
                       MIN(asof_date) AS min_asof,
                       MAX(asof_date) AS max_asof,
                       COUNT(DISTINCT expiry) AS expiries,
                       SUM(iv IS NOT NULL) AS iv_rows,
                       ROUND(AVG(CASE WHEN is_atm = 1 THEN 1.0 ELSE 0.0 END), 4) AS atm_share
                FROM options_quotes
                GROUP BY ticker
                ORDER BY rows DESC, ticker
                """
            )
        )

        db["ticker_date_gaps"] = _records(
            _df(
                conn,
                """
                SELECT ticker, asof_date,
                       COUNT(*) AS rows,
                       COUNT(DISTINCT expiry) AS expiries,
                       SUM(iv IS NOT NULL) AS iv_rows,
                       SUM(is_atm = 1) AS atm_rows,
                       ROUND(AVG(iv), 4) AS avg_iv
                FROM options_quotes
                GROUP BY ticker, asof_date
                HAVING rows < 20 OR expiries < 2 OR iv_rows < rows * 0.8 OR atm_rows = 0
                ORDER BY asof_date DESC, ticker
                LIMIT 200
                """
            ),
            limit=200,
        )

        db["duplicate_pk_groups"] = _records(
            _df(
                conn,
                """
                SELECT asof_date, ticker, expiry, strike, call_put, vendor, COUNT(*) AS n
                FROM options_quotes
                GROUP BY asof_date, ticker, expiry, strike, call_put, vendor
                HAVING n > 1
                LIMIT 50
                """
            )
        )

        db["sample_bad_rows"] = _records(
            _df(
                conn,
                """
                SELECT asof_date, ticker, expiry, strike, call_put, iv, spot, ttm_years,
                       moneyness, bid, ask, mid, volume, open_interest
                FROM options_quotes
                WHERE
                    (iv IS NOT NULL AND (iv <= 0 OR iv > 5))
                    OR (spot IS NOT NULL AND spot <= 0)
                    OR strike <= 0
                    OR (ttm_years IS NOT NULL AND ttm_years <= 0)
                    OR (moneyness IS NOT NULL AND (moneyness <= 0 OR moneyness > 5))
                    OR (bid IS NOT NULL AND ask IS NOT NULL AND bid > ask)
                    OR (mid IS NOT NULL AND bid IS NOT NULL AND ask IS NOT NULL AND (mid < bid OR mid > ask))
                ORDER BY asof_date DESC, ticker, expiry, strike
                LIMIT 50
                """
            )
        )

        if _table_exists(conn, "underlying_prices"):
            db["underlying_summary"] = {
                "rows": int(_scalar(conn, "SELECT COUNT(*) FROM underlying_prices") or 0),
                "tickers": int(_scalar(conn, "SELECT COUNT(DISTINCT ticker) FROM underlying_prices") or 0),
                "min_asof": _scalar(conn, "SELECT MIN(asof_date) FROM underlying_prices"),
                "max_asof": _scalar(conn, "SELECT MAX(asof_date) FROM underlying_prices"),
                "bad_close": int(_scalar(conn, "SELECT COUNT(*) FROM underlying_prices WHERE close <= 0") or 0),
            }

        if _table_exists(conn, "calc_cache"):
            cols = [r["name"] for r in db["schemas"].get("calc_cache", [])]
            if "key" in cols:
                db["calc_cache_summary"] = {
                    "rows": int(_scalar(conn, "SELECT COUNT(*) FROM calc_cache") or 0),
                    "sample_keys": _records(_df(conn, "SELECT key, created_at FROM calc_cache ORDER BY created_at DESC LIMIT 20")),
                }
    finally:
        conn.close()

    return db


def audit_files(root: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for path in sorted((root / "data").glob("**/*")):
        if path.suffix.lower() not in {".parquet", ".csv", ".db"}:
            continue
        info: dict[str, Any] = {
            "path": str(path.relative_to(root)),
            "size_bytes": path.stat().st_size,
            "suffix": path.suffix.lower(),
        }
        try:
            if path.suffix.lower() == ".parquet":
                df = pd.read_parquet(path)
                info.update(rows=len(df), columns=list(df.columns), null_cells=int(df.isna().sum().sum()))
            elif path.suffix.lower() == ".csv":
                df = pd.read_csv(path)
                info.update(rows=len(df), columns=list(df.columns), null_cells=int(df.isna().sum().sum()))
        except Exception as exc:
            info["read_error"] = str(exc)
        files.append(info)
    return {"files": files}


def classify_issues(raw: dict[str, Any], calc: dict[str, Any]) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    if raw.get("quick_check") != "ok":
        issues.append({"severity": "critical", "area": "sqlite", "message": f"iv_data.db quick_check={raw.get('quick_check')}"})

    total = raw.get("options_summary", {}).get("rows", 0) or 0
    nulls = raw.get("nulls", {})
    bad = raw.get("bad_counts", {})
    for key, val in bad.items():
        if val:
            sev = "high" if key in {"bad_iv", "bad_spot", "bad_strike", "bad_ttm", "crossed_market"} else "medium"
            issues.append({"severity": sev, "area": "options_quotes", "message": f"{key}: {_fmt_int(val)} rows ({_pct(float(val), float(total))})"})

    for key in ("iv_null", "spot_null", "ttm_null", "moneyness_null"):
        val = nulls.get(key, 0) or 0
        if total and val / total > 0.05:
            issues.append({"severity": "high", "area": "options_quotes", "message": f"{key}: {_fmt_int(val)} rows ({_pct(float(val), float(total))})"})

    for key in ("bid_null", "ask_null", "open_interest_null"):
        val = nulls.get(key, 0) or 0
        if total and val / total > 0.50:
            issues.append({"severity": "medium", "area": "market_fields", "message": f"{key}: {_fmt_int(val)} rows ({_pct(float(val), float(total))}); OI/liquidity-weighted paths may be unreliable"})

    gaps = raw.get("ticker_date_gaps", [])
    if gaps:
        issues.append({"severity": "medium", "area": "coverage", "message": f"{len(gaps)} ticker/date groups have sparse rows, missing ATM rows, or low IV coverage"})

    coverage = raw.get("ticker_coverage", [])
    if coverage:
        max_asof = raw.get("options_summary", {}).get("max_asof")
        one_date = [r for r in coverage if int(r.get("asof_dates") or 0) <= 1]
        stale = [r for r in coverage if max_asof and r.get("max_asof") != max_asof]
        low_atm = [r for r in coverage if float(r.get("atm_share") or 0.0) < 0.02]
        if one_date:
            issues.append({"severity": "high", "area": "coverage", "message": f"{len(one_date)} tickers have only one as-of date; time-series/spillover/correlation history is weak"})
        if stale:
            issues.append({"severity": "medium", "area": "coverage", "message": f"{len(stale)} tickers are stale versus latest as-of {max_asof}"})
        if low_atm:
            issues.append({"severity": "medium", "area": "atm_flags", "message": f"{len(low_atm)} tickers have <2% rows flagged ATM; do not rely on persisted is_atm alone"})

    if calc.get("exists") and calc.get("quick_check") != "ok":
        issues.append({"severity": "medium", "area": "cache", "message": f"calculations.db quick_check={calc.get('quick_check')}"})
    return issues


def render_markdown(payload: dict[str, Any]) -> str:
    raw = payload["raw_db"]
    calc = payload["calc_db"]
    files = payload["files"]["files"]
    issues = payload["issues"]
    lines: list[str] = []

    lines.append("# Data Quality Audit")
    lines.append("")
    lines.append(f"Generated: {payload['generated_at']}")
    lines.append(f"Project root: `{payload['root']}`")
    lines.append("")

    lines.append("## Storage Map")
    lines.append("")
    lines.append("- Raw options and underlying prices: `data/iv_data.db`")
    lines.append("- GUI/computation cache table: `data/iv_data.db::calc_cache`")
    lines.append("- Background warmup cache target: `data/calculations.db`")
    lines.append("- Model parameter history: `data/model_params.parquet`")
    lines.append("- External rate CSVs: `data/ML_Rates/*.csv`")
    lines.append("")

    lines.append("## Access Map")
    lines.append("")
    lines.append("| Data | Stored In | Main Accessors | Notes |")
    lines.append("|---|---|---|---|")
    lines.append("| Raw option chains | `data/iv_data.db::options_quotes` | `data.db_utils.get_conn`, `data.db_utils.fetch_quotes`, `analysis.analysis_pipeline.get_smile_slice` | Primary source for smiles, terms, surfaces, weights. `DB_PATH` can override the file. |")
    lines.append("| Underlying closes | `data/iv_data.db::underlying_prices` | `analysis.unified_weights.underlying_returns_matrix` | Used by `ul` feature modes and fallback weights. |")
    lines.append("| Ticker presets | `data/iv_data.db::ticker_groups` | `data.ticker_groups`, `display.gui.gui_input.InputPanel` | GUI universe presets. |")
    lines.append("| Interest rates | `data/iv_data.db::interest_rates`, `ticker_interest_rates` | `data.interest_rates`, GUI rate controls | Used during ingestion/Greek enrichment. |")
    lines.append("| GUI calc cache | `data/iv_data.db::calc_cache` | `analysis.compute_or_load.compute_or_load` | Pickled cached smiles/terms/corr/surface grids; no TTL in current implementation. |")
    lines.append("| Warmup cache | `data/calculations.db::calc_cache` | `analysis.cache_io.WarmupWorker` | Separate cache implementation; schema differs from `analysis.compute_or_load`. |")
    lines.append("| Model params | `data/model_params.parquet` | `analysis.model_params_logger` | Fit parameter history shown in parameter views. |")
    lines.append("")

    lines.append("## Issues")
    lines.append("")
    if issues:
        lines.append("| Severity | Area | Finding |")
        lines.append("|---|---|---|")
        for issue in issues:
            lines.append(f"| {issue['severity']} | {issue['area']} | {issue['message']} |")
    else:
        lines.append("No high-level issues detected by the current checks.")
    lines.append("")

    opt = raw.get("options_summary", {})
    lines.append("## Raw Options Summary")
    lines.append("")
    lines.append(f"- Rows: {_fmt_int(opt.get('rows', 0))}")
    lines.append(f"- Tickers: {_fmt_int(opt.get('tickers', 0))}")
    lines.append(f"- As-of dates: {_fmt_int(opt.get('asof_dates', 0))}")
    lines.append(f"- Date range: `{opt.get('min_asof')}` to `{opt.get('max_asof')}`")
    lines.append(f"- SQLite quick_check: `{raw.get('quick_check')}`")
    lines.append("")

    if raw.get("bad_counts"):
        lines.append("### Sanity Counts")
        lines.append("")
        lines.append("| Check | Rows |")
        lines.append("|---|---:|")
        for key, val in raw["bad_counts"].items():
            lines.append(f"| {key} | {_fmt_int(val)} |")
        lines.append("")

    if raw.get("nulls"):
        lines.append("### Null Counts")
        lines.append("")
        lines.append("| Column | Null rows |")
        lines.append("|---|---:|")
        for key, val in raw["nulls"].items():
            lines.append(f"| {key.replace('_null', '')} | {_fmt_int(val)} |")
        lines.append("")

    cov = raw.get("ticker_coverage", [])
    if cov:
        lines.append("### Ticker Coverage")
        lines.append("")
        lines.append("| Ticker | Rows | Dates | Date Range | Expiries | IV Rows | ATM Share |")
        lines.append("|---|---:|---:|---|---:|---:|---:|")
        for row in cov:
            lines.append(
                f"| {row['ticker']} | {_fmt_int(row['rows'])} | {_fmt_int(row['asof_dates'])} | "
                f"{row['min_asof']} to {row['max_asof']} | {_fmt_int(row['expiries'])} | "
                f"{_fmt_int(row['iv_rows'])} | {row.get('atm_share')} |"
            )
        lines.append("")

    gaps = raw.get("ticker_date_gaps", [])
    if gaps:
        lines.append("### Sparse Or Suspect Ticker-Date Groups")
        lines.append("")
        lines.append("| Ticker | Date | Rows | Expiries | IV Rows | ATM Rows | Avg IV |")
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        for row in gaps[:50]:
            lines.append(
                f"| {row['ticker']} | {row['asof_date']} | {_fmt_int(row['rows'])} | "
                f"{_fmt_int(row['expiries'])} | {_fmt_int(row['iv_rows'])} | "
                f"{_fmt_int(row['atm_rows'])} | {row.get('avg_iv')} |"
            )
        lines.append("")

    if raw.get("sample_bad_rows"):
        lines.append("### Sample Bad Rows")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(raw["sample_bad_rows"], indent=2))
        lines.append("```")
        lines.append("")

    lines.append("## Cache Summary")
    lines.append("")
    lines.append(f"- `data/calculations.db` exists: `{calc.get('exists')}`, quick_check: `{calc.get('quick_check')}`")
    if raw.get("calc_cache_summary"):
        lines.append(f"- `iv_data.db::calc_cache` rows: {_fmt_int(raw['calc_cache_summary'].get('rows'))}")
    if calc.get("table_counts"):
        for table, count in calc["table_counts"].items():
            lines.append(f"- `calculations.db::{table}` rows: {_fmt_int(count)}")
    lines.append("")

    lines.append("## Data Files")
    lines.append("")
    lines.append("| Path | Type | Size | Rows | Null cells |")
    lines.append("|---|---|---:|---:|---:|")
    for info in files:
        lines.append(
            f"| `{info['path']}` | {info['suffix']} | {_fmt_int(info['size_bytes'])} | "
            f"{_fmt_int(info.get('rows', ''))} | {_fmt_int(info.get('null_cells', ''))} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit LivePeerIVApp data quality and storage.")
    parser.add_argument("--raw-db", type=Path, default=Path(os.getenv("DB_PATH", DEFAULT_RAW_DB)))
    parser.add_argument("--calc-db", type=Path, default=DEFAULT_CALC_DB)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    args = parser.parse_args()

    raw = audit_options_db(args.raw_db)
    calc = audit_sqlite(args.calc_db)
    files = audit_files(ROOT)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(ROOT),
        "raw_db": raw,
        "calc_db": calc,
        "files": files,
        "issues": classify_issues(raw, calc),
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    args.report.write_text(render_markdown(payload), encoding="utf-8")

    print(f"Wrote {args.report}")
    print(f"Wrote {args.json}")
    print(f"Issues: {len(payload['issues'])}")
    for issue in payload["issues"][:10]:
        print(f"- {issue['severity']} {issue['area']}: {issue['message']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
