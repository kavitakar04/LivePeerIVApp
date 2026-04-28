#!/usr/bin/env python3
"""Recover readable model-parameter snapshots from quarantined parquet files."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import analysis.persistence.model_params_logger as model_params_logger

KEY_COLUMNS = ["asof_date", "ticker", "expiry", "model", "param"]


@dataclass(frozen=True)
class RecoverySummary:
    store: Path
    archive_dir: Path | None
    current_readable: bool
    quarantine_files: int
    readable_quarantine_files: int
    unreadable_quarantine_files: int
    recovered_rows: int
    output_rows: int
    dry_run: bool


def _fit_meta_to_json(value: Any) -> str:
    if value is None:
        return "{}"
    try:
        if pd.isna(value):
            return "{}"
    except (TypeError, ValueError):
        pass
    if isinstance(value, float) and pd.isna(value):
        return "{}"
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, default=str)
    if isinstance(value, str):
        return value if value else "{}"
    return json.dumps(value, sort_keys=True, default=str)


def _normalize_frame(df: pd.DataFrame, source_mtime: float, source_order: int) -> pd.DataFrame:
    work = df.copy()
    for col in model_params_logger.PARAM_COLUMNS:
        if col not in work:
            work[col] = pd.NA
    work = work[model_params_logger.PARAM_COLUMNS].copy()
    work["asof_date"] = pd.to_datetime(work["asof_date"], errors="coerce").dt.normalize()
    work["expiry"] = pd.to_datetime(work["expiry"], errors="coerce")
    work["tenor_d"] = pd.to_numeric(work["tenor_d"], errors="coerce")
    work["value"] = pd.to_numeric(work["value"], errors="coerce")
    work["fit_meta"] = work["fit_meta"].map(_fit_meta_to_json)
    for col in ("ticker", "model", "param"):
        work[col] = work[col].where(work[col].notna(), pd.NA)
        work[col] = work[col].astype("string").str.strip()
    work["ticker"] = work["ticker"].str.upper()
    work["model"] = work["model"].str.lower()
    work = work.dropna(subset=["asof_date", "ticker", "model", "param", "value"])
    work = work[(work["ticker"] != "") & (work["model"] != "") & (work["param"] != "")]
    work = work[np.isfinite(work["value"].to_numpy(dtype=float))]
    work["_source_mtime"] = float(source_mtime)
    work["_source_order"] = int(source_order)
    return work


def _unique_archive_dir(root: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = root / stamp
    idx = 1
    while archive_dir.exists():
        archive_dir = root / f"{stamp}_{idx}"
        idx += 1
    return archive_dir


def recover_model_params(
    store: Path = model_params_logger.STORE_PATH,
    archive_root: Path | None = None,
    archive: bool = True,
    dry_run: bool = False,
) -> RecoverySummary:
    """Merge readable quarantine files into the live store and archive old files."""
    store = Path(store)
    archive_root = archive_root or store.parent / "cache" / "model_params_corrupt_archive"
    quarantine_files = sorted(
        store.parent.glob(store.name + ".corrupt*"),
        key=lambda path: (path.stat().st_mtime, path.name),
    )
    archive_dir = _unique_archive_dir(archive_root) if archive and quarantine_files else None

    frames: list[pd.DataFrame] = []
    current_readable = False
    readable_quarantine_files = 0
    unreadable_quarantine_files = 0

    with model_params_logger._store_lock(store):
        if store.exists():
            try:
                frames.append(_normalize_frame(pd.read_parquet(store), store.stat().st_mtime, 0))
                current_readable = True
            except Exception:
                unreadable_quarantine_files += 1

        for idx, path in enumerate(quarantine_files, start=1):
            try:
                frame = pd.read_parquet(path)
            except Exception:
                unreadable_quarantine_files += 1
                continue
            readable_quarantine_files += 1
            frames.append(_normalize_frame(frame, path.stat().st_mtime, idx))

        if not frames:
            raise RuntimeError(f"No readable model parameter snapshots found for {store}")

        merged = pd.concat(frames, ignore_index=True)
        recovered_rows = int(len(merged))
        merged = (
            merged.sort_values(KEY_COLUMNS + ["_source_mtime", "_source_order"])
            .drop_duplicates(KEY_COLUMNS, keep="last")
            .drop(columns=["_source_mtime", "_source_order"])
            .sort_values(["asof_date", "ticker", "expiry", "model", "param"])
            .reset_index(drop=True)
        )

        if not dry_run:
            if archive_dir is not None:
                archive_dir.mkdir(parents=True, exist_ok=True)
                if store.exists():
                    shutil.copy2(store, archive_dir / f"{store.name}.before_recovery")
            model_params_logger._write_parquet_atomic(merged, store)
            if archive_dir is not None:
                for path in quarantine_files:
                    shutil.move(str(path), archive_dir / path.name)

    return RecoverySummary(
        store=store,
        archive_dir=archive_dir,
        current_readable=current_readable,
        quarantine_files=len(quarantine_files),
        readable_quarantine_files=readable_quarantine_files,
        unreadable_quarantine_files=unreadable_quarantine_files,
        recovered_rows=recovered_rows,
        output_rows=int(len(merged)),
        dry_run=dry_run,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=model_params_logger.STORE_PATH)
    parser.add_argument("--archive-root", type=Path, default=None)
    parser.add_argument("--no-archive", action="store_true", help="Leave quarantine files in place")
    parser.add_argument("--dry-run", action="store_true", help="Read and report without writing")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    summary = recover_model_params(
        store=args.store,
        archive_root=args.archive_root,
        archive=not args.no_archive,
        dry_run=args.dry_run,
    )
    print(f"store={summary.store}")
    print(f"quarantine_files={summary.quarantine_files}")
    print(f"readable_quarantine_files={summary.readable_quarantine_files}")
    print(f"unreadable_quarantine_files={summary.unreadable_quarantine_files}")
    print(f"recovered_rows_before_dedupe={summary.recovered_rows}")
    print(f"output_rows={summary.output_rows}")
    print(f"archive_dir={summary.archive_dir}")
    print(f"dry_run={summary.dry_run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
