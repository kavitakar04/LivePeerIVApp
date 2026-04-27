from __future__ import annotations

import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional

STORE_PATH = Path(__file__).resolve().parents[1] / "data" / "model_params.parquet"
PARAM_COLUMNS = ["asof_date", "ticker", "expiry", "tenor_d", "model", "param", "value", "fit_meta"]


def _empty_params_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=PARAM_COLUMNS)


def _encode_fit_meta(meta: Dict[str, Any]) -> str:
    return json.dumps(meta or {}, sort_keys=True, default=str)


def _decode_fit_meta(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value:
        try:
            parsed = json.loads(value)
        except Exception:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _quarantine_corrupt_store(path: Path = STORE_PATH) -> None:
    """Move an unreadable parameter log aside so future writes can recreate it."""
    if not path.exists():
        return
    try:
        backup = path.with_suffix(path.suffix + ".corrupt")
        idx = 1
        while backup.exists():
            backup = path.with_suffix(path.suffix + f".corrupt.{idx}")
            idx += 1
        path.replace(backup)
    except Exception:
        pass


def append_params(
    asof_date: str,
    ticker: str,
    expiry: Optional[str],
    model: str,
    params: Dict[str, float],
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    """Log fitted parameters to disk."""
    meta = meta or {}
    asof_ts = pd.to_datetime(asof_date).normalize()
    expiry_dt = pd.to_datetime(expiry) if expiry else None
    tenor_d = (expiry_dt - asof_ts).days if expiry_dt is not None else None
    rows = []
    for key, val in params.items():
        try:
            fval = float(val)
        except Exception:
            continue
        rows.append(
            {
                "asof_date": asof_ts,
                "ticker": ticker,
                "expiry": expiry_dt,
                "tenor_d": tenor_d,
                "model": model.lower(),
                "param": key,
                "value": fval,
                "fit_meta": _encode_fit_meta(meta),
            }
        )
    if not rows:
        return
    df_new = pd.DataFrame(rows)
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if STORE_PATH.exists():
        try:
            df_old = pd.read_parquet(STORE_PATH)
        except Exception:
            _quarantine_corrupt_store(STORE_PATH)
            df_old = _empty_params_frame()
        df = df_new if df_old.empty else pd.concat([df_old, df_new], ignore_index=True)
        df = df.sort_values(["asof_date", "ticker", "expiry", "model", "param"]).drop_duplicates(
            ["asof_date", "ticker", "expiry", "model", "param"], keep="last"
        )
    else:
        df = df_new
    df.to_parquet(STORE_PATH, index=False)


def load_model_params() -> pd.DataFrame:
    """Load the logged parameters as a DataFrame."""
    if not STORE_PATH.exists():
        return _empty_params_frame()
    try:
        df = pd.read_parquet(STORE_PATH)
    except Exception:
        _quarantine_corrupt_store(STORE_PATH)
        return _empty_params_frame()
    df["asof_date"] = pd.to_datetime(df["asof_date"])
    if "expiry" in df:
        df["expiry"] = pd.to_datetime(df["expiry"], errors="coerce")
    if "fit_meta" in df:
        df["fit_meta"] = df["fit_meta"].map(_decode_fit_meta)
    return df
