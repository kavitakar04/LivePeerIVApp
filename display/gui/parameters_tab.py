from __future__ import annotations

import math
import tkinter as tk
from tkinter import ttk
from typing import Dict, Any, Optional


def _fmt(val, fmt=".4f", fallback="—") -> str:
    try:
        value = float(val)
        return format(value, fmt) if math.isfinite(value) else fallback
    except (TypeError, ValueError):
        return fallback


def _pct(val, fallback="—") -> str:
    try:
        value = float(val)
        return f"{value * 100:.2f}%" if math.isfinite(value) else fallback
    except (TypeError, ValueError):
        return fallback


def _date_part(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    if "T" in text:
        return text.split("T")[0]
    if " " in text:
        return text.split(" ")[0]
    return text


def _quality_value(quality: dict[str, Any], key: str, fallback: Any = None) -> Any:
    if not isinstance(quality, dict):
        return fallback
    return quality.get(key, fallback)


def _params_summary(params: dict[str, Any]) -> str:
    if not isinstance(params, dict) or not params:
        return ""
    excluded = {"rmse", "n", "atm_vol", "skew", "curv", "quality_ok", "quality_reason"}
    parts: list[str] = []
    for key in sorted(params):
        if key in excluded:
            continue
        val = params.get(key)
        try:
            parts.append(f"{key}={float(val):.4g}")
        except (TypeError, ValueError):
            if val is not None:
                parts.append(f"{key}={val}")
    return ", ".join(parts)


def _weight_rows(info: Dict[str, Any]) -> list[dict[str, Any]]:
    weight_info = info.get("weight_info")
    if not isinstance(weight_info, dict):
        return []
    weights = weight_info.get("weights") or {}
    if not isinstance(weights, dict) or not weights:
        return []

    target = weight_info.get("target") or info.get("ticker", "")
    asof = weight_info.get("asof") or info.get("asof", "")
    mode = weight_info.get("mode") or ""
    warning = weight_info.get("warning") or ""
    rows: list[dict[str, Any]] = []
    for peer, weight in sorted(weights.items(), key=lambda item: abs(float(item[1])), reverse=True):
        rows.append({
            "Ticker": str(peer),
            "As Of": asof,
            "DTE": "",
            "Expiry": "",
            "Model": "weight",
            "Status": mode,
            "Fallback": "equal" if "using equal weights" in warning else "none",
            "ATM Vol": None,
            "Skew": None,
            "Curvature": None,
            "RMSE": None,
            "N": None,
            "Min IV": None,
            "Max IV": None,
            "Reason": warning,
            "Params": f"{target} peer weight={float(weight):.2%}",
        })
    return rows


def _status_rows(info: Dict[str, Any]) -> list[dict[str, Any]]:
    events = info.get("status_events") or []
    if not isinstance(events, list):
        return []
    ticker = info.get("ticker", "")
    asof = info.get("asof", "")
    rows: list[dict[str, Any]] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        rows.append({
            "Ticker": event.get("ticker", ticker),
            "As Of": event.get("asof", asof),
            "DTE": event.get("dte", ""),
            "Expiry": event.get("expiry", ""),
            "Model": event.get("category", "status"),
            "Status": event.get("status", "info"),
            "Fallback": event.get("fallback", "none"),
            "ATM Vol": None,
            "Skew": None,
            "Curvature": None,
            "RMSE": event.get("rmse"),
            "N": event.get("n"),
            "Min IV": None,
            "Max IV": None,
            "Reason": event.get("message", ""),
            "Params": event.get("detail", ""),
        })
    return rows


def flatten_diagnostics_info(info: Dict[str, Any] | None) -> list[dict[str, Any]]:
    """Return diagnostics rows from PlotManager.last_fit_info."""
    if not isinstance(info, dict):
        return []

    ticker = info.get("ticker", "")
    asof = info.get("asof", "")
    fit_map = info.get("fit_by_expiry")

    rows: list[dict[str, Any]] = _status_rows(info) + _weight_rows(info)
    if not isinstance(fit_map, dict):
        return rows

    for T_val, entry in sorted(fit_map.items(), key=lambda kv: float(kv[0])):
        if not isinstance(entry, dict):
            continue
        try:
            dte = int(round(float(T_val) * 365.25))
        except (TypeError, ValueError):
            dte = ""

        sens = entry.get("sens") or {}
        quality_map = entry.get("quality") or {}
        fallback_map = entry.get("fallback") or {}
        expiry = _date_part(entry.get("expiry"))

        for model_key in ("svi", "sabr", "tps"):
            params = entry.get(model_key) or {}
            quality = quality_map.get(model_key) or {}
            has_params = bool(params)
            quality_ok = _quality_value(quality, "ok")

            if quality_ok is True or has_params:
                status = "ok"
            elif quality_ok is False:
                status = "rejected"
            else:
                status = "not_run"

            rows.append({
                "Ticker": ticker,
                "As Of": asof,
                "DTE": dte,
                "Expiry": expiry,
                "Model": model_key,
                "Status": status,
                "Fallback": fallback_map.get(model_key, "none") if isinstance(fallback_map, dict) else "none",
                "ATM Vol": sens.get("atm_vol"),
                "Skew": sens.get("skew"),
                "Curvature": sens.get("curv"),
                "RMSE": params.get("rmse", _quality_value(quality, "rmse")),
                "N": params.get("n", _quality_value(quality, "n")),
                "Min IV": _quality_value(quality, "min_iv"),
                "Max IV": _quality_value(quality, "max_iv"),
                "Reason": _quality_value(quality, "reason", ""),
                "Params": _params_summary(params),
            })

    return rows


def flatten_fit_info(info: Dict[str, Any] | None) -> list[dict[str, Any]]:
    """Return user-facing expiry-level parameter rows."""
    return flatten_summary_info(info)


def _expiry_quality(entry: dict[str, Any]) -> str:
    quality_map = entry.get("quality") or {}
    if not isinstance(quality_map, dict) or not quality_map:
        return "Available"
    values = [q for q in quality_map.values() if isinstance(q, dict)]
    if any(q.get("ok") is True for q in values):
        return "Good"
    if any(q.get("ok") is False for q in values):
        return "Check"
    return "Available"


def flatten_summary_info(info: Dict[str, Any] | None) -> list[dict[str, Any]]:
    """Return user-facing expiry-level parameter rows."""
    if not isinstance(info, dict):
        return []

    ticker = info.get("ticker", "")
    asof = info.get("asof", "")
    fit_map = info.get("fit_by_expiry")
    if not isinstance(fit_map, dict):
        return []

    rows: list[dict[str, Any]] = []
    for T_val, entry in sorted(fit_map.items(), key=lambda kv: float(kv[0])):
        if not isinstance(entry, dict):
            continue
        try:
            dte = int(round(float(T_val) * 365.25))
        except (TypeError, ValueError):
            dte = ""
        sens = entry.get("sens") or {}
        rows.append({
            "Ticker": ticker,
            "As Of": asof,
            "DTE": dte,
            "Expiry": _date_part(entry.get("expiry")),
            "ATM Vol": sens.get("atm_vol"),
            "Skew": sens.get("skew"),
            "Curvature": sens.get("curv"),
            "Fit Quality": _expiry_quality(entry),
        })
    return rows


class ParametersTab(ttk.Frame):
    """User-facing expiry-level fitted surface summary."""

    _COLS = (
        "Ticker", "As Of", "DTE", "Expiry", "ATM Vol", "Skew", "Curvature", "Fit Quality",
    )
    _WIDTHS = (70, 90, 50, 95, 85, 85, 90, 95)

    def __init__(self, master):
        super().__init__(master)

        self.lbl_meta = ttk.Label(self, text="", anchor="w", foreground="gray")
        self.lbl_meta.pack(fill=tk.X, padx=6, pady=(6, 2))

        # ---- table ----
        tbl_frame = ttk.Frame(self)
        tbl_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))

        self.tree = ttk.Treeview(tbl_frame, columns=self._COLS,
                                  show="headings", height=20, selectmode="browse")
        vsb = ttk.Scrollbar(tbl_frame, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(tbl_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        for col, w in zip(self._COLS, self._WIDTHS):
            anchor = tk.W if col in self._left_anchor_columns() else tk.E
            self.tree.heading(col, text=col,
                              command=lambda c=col: self._sort_by(c))
            self.tree.column(col, anchor=anchor, width=w, stretch=(col in self._stretch_columns()))

        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        tbl_frame.rowconfigure(0, weight=1)
        tbl_frame.columnconfigure(0, weight=1)

        self._sort_col: Optional[str] = None
        self._sort_asc: bool = True
        self._rows: list = []

    def _left_anchor_columns(self) -> set[str]:
        return {"Ticker", "As Of", "Expiry", "Fit Quality", "Model", "Status", "Fallback", "Reason", "Params"}

    def _stretch_columns(self) -> set[str]:
        return {"Reason", "Params"}

    def update(self, info: Dict[str, Any] | None) -> None:
        self._clear()

        if not info:
            self.lbl_meta.config(text="No fit data")
            return

        ticker = info.get("ticker", "")
        asof = info.get("asof", "")
        self.lbl_meta.config(text=f"{ticker}   {asof}" if ticker or asof else "")

        rows = flatten_summary_info(info)
        self._rows = rows
        self._insert_rows(rows)

    # ---- internals ----

    def _clear(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        self._rows = []

    def _insert_rows(self, rows):
        for r in rows:
            self.tree.insert("", tk.END, values=(
                r["Ticker"],
                r["As Of"],
                r["DTE"],
                r["Expiry"],
                _pct(r["ATM Vol"]),
                _pct(r["Skew"]),
                _fmt(r["Curvature"], ".4f"),
                r["Fit Quality"],
            ))

    def _sort_by(self, col: str):
        if self._sort_col == col:
            self._sort_asc = not self._sort_asc
        else:
            self._sort_col = col
            self._sort_asc = True

        key_map = {
            "Ticker": lambda r: r["Ticker"] or "",
            "As Of": lambda r: r["As Of"] or "",
            "DTE": lambda r: int(r["DTE"]) if r["DTE"] != "" else -1,
            "Expiry": lambda r: r["Expiry"] or "",
            "ATM Vol": lambda r: float(r["ATM Vol"]) if r["ATM Vol"] is not None else -999,
            "Skew": lambda r: float(r["Skew"]) if r["Skew"] is not None else -999,
            "Curvature": lambda r: float(r["Curvature"]) if r["Curvature"] is not None else -999,
            "Fit Quality": lambda r: r["Fit Quality"] or "",
        }
        key = key_map.get(col, lambda r: 0)
        sorted_rows = sorted(self._rows, key=key, reverse=not self._sort_asc)
        self._clear_tree()
        self._insert_rows(sorted_rows)

    def _clear_tree(self):
        for item in self.tree.get_children():
            self.tree.delete(item)


class DiagnosticsTab(ParametersTab):
    """Settings/Status diagnostics table for model and data-integrity details."""

    _COLS = (
        "Ticker", "As Of", "DTE", "Expiry", "Model", "Status", "Fallback",
        "ATM Vol", "Skew", "Curvature", "RMSE", "N", "Min IV", "Max IV",
        "Reason", "Params",
    )
    _WIDTHS = (70, 90, 50, 95, 55, 75, 75, 80, 80, 85, 75, 50, 70, 70, 220, 260)

    def _insert_rows(self, rows):
        for r in rows:
            self.tree.insert("", tk.END, values=(
                r["Ticker"],
                r["As Of"],
                r["DTE"],
                r["Expiry"],
                r["Model"],
                r["Status"],
                r["Fallback"],
                _pct(r["ATM Vol"]),
                _pct(r["Skew"]),
                _fmt(r["Curvature"], ".4f"),
                _fmt(r["RMSE"], ".5f"),
                str(r["N"]) if r["N"] is not None else "—",
                _fmt(r["Min IV"], ".4f"),
                _fmt(r["Max IV"], ".4f"),
                r["Reason"] or "",
                r["Params"] or "",
            ))

    def update(self, info: Dict[str, Any] | None) -> None:
        self._clear()

        if not info:
            self.lbl_meta.config(text="No diagnostics")
            return

        ticker = info.get("ticker", "")
        asof = info.get("asof", "")
        self.lbl_meta.config(text=f"{ticker}   {asof}" if ticker or asof else "")

        rows = flatten_diagnostics_info(info)
        self._rows = rows
        self._insert_rows(rows)

    def _sort_by(self, col: str):
        if self._sort_col == col:
            self._sort_asc = not self._sort_asc
        else:
            self._sort_col = col
            self._sort_asc = True

        key_map = {
            "Ticker": lambda r: r["Ticker"] or "",
            "As Of": lambda r: r["As Of"] or "",
            "DTE": lambda r: int(r["DTE"]) if r["DTE"] != "" else -1,
            "Expiry": lambda r: r["Expiry"] or "",
            "Model": lambda r: r["Model"] or "",
            "Status": lambda r: r["Status"] or "",
            "Fallback": lambda r: r["Fallback"] or "",
            "ATM Vol": lambda r: float(r["ATM Vol"]) if r["ATM Vol"] is not None else -999,
            "Skew": lambda r: float(r["Skew"]) if r["Skew"] is not None else -999,
            "Curvature": lambda r: float(r["Curvature"]) if r["Curvature"] is not None else -999,
            "RMSE": lambda r: float(r["RMSE"]) if r["RMSE"] is not None else 999,
            "N": lambda r: int(r["N"]) if r["N"] is not None else -1,
            "Min IV": lambda r: float(r["Min IV"]) if r["Min IV"] is not None else -999,
            "Max IV": lambda r: float(r["Max IV"]) if r["Max IV"] is not None else -999,
            "Reason": lambda r: r["Reason"] or "",
            "Params": lambda r: r["Params"] or "",
        }
        key = key_map.get(col, lambda r: 0)
        sorted_rows = sorted(self._rows, key=key, reverse=not self._sort_asc)
        self._clear_tree()
        self._insert_rows(sorted_rows)
