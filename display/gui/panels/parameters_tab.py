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


def _model_display_name(model: Any) -> str:
    labels = {
        "svi": "SVI",
        "sabr": "SABR",
        "tps": "TPS",
        "poly": "Polynomial",
        "poly2": "Polynomial",
        "sens": "Sensitivity",
        "weight": "Weights",
    }
    text = str(model or "")
    return labels.get(text.lower(), text)


MODEL_HEALTH_MODELS = ("svi", "sabr", "tps")
MODEL_RMSE_DEGRADED_THRESHOLD = 0.05


def _model_health_status(params: dict[str, Any], quality: dict[str, Any]) -> tuple[str, str]:
    """Return ok/degraded/failed using the same explicit RMSE warning threshold as plots."""
    ok = _quality_value(quality, "ok")
    rmse = _quality_value(quality, "rmse", params.get("rmse") if isinstance(params, dict) else None)
    reason = str(_quality_value(quality, "reason", "") or "")
    try:
        rmse_f = float(rmse)
    except (TypeError, ValueError):
        rmse_f = math.nan

    if ok is False:
        if np_isfinite(rmse_f) and rmse_f > MODEL_RMSE_DEGRADED_THRESHOLD:
            return "degraded", reason or f"RMSE {rmse_f:.4f} exceeds {MODEL_RMSE_DEGRADED_THRESHOLD:.2f}"
        if "rmse" in reason.lower() or "too high" in reason.lower():
            return "degraded", reason
        return "failed", reason or "fit failed quality gate"
    if isinstance(params, dict) and params:
        if np_isfinite(rmse_f) and rmse_f > MODEL_RMSE_DEGRADED_THRESHOLD:
            return "degraded", f"RMSE {rmse_f:.4f} exceeds {MODEL_RMSE_DEGRADED_THRESHOLD:.2f}"
        return "ok", f"RMSE {_fmt(rmse_f, '.4f')}"
    return "failed", reason or "insufficient data or fit not run"


def np_isfinite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _status_symbol(status: str) -> str:
    if status == "ok":
        return "✅ ok"
    if status == "degraded":
        return "⚠ degraded"
    if status == "failed":
        return "❌ failed"
    return "—"


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
        rows.append(
            {
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
            }
        )
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
        rows.append(
            {
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
            }
        )
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

            if quality_ok is False:
                status = "rejected"
            elif quality_ok is True or has_params:
                status = "ok"
            else:
                status = "not_run"

            rows.append(
                {
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
                }
            )

    return rows


def flatten_fit_info(info: Dict[str, Any] | None) -> list[dict[str, Any]]:
    """Return user-facing expiry-level parameter rows."""
    return flatten_summary_info(info)


def build_model_health_grid(info: Dict[str, Any] | None) -> dict[str, Any]:
    """Build model x expiry health grid from the same fit metadata used downstream."""
    if not isinstance(info, dict) or not isinstance(info.get("fit_by_expiry"), dict):
        return {
            "expiries": [],
            "rows": [],
            "primary_model": "Unavailable",
            "rejected_models": [],
            "partially_reliable_models": [],
            "thresholds": {"rmse_degraded": MODEL_RMSE_DEGRADED_THRESHOLD},
        }

    fit_map = info.get("fit_by_expiry") or {}
    expiries: list[dict[str, Any]] = []
    status_by_model: dict[str, dict[str, str]] = {m: {} for m in MODEL_HEALTH_MODELS}
    reasons_by_model: dict[str, dict[str, str]] = {m: {} for m in MODEL_HEALTH_MODELS}
    expiry_lookup: dict[str, dict[str, Any]] = {}

    for T_val, entry in sorted(fit_map.items(), key=lambda kv: float(kv[0])):
        if not isinstance(entry, dict):
            continue
        try:
            dte = int(round(float(T_val) * 365.25))
        except (TypeError, ValueError):
            dte = len(expiries) + 1
        label = f"{dte}d"
        expiry = _date_part(entry.get("expiry"))
        expiries.append({"label": label, "dte": dte, "expiry": expiry})
        expiry_lookup[label] = {"DTE": dte, "Expiry": expiry}
        quality_map = entry.get("quality") or {}
        if not isinstance(quality_map, dict):
            quality_map = {}
        for model in MODEL_HEALTH_MODELS:
            params = entry.get(model) or {}
            quality = quality_map.get(model) or {}
            status, reason = _model_health_status(params, quality)
            status_by_model[model][label] = status
            reasons_by_model[model][label] = reason

    rows: list[dict[str, Any]] = []
    model_counts: dict[str, dict[str, int]] = {}
    for model in MODEL_HEALTH_MODELS:
        counts = {
            "ok": list(status_by_model[model].values()).count("ok"),
            "degraded": list(status_by_model[model].values()).count("degraded"),
            "failed": list(status_by_model[model].values()).count("failed"),
        }
        model_counts[model] = counts
        row = {
            "Model": _model_display_name(model),
            "_model": model,
            "_statuses": dict(status_by_model[model]),
            "_reasons": dict(reasons_by_model[model]),
        }
        for exp in expiries:
            label = exp["label"]
            row[label] = _status_symbol(status_by_model[model].get(label, "missing"))
        rows.append(row)

    if expiries:
        primary = max(
            MODEL_HEALTH_MODELS,
            key=lambda m: (model_counts[m]["ok"], -model_counts[m]["failed"], -model_counts[m]["degraded"]),
        )
        primary_model = _model_display_name(primary) if model_counts[primary]["ok"] else "Unavailable"
    else:
        primary_model = "Unavailable"

    rejected = [
        _model_display_name(m)
        for m in MODEL_HEALTH_MODELS
        if model_counts.get(m, {}).get("failed", 0) == len(expiries) and expiries
    ]
    partial = [
        _model_display_name(m)
        for m in MODEL_HEALTH_MODELS
        if expiries
        and model_counts.get(m, {}).get("ok", 0) > 0
        and (model_counts.get(m, {}).get("failed", 0) > 0 or model_counts.get(m, {}).get("degraded", 0) > 0)
    ]

    return {
        "expiries": expiries,
        "rows": rows,
        "primary_model": primary_model,
        "rejected_models": rejected,
        "partially_reliable_models": partial,
        "thresholds": {"rmse_degraded": MODEL_RMSE_DEGRADED_THRESHOLD},
        "expiry_lookup": expiry_lookup,
    }


def summarize_health_info(info: Dict[str, Any] | None) -> dict[str, Any]:
    """Summarize diagnostics into a first-screen system health view."""
    rows = flatten_diagnostics_info(info)
    grid = build_model_health_grid(info)
    if not rows:
        return {
            "overall": "No health data",
            "model_quality": "Unknown",
            "warnings": 0,
            "failures": 0,
            "fallbacks": 0,
            "reliable_models": [],
            "primary_model": grid["primary_model"],
            "rejected_models": grid["rejected_models"],
            "partially_reliable_models": grid["partially_reliable_models"],
            "thresholds": grid["thresholds"],
            "messages": ["Plot an IV view to populate model and data health."],
        }

    warnings = [
        r
        for r in rows
        if str(r.get("Status", "")).lower() in {"warning", "rejected", "failed", "error"} or bool(r.get("Reason"))
    ]
    failures = [r for r in rows if str(r.get("Status", "")).lower() in {"rejected", "failed", "error"}]
    fallbacks = [r for r in rows if str(r.get("Fallback", "none")).lower() not in {"", "none"}]
    model_rows = [r for r in rows if str(r.get("Model", "")).lower() in {"svi", "sabr", "tps", "poly", "poly2"}]
    reliable = sorted(
        {
            _model_display_name(r.get("Model"))
            for r in model_rows
            if str(r.get("Status", "")).lower() == "ok" and str(r.get("Fallback", "none")).lower() in {"", "none"}
        }
    )

    if failures:
        model_quality = "Degraded"
    elif warnings or fallbacks:
        model_quality = "Check"
    elif reliable:
        model_quality = "Good"
    else:
        model_quality = "Unknown"

    messages: list[str] = []
    if reliable:
        messages.append("Reliable models: " + ", ".join(reliable))
    if failures:
        messages.append(f"{len(failures)} model/data failure(s) need review.")
    if warnings and not failures:
        messages.append(f"{len(warnings)} warning(s) may affect downstream confidence.")
    if fallbacks:
        messages.append(f"{len(fallbacks)} fallback path(s) were used.")
    if not messages:
        messages.append("No model/data warnings detected in the latest plotted view.")

    return {
        "overall": model_quality,
        "model_quality": model_quality,
        "warnings": len(warnings),
        "failures": len(failures),
        "fallbacks": len(fallbacks),
        "reliable_models": reliable,
        "primary_model": grid["primary_model"],
        "rejected_models": grid["rejected_models"],
        "partially_reliable_models": grid["partially_reliable_models"],
        "thresholds": grid["thresholds"],
        "messages": messages,
    }


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
        rows.append(
            {
                "Ticker": ticker,
                "As Of": asof,
                "DTE": dte,
                "Expiry": _date_part(entry.get("expiry")),
                "ATM Vol": sens.get("atm_vol"),
                "Skew": sens.get("skew"),
                "Curvature": sens.get("curv"),
                "Fit Quality": _expiry_quality(entry),
            }
        )
    return rows


class ParametersTab(ttk.Frame):
    """User-facing expiry-level fitted surface summary."""

    _COLS = (
        "Ticker",
        "As Of",
        "DTE",
        "Expiry",
        "ATM Vol",
        "Skew",
        "Curvature",
        "Fit Quality",
    )
    _WIDTHS = (70, 90, 50, 95, 85, 85, 90, 95)

    def __init__(self, master):
        super().__init__(master)

        self.lbl_meta = ttk.Label(self, text="", anchor="w", foreground="gray")
        self.lbl_meta.pack(fill=tk.X, padx=6, pady=(6, 2))

        # ---- table ----
        tbl_frame = ttk.Frame(self)
        tbl_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))

        self.tree = ttk.Treeview(tbl_frame, columns=self._COLS, show="headings", height=20, selectmode="browse")
        vsb = ttk.Scrollbar(tbl_frame, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(tbl_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        for col, w in zip(self._COLS, self._WIDTHS):
            anchor = tk.W if col in self._left_anchor_columns() else tk.E
            self.tree.heading(col, text=col, command=lambda c=col: self._sort_by(c))
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
            self.tree.insert(
                "",
                tk.END,
                values=(
                    r["Ticker"],
                    r["As Of"],
                    r["DTE"],
                    r["Expiry"],
                    _pct(r["ATM Vol"]),
                    _pct(r["Skew"]),
                    _fmt(r["Curvature"], ".4f"),
                    r["Fit Quality"],
                ),
            )

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
        self._rows = sorted_rows
        self._clear_tree()
        self._insert_rows(sorted_rows)

    def _clear_tree(self):
        for item in self.tree.get_children():
            self.tree.delete(item)


class DiagnosticsTab(ParametersTab):
    """Runtime diagnostics table for model, weight, and data-integrity details."""

    _COLS = (
        "Ticker",
        "As Of",
        "DTE",
        "Expiry",
        "Model",
        "Status",
        "Fallback",
        "ATM Vol",
        "Skew",
        "Curvature",
        "RMSE",
        "N",
        "Min IV",
        "Max IV",
        "Reason",
        "Params",
    )
    _WIDTHS = (70, 90, 50, 95, 55, 75, 75, 80, 80, 85, 75, 50, 70, 70, 220, 260)

    def _insert_rows(self, rows):
        for r in rows:
            self.tree.insert(
                "",
                tk.END,
                values=(
                    r["Ticker"],
                    r["As Of"],
                    r["DTE"],
                    r["Expiry"],
                    _model_display_name(r["Model"]),
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
                ),
            )

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
        self._rows = sorted_rows
        self._clear_tree()
        self._insert_rows(sorted_rows)


class SystemHealthTab(ttk.Frame):
    """System integrity dashboard for data/model health and downstream trust."""

    def __init__(self, master, *, on_open_expiry=None, on_open_signals=None):
        super().__init__(master)
        self._on_open_expiry = on_open_expiry
        self._on_open_signals = on_open_signals
        self._rows: list[dict[str, Any]] = []
        self._details_visible = True
        self._selected_expiry: dict[str, Any] | None = None
        self._grid_rows: list[dict[str, Any]] = []
        self._grid_expiries: list[dict[str, Any]] = []
        self._build_ui()

    def _build_ui(self):
        intro = ttk.Label(
            self,
            text=(
                "System Health summarizes whether data, model fits, fallbacks, and weights "
                "are trustworthy enough to support RV Signals."
            ),
            anchor="w",
            foreground="gray35",
            wraplength=1100,
        )
        intro.pack(fill=tk.X, padx=8, pady=(8, 4))

        summary = ttk.LabelFrame(self, text="Summary")
        summary.pack(fill=tk.X, padx=8, pady=(0, 6))
        self.summary_labels: dict[str, ttk.Label] = {}
        specs = [
            ("primary_model", "Primary model"),
            ("rejected_models", "Rejected models"),
            ("partially_reliable_models", "Partially reliable"),
            ("overall", "Overall"),
            ("warnings", "Warnings"),
            ("fallbacks", "Fallbacks"),
        ]
        for i, (key, title) in enumerate(specs):
            frame = ttk.Frame(summary, padding=(6, 4))
            frame.grid(row=i // 3, column=i % 3, sticky="ew", padx=4, pady=3)
            ttk.Label(frame, text=title).pack(anchor="w")
            lbl = ttk.Label(frame, text="—", wraplength=320)
            lbl.pack(anchor="w", fill=tk.X)
            self.summary_labels[key] = lbl
        for col in range(3):
            summary.columnconfigure(col, weight=1)

        msg_box = ttk.LabelFrame(self, text="Interpretation")
        msg_box.pack(fill=tk.X, padx=8, pady=(0, 6))
        self.txt_messages = tk.Text(msg_box, height=4, wrap="word", relief="flat", padx=8, pady=6)
        self.txt_messages.pack(fill=tk.X, expand=False)
        self.txt_messages.configure(state=tk.DISABLED)

        grid_box = ttk.LabelFrame(self, text="Model Health Grid")
        grid_box.pack(fill=tk.X, padx=8, pady=(0, 6))
        self.model_grid = ttk.Treeview(grid_box, columns=("Model",), show="headings", height=3, selectmode="browse")
        self.model_grid.heading("Model", text="Model")
        self.model_grid.column("Model", width=90, anchor=tk.W, stretch=False)
        self.model_grid.pack(fill=tk.X, padx=6, pady=(6, 2))
        self.model_grid.bind("<<TreeviewSelect>>", self._on_model_grid_selected)
        self.model_grid.bind("<ButtonRelease-1>", self._on_model_grid_click)
        self.model_grid.bind("<Double-1>", self._on_model_grid_double_click)
        self.lbl_grid_reason = ttk.Label(
            grid_box,
            text=(
                f"Status threshold: degraded when RMSE > {MODEL_RMSE_DEGRADED_THRESHOLD:.2f}. "
                "Select a cell for reason."
            ),
            anchor="w",
            foreground="gray35",
            wraplength=1100,
        )
        self.lbl_grid_reason.pack(fill=tk.X, padx=6, pady=(0, 6))

        feature_box = ttk.LabelFrame(self, text="Feature Health")
        feature_box.pack(fill=tk.BOTH, expand=False, padx=8, pady=(0, 6))
        self.feature_summary = ttk.Label(
            feature_box, text="No feature diagnostics available.", anchor="w", wraplength=1100
        )
        self.feature_summary.pack(fill=tk.X, padx=6, pady=(6, 2))
        self.feature_warnings = ttk.Label(feature_box, text="", anchor="w", foreground="firebrick", wraplength=1100)
        self.feature_warnings.pack(fill=tk.X, padx=6, pady=(0, 4))

        feature_tables = ttk.Frame(feature_box)
        feature_tables.pack(fill=tk.X, padx=6, pady=(0, 6))
        feature_tables.columnconfigure(0, weight=1)
        feature_tables.columnconfigure(1, weight=1)

        self.feature_dist = ttk.Treeview(
            feature_tables,
            columns=("Ticker", "Coverage", "Mean", "Std", "Min", "Max"),
            show="headings",
            height=5,
        )
        for col, width in (("Ticker", 70), ("Coverage", 80), ("Mean", 80), ("Std", 80), ("Min", 80), ("Max", 80)):
            self.feature_dist.heading(col, text=col)
            self.feature_dist.column(col, width=width, anchor=tk.CENTER)
        self.feature_dist.grid(row=0, column=0, sticky="ew", padx=(0, 4))

        self.feature_pairs = ttk.Treeview(
            feature_tables,
            columns=("Peer", "Corr", "Mean Diff", "Sign %", "Common", "Flag"),
            show="headings",
            height=5,
        )
        for col, width in (
            ("Peer", 70),
            ("Corr", 70),
            ("Mean Diff", 90),
            ("Sign %", 70),
            ("Common", 70),
            ("Flag", 150),
        ):
            self.feature_pairs.heading(col, text=col)
            self.feature_pairs.column(col, width=width, anchor=tk.CENTER)
        self.feature_pairs.grid(row=0, column=1, sticky="ew", padx=(4, 0))

        self.feature_log = ttk.Label(feature_box, text="", anchor="w", foreground="gray35", wraplength=1100)
        self.feature_log.pack(fill=tk.X, padx=6, pady=(0, 6))

        nav = ttk.Frame(self)
        nav.pack(fill=tk.X, padx=8, pady=(0, 6))
        self.btn_open_expiry = ttk.Button(nav, text="Open Expiry in IV Explorer", command=self._open_selected_expiry)
        self.btn_open_expiry.pack(side=tk.LEFT, padx=(0, 6))
        self.btn_open_signals = ttk.Button(nav, text="Open Related RV Signals", command=self._open_signals)
        self.btn_open_signals.pack(side=tk.LEFT, padx=(0, 6))
        self.btn_toggle = ttk.Button(nav, text="Hide Detailed Diagnostics", command=self._toggle_details)
        self.btn_toggle.pack(side=tk.RIGHT)

        self.details = ttk.LabelFrame(self, text="Detailed Diagnostics")
        self.details.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
        self.diagnostics = DiagnosticsTab(self.details)
        self.diagnostics.pack(fill=tk.BOTH, expand=True)
        self.diagnostics.tree.bind("<<TreeviewSelect>>", self._on_detail_selected)
        self.diagnostics.tree.bind("<Double-1>", self._on_detail_double_click)

    def update(self, info: Dict[str, Any] | None) -> None:
        health = summarize_health_info(info)
        grid = build_model_health_grid(info)
        self._rows = flatten_diagnostics_info(info)
        for key, label in self.summary_labels.items():
            value = health.get(key, "—")
            if isinstance(value, list):
                value = ", ".join(value) if value else "None"
            label.config(text=str(value))
        self._set_messages(health.get("messages") or [])
        self._update_model_grid(grid)
        feature_health = info.get("feature_health") if isinstance(info, dict) else None
        if not feature_health and isinstance(info, dict) and isinstance(info.get("weight_info"), dict):
            feature_health = info["weight_info"].get("feature_health")
        self._update_feature_health(feature_health)
        self.diagnostics.update(info)
        self._selected_expiry = None

    def _set_messages(self, messages: list[str]) -> None:
        text = "\n".join(f"- {m}" for m in messages)
        self.txt_messages.configure(state=tk.NORMAL)
        self.txt_messages.delete("1.0", tk.END)
        self.txt_messages.insert(tk.END, text)
        self.txt_messages.configure(state=tk.DISABLED)

    def _on_detail_selected(self, _event=None):
        selected = self.diagnostics.tree.selection()
        if not selected:
            self._selected_expiry = None
            return
        try:
            idx = self.diagnostics.tree.index(selected[0])
            self._selected_expiry = self._rows[idx] if 0 <= idx < len(self._rows) else None
        except Exception:
            self._selected_expiry = None

    def _update_model_grid(self, grid: dict[str, Any]) -> None:
        self._grid_rows = list(grid.get("rows") or [])
        self._grid_expiries = list(grid.get("expiries") or [])
        columns = ("Model",) + tuple(exp["label"] for exp in self._grid_expiries)
        self.model_grid.configure(columns=columns)
        for col in columns:
            self.model_grid.heading(col, text=col)
            self.model_grid.column(
                col,
                width=90 if col == "Model" else 115,
                anchor=tk.W if col == "Model" else tk.CENTER,
                stretch=True,
            )
        for item in self.model_grid.get_children():
            self.model_grid.delete(item)
        for i, row in enumerate(self._grid_rows):
            self.model_grid.insert("", tk.END, iid=str(i), values=[row.get(col, "") for col in columns])
        if self._grid_rows:
            self.lbl_grid_reason.config(
                text=(
                    f"Status threshold: degraded when RMSE > {MODEL_RMSE_DEGRADED_THRESHOLD:.2f}. "
                    "Select a cell for reason."
                )
            )
        else:
            self.lbl_grid_reason.config(
                text="No model health grid available. Plot an IV view to populate model fit diagnostics."
            )

    def _grid_cell_context(self, event=None) -> tuple[dict[str, Any] | None, str]:
        selected = self.model_grid.selection()
        iid = selected[0] if selected else ""
        if event is not None:
            row_id = self.model_grid.identify_row(event.y)
            if row_id:
                iid = row_id
        try:
            row = self._grid_rows[int(iid)]
        except Exception:
            return None, ""
        col_name = ""
        if event is not None:
            col_id = self.model_grid.identify_column(event.x)
            try:
                col_name = self.model_grid["columns"][int(col_id.replace("#", "")) - 1]
            except Exception:
                col_name = ""
        if not col_name or col_name == "Model":
            for exp in self._grid_expiries:
                label = exp["label"]
                if row.get("_statuses", {}).get(label) in {"degraded", "failed"}:
                    col_name = label
                    break
            if not col_name and self._grid_expiries:
                col_name = self._grid_expiries[0]["label"]
        return row, col_name

    def _on_model_grid_selected(self, event=None):
        row, col_name = self._grid_cell_context(event)
        if not row or not col_name:
            return
        reason = row.get("_reasons", {}).get(col_name, "")
        status = row.get("_statuses", {}).get(col_name, "")
        model = row.get("Model", "")
        self.lbl_grid_reason.config(
            text=f"{model} {col_name}: {status or 'unknown'} - {reason or 'No reason recorded.'}"
        )
        for exp in self._grid_expiries:
            if exp["label"] == col_name:
                self._selected_expiry = {"DTE": exp.get("dte"), "Expiry": exp.get("expiry")}
                break

    def _on_model_grid_click(self, event):
        self._on_model_grid_selected(event)

    def _on_model_grid_double_click(self, event):
        row, col_name = self._grid_cell_context(event)
        if row and col_name and col_name != "Model":
            self._on_model_grid_selected(event)
            self._open_selected_expiry()

    def _update_feature_health(self, feature_health: dict[str, Any] | None) -> None:
        for tree in (self.feature_dist, self.feature_pairs):
            for item in tree.get_children():
                tree.delete(item)

        if not isinstance(feature_health, dict) or not feature_health.get("available"):
            self.feature_summary.config(
                text="No feature diagnostics available. Plot the Relative Weight Matrix to inspect feature construction."  # noqa: E501
            )
            self.feature_warnings.config(text="")
            self.feature_log.config(text="")
            return

        summary = feature_health.get("summary") or {}
        alignment = feature_health.get("alignment") or {}
        self.feature_summary.config(
            text=(
                f"Feature: {summary.get('feature_set', 'unknown')} | "
                f"Grid: {summary.get('coordinate_system', 'unknown')} | "
                f"Points: {summary.get('total_points', 0)} | "
                f"Normalization: {summary.get('normalization', 'unknown')} | "
                f"Shared grid: {'yes' if alignment.get('shared_grid') else 'no'} | "
                f"Sparse points: {alignment.get('sparse_points', 0)}"
            )
        )
        warnings = feature_health.get("warnings") or []
        self.feature_warnings.config(
            text=("Warnings: " + " ".join(str(w) for w in warnings)) if warnings else "Warnings: none detected."
        )

        for row in feature_health.get("distribution") or []:
            self.feature_dist.insert(
                "",
                tk.END,
                values=(
                    row.get("ticker", ""),
                    _pct(row.get("coverage")),
                    _fmt(row.get("mean"), ".4f"),
                    _fmt(row.get("std"), ".4f"),
                    _fmt(row.get("min"), ".4f"),
                    _fmt(row.get("max"), ".4f"),
                ),
            )
        for row in feature_health.get("pairs") or []:
            self.feature_pairs.insert(
                "",
                tk.END,
                values=(
                    row.get("ticker", ""),
                    _fmt(row.get("correlation"), ".3f"),
                    _fmt(row.get("mean_difference"), ".4f"),
                    _pct(row.get("sign_consistency")),
                    row.get("common_points", ""),
                    row.get("flag", ""),
                ),
            )
        self.feature_log.config(
            text="Pipeline: " + " → ".join(str(x) for x in feature_health.get("transformation_log") or [])
        )

    def _open_selected_expiry(self):
        if callable(self._on_open_expiry):
            self._on_open_expiry(self._selected_expiry)

    def _open_signals(self):
        if callable(self._on_open_signals):
            self._on_open_signals(self._selected_expiry)

    def _on_detail_double_click(self, event):
        region = self.diagnostics.tree.identify("region", event.x, event.y)
        if region != "cell":
            return
        col_id = self.diagnostics.tree.identify_column(event.x)
        try:
            col_name = self.diagnostics.tree["columns"][int(col_id.replace("#", "")) - 1]
        except Exception:
            col_name = ""
        if col_name in {"Expiry", "DTE"}:
            self._on_detail_selected()
            self._open_selected_expiry()

    def _toggle_details(self):
        if self._details_visible:
            self.details.pack_forget()
            self.btn_toggle.config(text="Show Detailed Diagnostics")
            self._details_visible = False
        else:
            self.details.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
            self.btn_toggle.config(text="Hide Detailed Diagnostics")
            self._details_visible = True
