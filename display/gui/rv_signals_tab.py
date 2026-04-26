"""
RV Signals tab for the main GUI browser.

This tab is the final synthesis layer for relative-value research.  It turns
target-vs-peer IV, model quality, spillover, and surface-comparability context
into ranked trade theses rather than exposing a raw spread table.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import tkinter as tk
from tkinter import ttk

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


_COLUMNS = (
    "rank",
    "opportunity",
    "direction",
    "feature",
    "maturity",
    "spread",
    "z_score",
    "percentile",
    "confidence",
    "event_context",
    "spillover_support",
    "data_quality",
    "why",
)

_HEADERS = {
    "rank": ("Rank", 54),
    "opportunity": ("Opportunity", 260),
    "direction": ("Direction", 82),
    "feature": ("Feature", 110),
    "maturity": ("Maturity", 82),
    "spread": ("Spread", 82),
    "z_score": ("Z-Score", 82),
    "percentile": ("Percentile", 86),
    "confidence": ("Confidence", 94),
    "event_context": ("Event Context", 118),
    "spillover_support": ("Spillover Support", 170),
    "data_quality": ("Data Quality", 105),
    "why": ("Why", 380),
}


class RVSignalsFrame(ttk.Frame):
    """Ranked opportunity dashboard for the current target/peer group."""

    def __init__(self, master, input_panel=None, **kwargs):
        super().__init__(master, **kwargs)
        self._input_panel = input_panel
        self._last_opportunities = None
        self._build_ui()

    def _build_ui(self):
        ctrl = ttk.Frame(self)
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        ttk.Label(ctrl, text="Lookback").grid(row=0, column=0, sticky="w")
        self.ent_lookback = ttk.Entry(ctrl, width=6)
        self.ent_lookback.insert(0, "60")
        self.ent_lookback.grid(row=0, column=1, padx=(4, 12))

        ttk.Label(ctrl, text="Min |Z|").grid(row=0, column=2, sticky="w")
        self.ent_min_z = ttk.Entry(ctrl, width=5)
        self.ent_min_z.insert(0, "1.0")
        self.ent_min_z.grid(row=0, column=3, padx=(4, 12))

        self.btn_refresh = ttk.Button(ctrl, text="Refresh Signals", command=self._refresh)
        self.btn_refresh.grid(row=0, column=4, padx=(0, 12))

        self.lbl_status = ttk.Label(ctrl, text="Press Refresh to synthesize RV opportunities.")
        self.lbl_status.grid(row=0, column=5, sticky="w")
        ctrl.columnconfigure(5, weight=1)

        top = ttk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0, 6))
        top.columnconfigure(0, weight=3)
        top.columnconfigure(1, weight=4)

        summary_box = ttk.LabelFrame(top, text="Executive Summary")
        summary_box.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self.summary_text = tk.Text(
            summary_box,
            height=6,
            wrap="word",
            relief="flat",
            background="#f7f8fa",
            padx=8,
            pady=6,
        )
        self.summary_text.pack(fill=tk.BOTH, expand=True)
        self.summary_text.configure(state=tk.DISABLED)

        cards = ttk.LabelFrame(top, text="Context")
        cards.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        self.card_labels: dict[str, ttk.Label] = {}
        card_specs = [
            ("strongest_dislocation", "Strongest dislocation"),
            ("most_tradeable", "Most tradeable"),
            ("most_systemic", "Most systemic"),
            ("weakest_signal", "Weakest / least reliable"),
            ("data_quality_warnings", "Data-quality warnings"),
        ]
        for i, (key, title) in enumerate(card_specs):
            frame = ttk.Frame(cards, padding=(6, 4))
            frame.grid(row=i // 2, column=i % 2, sticky="ew", padx=4, pady=3)
            ttk.Label(frame, text=title).pack(anchor="w")
            value = ttk.Label(frame, text="Unavailable", wraplength=310, foreground="#333333")
            value.pack(anchor="w", fill=tk.X)
            self.card_labels[key] = value
        cards.columnconfigure(0, weight=1)
        cards.columnconfigure(1, weight=1)

        main = ttk.PanedWindow(self, orient=tk.VERTICAL)
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        table_frame = ttk.Frame(main)
        main.add(table_frame, weight=3)
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)

        self.tree = ttk.Treeview(table_frame, columns=_COLUMNS, show="headings", selectmode="browse")
        for col in _COLUMNS:
            header, width = _HEADERS[col]
            self.tree.heading(col, text=header)
            anchor = "w" if col in {"opportunity", "why", "spillover_support"} else "center"
            self.tree.column(col, width=width, anchor=anchor, stretch=(col in {"opportunity", "why"}))
        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(table_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")

        self.tree.tag_configure("rich", background="#ffe8e8")
        self.tree.tag_configure("cheap", background="#e8f0ff")
        self.tree.tag_configure("neutral", background="#f5f5f5")
        self.tree.tag_configure("warning", background="#fff6d9")
        self.tree.bind("<<TreeviewSelect>>", self._on_row_selected)

        detail = ttk.LabelFrame(main, text="Signal Detail")
        main.add(detail, weight=2)
        self.detail_text = tk.Text(detail, height=10, wrap="word", relief="flat", padx=8, pady=6)
        self.detail_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        detail_vsb = ttk.Scrollbar(detail, orient="vertical", command=self.detail_text.yview)
        self.detail_text.configure(yscrollcommand=detail_vsb.set)
        detail_vsb.pack(side=tk.LEFT, fill=tk.Y)
        self.detail_text.configure(state=tk.DISABLED)

    def on_browser_selection_changed(self):
        """Called by BrowserApp when target / peers change."""
        self._last_opportunities = None
        self._clear_dashboard("Press Refresh to synthesize RV opportunities.")

    def _clear_dashboard(self, status: str):
        for item in self.tree.get_children():
            self.tree.delete(item)
        self._set_summary([])
        self._set_detail("")
        for label in self.card_labels.values():
            label.config(text="Unavailable")
        self.lbl_status.config(text=status)

    def _refresh(self):
        if self._input_panel is None:
            self.lbl_status.config(text="No input panel connected.")
            return

        settings = self._input_panel.get_settings()
        target = (settings.get("target") or "").upper().strip()
        peers = [p.upper().strip() for p in (settings.get("peers") or []) if str(p).strip()]
        asof = settings.get("asof")
        weight_method = settings.get("weight_method", "corr")
        feature_mode = settings.get("feature_mode", "iv_atm")
        weight_mode = "oi" if weight_method == "oi" else f"{weight_method}_{feature_mode}"
        try:
            max_expiries = int(settings.get("max_expiries", 6) or 6)
        except (TypeError, ValueError):
            max_expiries = 6

        if not target or not peers or not asof:
            self.lbl_status.config(text="Set target, peers, and date first.")
            return

        try:
            lookback = int(self.ent_lookback.get())
        except (ValueError, tk.TclError):
            lookback = 60

        try:
            min_z = float(self.ent_min_z.get())
        except (ValueError, tk.TclError):
            min_z = 1.0

        self.lbl_status.config(text="Synthesizing opportunities...")
        self.btn_refresh.config(state=tk.DISABLED)

        def worker():
            try:
                from analysis.rv_analysis import generate_rv_opportunity_dashboard

                payload = generate_rv_opportunity_dashboard(
                    target=target,
                    peers=peers,
                    asof=asof,
                    weight_mode=weight_mode,
                    lookback=lookback,
                    max_expiries=max_expiries,
                    min_abs_z=min_z,
                )
                self.after(0, lambda p=payload: self._populate_dashboard(p))
            except Exception as exc:
                msg = str(exc)
                self.after(0, lambda m=msg: self.lbl_status.config(text=f"Error: {m}"))
            finally:
                self.after(0, lambda: self.btn_refresh.config(state=tk.NORMAL))

        threading.Thread(target=worker, daemon=True).start()

    def _populate_dashboard(self, payload):
        import numpy as np
        import pandas as pd

        opportunities = payload.get("opportunities")
        self._last_opportunities = opportunities
        self._set_summary(payload.get("executive_summary") or [])

        cards = payload.get("context_cards") or {}
        for key, label in self.card_labels.items():
            label.config(text=str(cards.get(key, "Unavailable")))

        for item in self.tree.get_children():
            self.tree.delete(item)

        if opportunities is None or opportunities.empty:
            self.tree.insert(
                "", "end",
                values=("", "No opportunities", "", "", "", "", "", "", "", "", "", "", "Insufficient data or no threshold breach."),
                tags=("neutral",),
            )
            warnings = payload.get("warnings") or []
            self._set_detail("\n".join(warnings) if warnings else "No selected signal.")
            self.lbl_status.config(text="No opportunity theses found.")
            return

        for idx, row in opportunities.iterrows():
            values = []
            for col in _COLUMNS:
                value = row.get(col, "")
                if col == "spread":
                    f = float(value) if pd.notna(value) and np.isfinite(float(value)) else np.nan
                    values.append(f"{f:+.2%}" if np.isfinite(f) else "-")
                elif col == "z_score":
                    f = float(value) if pd.notna(value) and np.isfinite(float(value)) else np.nan
                    values.append(f"{f:+.2f}" if np.isfinite(f) else "-")
                elif col == "percentile":
                    f = float(value) if pd.notna(value) and np.isfinite(float(value)) else np.nan
                    values.append(f"{f:.0f}" if np.isfinite(f) else "-")
                else:
                    values.append(str(value) if value != "" and pd.notna(value) else "-")

            direction = str(row.get("direction", "Neutral"))
            data_quality = str(row.get("data_quality", ""))
            if data_quality in {"Degraded", "Poor", "Unknown"}:
                tag = "warning"
            elif direction == "Rich":
                tag = "rich"
            elif direction == "Cheap":
                tag = "cheap"
            else:
                tag = "neutral"
            self.tree.insert("", "end", iid=str(idx), values=values, tags=(tag,))

        first = self.tree.get_children()
        if first:
            self.tree.selection_set(first[0])
            self.tree.focus(first[0])
            self._render_detail_for_index(int(first[0]))
        warning_count = len(payload.get("warnings") or [])
        self.lbl_status.config(text=f"{len(opportunities)} opportunity thesis/theses synthesized; {warning_count} warning(s).")

    def _set_summary(self, items):
        text = "\n".join(f"{i + 1}. {item}" for i, item in enumerate(items))
        if not text:
            text = "Refresh to generate synthesized conclusions for the selected peer group."
        self.summary_text.configure(state=tk.NORMAL)
        self.summary_text.delete("1.0", tk.END)
        self.summary_text.insert(tk.END, text)
        self.summary_text.configure(state=tk.DISABLED)

    def _set_detail(self, text: str):
        self.detail_text.configure(state=tk.NORMAL)
        self.detail_text.delete("1.0", tk.END)
        self.detail_text.insert(tk.END, text or "Select a signal to inspect the thesis.")
        self.detail_text.configure(state=tk.DISABLED)

    def _on_row_selected(self, _event=None):
        selected = self.tree.selection()
        if not selected:
            return
        try:
            self._render_detail_for_index(int(selected[0]))
        except ValueError:
            self._set_detail("No selected signal.")

    def _render_detail_for_index(self, idx: int):
        if self._last_opportunities is None or idx not in self._last_opportunities.index:
            self._set_detail("No selected signal.")
            return
        row = self._last_opportunities.loc[idx]
        warnings = str(row.get("warnings", "") or "None")
        detail = (
            f"Opportunity: {row.get('opportunity', '')}\n\n"
            f"What differs vs peers:\n{row.get('what_differs', '')}\n\n"
            f"Why it matters:\n{row.get('why_matters', '')}\n\n"
            f"Statistical read:\n{row.get('statistical_read', '')}\n\n"
            f"Structural comparability:\n{row.get('comparability', '')}\n\n"
            f"Spillover / convergence support:\n{row.get('spillover_support', '')}\n\n"
            f"Event context:\n{row.get('event_context', '')}\n\n"
            f"Warnings:\n{warnings}"
        )
        self._set_detail(detail)
