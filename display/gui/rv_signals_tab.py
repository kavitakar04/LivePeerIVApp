"""
RV Signals tab for the main GUI browser.

Displays a ranked, colour-coded table of relative-value dislocation
signals for the currently selected target and peers.  Heavy computation
runs in a background thread; the Treeview is updated on the Tk main
thread via ``after()``.
"""

from __future__ import annotations

import threading
import sys
from pathlib import Path

import tkinter as tk
from tkinter import ttk

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


_COLUMNS = (
    "signal_type",
    "T_days",
    "value",
    "synth_value",
    "spread",
    "z_score",
    "pct_rank",
    "description",
)

_HEADERS = {
    "signal_type": ("Type", 120),
    "T_days":       ("T (d)", 60),
    "value":        ("Target", 85),
    "synth_value":  ("Synthetic", 85),
    "spread":       ("Spread", 85),
    "z_score":      ("Z-Score", 75),
    "pct_rank":     ("Pct Rank", 75),
    "description":  ("Description", 260),
}


class RVSignalsFrame(ttk.Frame):
    """Tab showing ranked RV signals for the current target/peers."""

    def __init__(self, master, input_panel=None, **kwargs):
        super().__init__(master, **kwargs)
        self._input_panel = input_panel
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        # ---- Controls row ----
        ctrl = ttk.Frame(self)
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        ttk.Label(ctrl, text="Lookback (days):").grid(row=0, column=0, sticky="w")
        self.ent_lookback = ttk.Entry(ctrl, width=6)
        self.ent_lookback.insert(0, "60")
        self.ent_lookback.grid(row=0, column=1, padx=4)

        ttk.Label(ctrl, text="Min |Z|:").grid(row=0, column=2, sticky="w", padx=(8, 0))
        self.ent_min_z = ttk.Entry(ctrl, width=5)
        self.ent_min_z.insert(0, "1.0")
        self.ent_min_z.grid(row=0, column=3, padx=4)

        self.btn_refresh = ttk.Button(
            ctrl, text="Refresh Signals", command=self._refresh
        )
        self.btn_refresh.grid(row=0, column=4, padx=8)

        self.lbl_status = ttk.Label(ctrl, text="Press Refresh to compute signals.")
        self.lbl_status.grid(row=0, column=5, padx=8, sticky="w")

        # ---- Legend ----
        legend_frame = ttk.Frame(self)
        legend_frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0, 2))
        tk.Label(legend_frame, text="  Rich (z>1.5) ", bg="#ffe8e8", relief="solid",
                 bd=1, padx=4).pack(side=tk.LEFT)
        tk.Label(legend_frame, text="  Cheap (z<−1.5) ", bg="#e8f0ff", relief="solid",
                 bd=1, padx=4).pack(side=tk.LEFT, padx=(4, 0))
        tk.Label(legend_frame, text="  Neutral / no z ", bg="#f5f5f5", relief="solid",
                 bd=1, padx=4).pack(side=tk.LEFT, padx=(4, 0))

        # ---- Treeview ----
        tree_frame = ttk.Frame(self)
        tree_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=4)

        self.tree = ttk.Treeview(
            tree_frame, columns=_COLUMNS, show="headings", selectmode="browse"
        )
        for col in _COLUMNS:
            header, width = _HEADERS[col]
            self.tree.heading(col, text=header)
            anchor = "w" if col in ("signal_type", "description") else "center"
            self.tree.column(col, width=width, anchor=anchor, stretch=(col == "description"))

        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)

        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.LEFT, fill=tk.Y)

        # Colour tags
        self.tree.tag_configure("rich",    background="#ffe8e8")
        self.tree.tag_configure("cheap",   background="#e8f0ff")
        self.tree.tag_configure("neutral", background="#f5f5f5")

    # ------------------------------------------------------------------
    # Public callback (called by browser when selection changes)
    # ------------------------------------------------------------------
    def on_browser_selection_changed(self):
        """Called by BrowserApp when target / peers change."""
        # Silently clear the table so stale results are not shown
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.lbl_status.config(text="Press Refresh to compute signals.")

    # ------------------------------------------------------------------
    # Refresh logic (background thread + Tk marshal)
    # ------------------------------------------------------------------
    def _refresh(self):
        if self._input_panel is None:
            self.lbl_status.config(text="No input panel connected.")
            return

        settings = self._input_panel.get_settings()
        target = (settings.get("target") or "").upper().strip()
        peers = [
            p.upper().strip()
            for p in (settings.get("peers") or [])
            if str(p).strip()
        ]
        asof = settings.get("asof")

        weight_method = settings.get("weight_method", "corr")
        feature_mode = settings.get("feature_mode", "iv_atm")
        weight_mode = (
            "oi" if weight_method == "oi"
            else f"{weight_method}_{feature_mode}"
        )

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

        self.lbl_status.config(text="Computing…")
        self.btn_refresh.config(state=tk.DISABLED)

        def worker():
            try:
                from analysis.rv_analysis import generate_rv_signals
                signals = generate_rv_signals(
                    target=target,
                    peers=peers,
                    asof=asof,
                    weight_mode=weight_mode,
                    lookback=lookback,
                    min_abs_z=min_z,
                )
                count = len(signals) if signals is not None else 0
                self.after(0, lambda: self._populate_table(signals))
                self.after(
                    0,
                    lambda c=count: self.lbl_status.config(
                        text=f"{c} signal(s) found."
                    ),
                )
            except Exception as exc:
                msg = str(exc)
                self.after(
                    0,
                    lambda m=msg: self.lbl_status.config(text=f"Error: {m}"),
                )
            finally:
                self.after(0, lambda: self.btn_refresh.config(state=tk.NORMAL))

        threading.Thread(target=worker, daemon=True).start()

    # ------------------------------------------------------------------
    # Table population
    # ------------------------------------------------------------------
    def _populate_table(self, signals_df):
        """Update the Treeview from the signals DataFrame on the main thread."""
        import numpy as np
        import pandas as pd

        for item in self.tree.get_children():
            self.tree.delete(item)

        if signals_df is None or signals_df.empty:
            self.tree.insert(
                "", "end",
                values=("No signals", "", "", "", "", "", "", "Insufficient data"),
                tags=("neutral",),
            )
            return

        for _, row in signals_df.iterrows():
            vals = []
            for col in _COLUMNS:
                v = row.get(col, "")
                if isinstance(v, float):
                    if col in ("z_score", "pct_rank"):
                        vals.append(f"{v:.2f}" if (pd.notna(v) and np.isfinite(v)) else "—")
                    else:
                        vals.append(f"{v:.4f}" if (pd.notna(v) and np.isfinite(v)) else "—")
                else:
                    vals.append(str(v) if v != "" and pd.notna(v) else "—")

            z = row.get("z_score")
            z_f = float(z) if (pd.notna(z) and np.isfinite(float(z) if pd.notna(z) else float("nan"))) else float("nan")
            if np.isfinite(z_f) and z_f > 1.5:
                tag = "rich"
            elif np.isfinite(z_f) and z_f < -1.5:
                tag = "cheap"
            else:
                tag = "neutral"

            self.tree.insert("", "end", values=vals, tags=(tag,))
