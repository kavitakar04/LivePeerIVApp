from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Dict, Any, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def _fmt(val, fmt=".4f", fallback="—") -> str:
    try:
        return format(float(val), fmt)
    except (TypeError, ValueError):
        return fallback


def _pct(val, fallback="—") -> str:
    try:
        return f"{float(val) * 100:.2f}%"
    except (TypeError, ValueError):
        return fallback


class ParametersTab(ttk.Frame):
    """Term structure summary: ATM vol, skew, curvature, fit quality per expiry."""

    _COLS = ("DTE", "Expiry", "ATM Vol", "Skew", "Curvature", "RMSE", "N")
    _WIDTHS = (50, 100, 80, 80, 80, 80, 50)

    def __init__(self, master):
        super().__init__(master)

        self.lbl_meta = ttk.Label(self, text="", anchor="w", foreground="gray")
        self.lbl_meta.pack(fill=tk.X, padx=6, pady=(6, 2))

        # ---- table ----
        tbl_frame = ttk.Frame(self)
        tbl_frame.pack(fill=tk.BOTH, expand=False, padx=6, pady=(0, 4))

        self.tree = ttk.Treeview(tbl_frame, columns=self._COLS,
                                  show="headings", height=8, selectmode="browse")
        vsb = ttk.Scrollbar(tbl_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)

        for col, w in zip(self._COLS, self._WIDTHS):
            anchor = tk.W if col == "Expiry" else tk.E
            self.tree.heading(col, text=col,
                              command=lambda c=col: self._sort_by(c))
            self.tree.column(col, anchor=anchor, width=w, stretch=(col == "Expiry"))

        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        # ---- term structure plot ----
        self.fig, self.ax = plt.subplots(figsize=(5, 2.4))
        self.fig.subplots_adjust(left=0.12, right=0.97, top=0.88, bottom=0.18)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))

        self._sort_col: Optional[str] = None
        self._sort_asc: bool = True
        self._rows: list = []

    def update(self, info: Dict[str, Any] | None) -> None:
        self._clear()

        if not info:
            self.lbl_meta.config(text="No fit data")
            self._draw_empty()
            return

        ticker = info.get("ticker", "")
        asof = info.get("asof", "")
        self.lbl_meta.config(text=f"{ticker}   {asof}" if ticker or asof else "")

        fit_map = info.get("fit_by_expiry") if isinstance(info, dict) else None
        if not fit_map:
            self._draw_empty()
            return

        rows = []
        for T_val, entry in sorted(fit_map.items(), key=lambda kv: kv[0]):
            dte = int(round(float(T_val) * 365.25))
            expiry = entry.get("expiry") or ""
            # expiry might be a full datetime string — keep date part only
            if expiry and "T" in expiry:
                expiry = expiry.split("T")[0]
            elif expiry and " " in expiry:
                expiry = expiry.split(" ")[0]

            sens = entry.get("sens") or {}
            atm_vol = sens.get("atm_vol")
            skew = sens.get("skew")
            curv = sens.get("curv")

            # best available RMSE + N
            rmse, n = None, None
            for model_key in ("svi", "sabr", "tps"):
                mp = entry.get(model_key) or {}
                if mp.get("rmse") is not None:
                    rmse = mp["rmse"]
                    n = mp.get("n")
                    break

            rows.append({
                "DTE": dte,
                "Expiry": expiry,
                "ATM Vol": atm_vol,
                "Skew": skew,
                "Curvature": curv,
                "RMSE": rmse,
                "N": n,
            })

        self._rows = rows
        self._insert_rows(rows)
        self._draw_chart(rows)

    # ---- internals ----

    def _clear(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        self._rows = []

    def _insert_rows(self, rows):
        for r in rows:
            self.tree.insert("", tk.END, values=(
                r["DTE"],
                r["Expiry"],
                _pct(r["ATM Vol"]),
                _pct(r["Skew"]),
                _fmt(r["Curvature"], ".4f"),
                _fmt(r["RMSE"], ".5f"),
                str(r["N"]) if r["N"] is not None else "—",
            ))

    def _sort_by(self, col: str):
        if self._sort_col == col:
            self._sort_asc = not self._sort_asc
        else:
            self._sort_col = col
            self._sort_asc = True

        key_map = {
            "DTE": lambda r: r["DTE"],
            "Expiry": lambda r: r["Expiry"] or "",
            "ATM Vol": lambda r: float(r["ATM Vol"]) if r["ATM Vol"] is not None else -999,
            "Skew": lambda r: float(r["Skew"]) if r["Skew"] is not None else -999,
            "Curvature": lambda r: float(r["Curvature"]) if r["Curvature"] is not None else -999,
            "RMSE": lambda r: float(r["RMSE"]) if r["RMSE"] is not None else 999,
            "N": lambda r: int(r["N"]) if r["N"] is not None else -1,
        }
        key = key_map.get(col, lambda r: 0)
        sorted_rows = sorted(self._rows, key=key, reverse=not self._sort_asc)
        self._clear_tree()
        self._insert_rows(sorted_rows)

    def _clear_tree(self):
        for item in self.tree.get_children():
            self.tree.delete(item)

    def _draw_chart(self, rows):
        self.ax.clear()
        dtes = [r["DTE"] for r in rows if r["ATM Vol"] is not None]
        vols = [float(r["ATM Vol"]) * 100 for r in rows if r["ATM Vol"] is not None]
        skews = [float(r["Skew"]) * 100 for r in rows if r["Skew"] is not None]

        if not dtes:
            self._draw_empty()
            return

        self.ax.plot(dtes, vols, "o-", color="#1f77b4", lw=1.5, ms=5, label="ATM Vol %")

        if len(skews) == len(dtes):
            ax2 = self.ax.twinx()
            ax2.plot(dtes, skews, "s--", color="#d62728", lw=1, ms=4, alpha=0.75, label="Skew %")
            ax2.set_ylabel("Skew (%)", fontsize=7, color="#d62728")
            ax2.tick_params(axis="y", labelsize=7, colors="#d62728")
            ax2.axhline(0, color="#d62728", lw=0.5, alpha=0.4)

        self.ax.set_xlabel("DTE (days)", fontsize=7)
        self.ax.set_ylabel("ATM Vol (%)", fontsize=7, color="#1f77b4")
        self.ax.tick_params(axis="both", labelsize=7)
        self.ax.tick_params(axis="y", colors="#1f77b4")
        self.ax.set_title("Term Structure", fontsize=8)

        self.fig.canvas.draw_idle()

    def _draw_empty(self):
        self.ax.clear()
        self.ax.set_title("No data", fontsize=8)
        self.fig.canvas.draw_idle()
