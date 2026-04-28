"""
Extended GUI for displaying model parameters in IV_Correlation.

This module defines two widgets:

* `ModelParamsFrame`: A frame that allows the user to select a ticker, model,
  and cutoff date and then plots the time series of calibrated model
  parameters.  It now also embeds a `ParametersTab` underneath the plot to
  show the latest parameter values in a tabular view.

* `ParametersTab`: A table widget implemented with `ttk.Treeview` that can
  display model and sensitivity parameters from a `pandas.DataFrame`.  It
  groups the data by model and parameter name and shows the most recent
  values along with the as‑of date and any available fit metric (such as
  RMSE).  It also retains backward compatibility with the original
  dictionary‑based API so that callers can provide a simple mapping of
  model names to parameter dictionaries.

This code is intended to live in the GUI layer of the IV_Correlation
project (e.g. ``display/gui/model_params_gui.py``) and does not add any
new dependencies beyond ``tkinter``, ``matplotlib`` and ``pandas``.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from typing import Dict, Any, Optional

import pandas as pd
from volModel.models import GUI_MODELS

# Attempt to import load_model_params from the analysis package.  If this
# import fails (for example when running the file stand‑alone for testing),
# callers must ensure that `load_model_params` is on the PYTHONPATH.
try:
    from analysis.persistence.model_params_logger import load_model_params
except Exception:
    # Fallback stub for load_model_params if analysis package is unavailable.
    def load_model_params() -> pd.DataFrame:  # type: ignore[misc]
        raise ImportError("load_model_params could not be imported; ensure analysis package is available")


class ParametersTab(ttk.Frame):
    """Table view for model and sensitivity parameters.

    This widget displays parameter values grouped by model and parameter name.
    It accepts either a ``pandas.DataFrame`` (preferred) or a dictionary of
    model names to parameter dictionaries for backward compatibility.  When
    provided with a DataFrame, it automatically extracts the most recent
    value for each parameter and shows the as‑of date and an optional
    fit metric (e.g. RMSE) if present in the input.
    """

    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master)
        # Define table columns: model name, parameter name, value, as‑of date, metric
        cols = ("Model", "Parameter", "Value", "As of", "Metric")
        tree_frame = ttk.Frame(self)
        tree_frame.pack(fill=tk.BOTH, expand=True)

        self.tree = ttk.Treeview(tree_frame, columns=cols, show="headings")
        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        for c in cols:
            self.tree.heading(c, text=c)
            # Left‑align text and allow columns to stretch
            self.tree.column(c, anchor=tk.W, stretch=True)

        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        tree_frame.rowconfigure(0, weight=1)
        tree_frame.columnconfigure(0, weight=1)

    def update(self, info: Optional[Dict[str, Any] | pd.DataFrame]) -> None:
        """Update table with latest fit information.

        Parameters
        ----------
        info : dict or pandas.DataFrame or None
            If ``info`` is a DataFrame, it should have at least the columns
            ``['model', 'param', 'value', 'asof_date']`` and may optionally
            include a ``'rmse'`` column.  The most recent value for each
            ``(model, param)`` group will be displayed.  If a dictionary is
            provided, it should map model names to nested dictionaries of
            parameter names and values, as in the original implementation.
        """
        # Clear any existing rows
        for i in self.tree.get_children():
            self.tree.delete(i)
        if not info:
            return

        # DataFrame path: group by model and parameter name and extract latest
        if isinstance(info, pd.DataFrame):
            df = info.copy()
            # Ensure required columns are present
            required = {"model", "param", "value", "asof_date"}
            if not required.issubset(df.columns):
                raise ValueError(f"DataFrame must contain columns {required}, got {set(df.columns)}")
            # Sort by asof_date to ensure most recent entries are last
            df = df.sort_values("asof_date")
            # Iterate over unique (model, param) combinations
            for (model, param_name), group in df.groupby(["model", "param"]):
                # Take the last row in the group (most recent)
                row = group.iloc[-1]
                val = row["value"]
                as_of = row["asof_date"]
                # Extract metric if available
                metric = row.get("rmse") or row.get("metric") or ""
                # Format value as float if possible
                try:
                    val = float(val)
                    val_str = f"{val:.6g}"
                except Exception:
                    val_str = str(val)
                # Format as‑of date
                try:
                    as_of_str = pd.to_datetime(as_of).strftime("%Y‑%m‑%d")
                except Exception:
                    as_of_str = str(as_of)
                # Format metric
                metric_str = f"{metric:.4g}" if isinstance(metric, (int, float)) else str(metric)
                self.tree.insert(
                    "",
                    tk.END,
                    values=(model.upper(), param_name, val_str, as_of_str, metric_str),
                )
            return

        # Dictionary path: iterate over nested dicts
        if isinstance(info, dict):
            for model, params in info.items():
                if not isinstance(params, dict):
                    continue
                for k, v in params.items():
                    try:
                        val = float(v)
                        val_str = f"{val:.6g}"
                    except Exception:
                        val_str = str(v)
                    self.tree.insert(
                        "",
                        tk.END,
                        values=(model, k, val_str, "", ""),
                    )


class ModelParamsFrame(ttk.Frame):
    """Frame providing a simple interface for model parameter time series and tabular view."""

    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master)
        self.pack(fill=tk.BOTH, expand=True)

        # Controls for selecting ticker, model and as‑of date
        ctrl = ttk.Frame(self)
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        ttk.Label(ctrl, text="Ticker:").grid(row=0, column=0, sticky=tk.W)
        self.ent_ticker = ttk.Entry(ctrl, width=12)
        self.ent_ticker.grid(row=0, column=1, padx=4)

        ttk.Label(ctrl, text="Model:").grid(row=0, column=2, sticky=tk.W)
        self.cmb_model = ttk.Combobox(
            ctrl,
            values=list(GUI_MODELS),
            width=8,
            state="readonly",
        )
        self.cmb_model.set("svi")
        self.cmb_model.grid(row=0, column=3, padx=4)

        ttk.Label(ctrl, text="As of ≤:").grid(row=0, column=4, sticky=tk.W)
        self.ent_asof = ttk.Entry(ctrl, width=12)
        self.ent_asof.grid(row=0, column=5, padx=4)

        btn_plot = ttk.Button(ctrl, text="Plot", command=self._plot)
        btn_plot.grid(row=0, column=6, padx=4)

        # Matplotlib figure for plotting parameter time series
        self.fig = plt.Figure(figsize=(6, 4))
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Embedded ParametersTab for displaying latest parameter values
        self.table = ParametersTab(self)
        self.table.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def _plot(self) -> None:
        ticker = self.ent_ticker.get().strip().upper()
        model = self.cmb_model.get().strip().lower()
        asof = self.ent_asof.get().strip()

        # Clear existing plot
        self.ax.clear()
        if not ticker:
            self.ax.text(0.5, 0.5, "Enter ticker", ha="center", va="center")
            self.canvas.draw()
            return

        try:
            df = load_model_params()
        except Exception as exc:
            messagebox.showerror("Data error", f"Failed to load model parameters: {exc}")
            self.canvas.draw()
            return
        # Filter by ticker and model
        df = df[(df["ticker"] == ticker) & (df["model"] == model)]
        # Apply as‑of cutoff if provided
        if asof:
            try:
                cutoff = pd.to_datetime(asof)
                df = df[df["asof_date"] <= cutoff]
            except Exception:
                messagebox.showerror("Input error", "Invalid asof date")
                self.canvas.draw()
                return
        if df.empty:
            self.ax.text(0.5, 0.5, "No parameter data", ha="center", va="center")
            self.ax.set_title("Model Parameters")
            self.canvas.draw()
            # Clear the table as well
            self.table.update(None)
            return
        # Sort by date for plotting
        df = df.sort_values("asof_date")
        # Plot each parameter as a separate line
        for param_name in df["param"].unique():
            sub = df[df["param"] == param_name]
            self.ax.plot(sub["asof_date"], sub["value"], label=param_name)
        self.ax.set_xlabel("Date")
        self.ax.set_ylabel("Parameter value")
        self.ax.set_title(f"{ticker} {model.upper()} parameter trends")
        self.ax.legend(loc="best", fontsize=8)
        self.ax.tick_params(axis="x", rotation=45)
        self.canvas.draw()
        # Update the table with the latest values for each parameter
        self.table.update(df)


class ModelParamsApp(tk.Tk):
    """Standalone application wrapper for :class:`ModelParamsFrame`."""

    def __init__(self) -> None:
        super().__init__()
        self.title("Model Parameters Viewer")
        self.geometry("900x700")
        panel = ModelParamsFrame(self)
        panel.pack(fill=tk.BOTH, expand=True)


def launch_model_params(parent: Optional[tk.Misc] = None) -> tk.Tk | tk.Toplevel:
    """Launch the model parameters viewer window.

    If called with no parent, returns a new top‑level application window.  If
    ``parent`` is supplied, creates a ``tk.Toplevel`` anchored to the parent.
    """
    if parent is None:
        return ModelParamsApp()
    else:
        window = tk.Toplevel(parent)
        window.title("Model Parameters Viewer")
        window.geometry("900x700")
        panel = ModelParamsFrame(window)
        panel.pack(fill=tk.BOTH, expand=True)
        return window


if __name__ == "__main__":
    app = launch_model_params()
    app.mainloop()
