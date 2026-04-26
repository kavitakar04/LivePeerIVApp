# This file is based on the upstream IVCorrelation project but has been
# modified to improve GUI responsiveness. The changes revolve around
# running potentially long‑running operations (database queries and
# ingestion) in background threads and then marshaling UI updates back
# to the Tkinter main thread via `after()`. These modifications help
# prevent the UI from freezing while data is downloaded or dates are
# fetched.

from __future__ import annotations
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sys
from pathlib import Path
import argparse

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.data_availability_service import available_tickers, available_dates, ingest_and_process
from data.historical_saver import get_last_coverage_report
from display.gui.gui_input import InputPanel
from display.gui.gui_plot_manager import PlotManager, plot_id
from display.gui.spillover_gui import SpilloverFrame
from display.gui.parameters_tab import ParametersTab, SystemHealthTab
from display.gui.rv_signals_tab import RVSignalsFrame


BROWSER_TAB_LABELS = (
    "IV Explorer",
    "Settings / Data & Model Health",
    "Parameter Summary",
    "Spillover",
    "RV Signals",
)


class BrowserApp(tk.Tk):
    def __init__(self, *, overlay_synth: bool = True, overlay_peers: bool = False,
                 ci_percent: float = 68.0):
        super().__init__()
        self.title("Implied Volatility Browser")
        self.geometry("1200x820")
        self.minsize(800, 600)

        # Notebook with tabs
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # ---- Main exploration tab ----
        self.tab_browser = ttk.Frame(self.notebook)
        # Clarify purpose: this tab lets users explore IV surfaces
        self.notebook.add(self.tab_browser, text=BROWSER_TAB_LABELS[0])

        # ---- Settings / system health tab ----
        self.tab_status = ttk.Frame(self.notebook)
        self.settings_controls = ttk.Frame(self.tab_status)
        self.settings_controls.pack(side=tk.TOP, fill=tk.X)

        # Inputs
        self.inputs = InputPanel(self.tab_browser, overlay_synth=overlay_synth,
                                 overlay_peers=overlay_peers,
                                 ci_percent=ci_percent,
                                 settings_parent=self.settings_controls)
        # Bind events
        self.inputs.bind_download(self._on_download)
        self.inputs.bind_plot(self._refresh_plot)
        self.inputs.bind_target_change(self._on_target_change)

        # Expiry navigation controls
        nav = ttk.Frame(self.tab_browser)
        nav.pack(side=tk.TOP, fill=tk.X, pady=(0, 4))

        self.btn_prev = ttk.Button(nav, text="Prev Expiry", command=self._prev_expiry)
        self.btn_prev.pack(side=tk.LEFT, padx=4)
        self.btn_next = ttk.Button(nav, text="Next Expiry", command=self._next_expiry)
        self.btn_next.pack(side=tk.LEFT, padx=4)

        # Description bar — plain-English context for the current plot
        self.lbl_desc = ttk.Label(
            self.tab_browser, text="", anchor="w", foreground="gray40",
            font=("TkDefaultFont", 9), wraplength=1100, justify=tk.LEFT,
        )
        self.lbl_desc.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(0, 2))

        # Canvas
        self.fig = plt.Figure(figsize=(11.2, 6.4))
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.tab_browser)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.plot_mgr = PlotManager()
        self.plot_mgr.attach_canvas(self.canvas)

        # ---- System integrity dashboard ----
        self.tab_health = SystemHealthTab(
            self.tab_status,
            on_open_expiry=self._open_health_expiry,
            on_open_signals=self._open_health_signals,
        )
        self.tab_health.pack(fill=tk.BOTH, expand=True)
        self.notebook.add(self.tab_status, text=BROWSER_TAB_LABELS[1])

        # ---- Parameter summary tab ----
        self.tab_params = ParametersTab(self.notebook)
        self.notebook.add(self.tab_params, text=BROWSER_TAB_LABELS[2])

        # ---- Spillover tab ----
        self.tab_spillover = SpilloverFrame(self.notebook, input_panel=self.inputs)
        self.notebook.add(self.tab_spillover, text=BROWSER_TAB_LABELS[3])

        # ---- RV Signals tab ----
        self.tab_rv_signals = RVSignalsFrame(self.notebook, input_panel=self.inputs)
        self.notebook.add(self.tab_rv_signals, text=BROWSER_TAB_LABELS[4])

        # Status bar for user feedback
        self.status = ttk.Label(self, text="Ready", anchor="w")
        self.status.pack(side=tk.BOTTOM, fill=tk.X)
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        # Default target suggestion
        tickers = self._load_tickers()
        if tickers and not self.inputs.get_target():
            self.inputs.ent_target.insert(0, tickers[0])
            # Perform initial date load asynchronously
            self._on_target_change()

        self._update_nav_buttons()


    # ---------- events ----------
    def _on_target_change(self, *_):
        """
        Handle changes to the target ticker. To avoid thread‑affinity issues
        with SQLite, this method spawns a worker thread that opens a fresh
        database connection and queries the available dates for the current
        ticker. The results are marshalled back to the Tkinter main thread
        using ``after()`` to safely update the UI without blocking the event
        loop.
        """
        t = self.inputs.get_target()
        if not t:
            return

        # Indicate loading in status bar
        self.status.config(text="Loading available dates...")

        def worker():
            from data.db_utils import get_conn
            import pandas as pd
            dates: list[str] = []
            conn = None
            try:
                conn = get_conn()
                if t:
                    # Query available dates for a specific ticker
                    df = pd.read_sql_query(
                        "SELECT DISTINCT asof_date FROM options_quotes WHERE ticker = ? ORDER BY 1",
                        conn,
                        params=[t],
                    )
                else:
                    # Query all available dates
                    df = pd.read_sql_query(
                        "SELECT DISTINCT asof_date FROM options_quotes ORDER BY 1",
                        conn,
                    )
                dates = df["asof_date"].tolist()
            except Exception:
                dates = []
            finally:
                if conn is not None:
                    try:
                        conn.close()
                    except Exception:
                        pass

            # Schedule UI update on main thread
            def update_ui():
                self.inputs.set_dates(dates)
                self.status.config(text="Ready")
                if hasattr(self, "tab_spillover"):
                    self.tab_spillover.on_browser_selection_changed()
                if hasattr(self, "tab_rv_signals"):
                    self.tab_rv_signals.on_browser_selection_changed()

            self.after(0, update_ui)

        threading.Thread(target=worker, daemon=True).start()

    def _on_download(self):
        """
        Trigger ingestion of data. This can take a long time due to
        network/database operations. Run ingestion in a background thread
        and marshal UI updates to the main thread when complete. Also
        disable the download button while work is in progress to prevent
        multiple concurrent ingestions.
        """
        target = self.inputs.get_target()
        peers = self.inputs.get_peers()
        universe = [x for x in [target] + peers if x]
        if not universe:
            messagebox.showerror("No tickers", "Enter a target and/or peers first.")
            self.status.config(text="No tickers specified")
            return
        max_exp = self.inputs.get_max_exp()
        underlying_lookback_days = self.inputs.get_underlying_lookback_days()
        r, q = self.inputs.get_rates()

        # Provide immediate feedback and disable download button
        self.status.config(text="Downloading data...")
        self.inputs.btn_download.config(state=tk.DISABLED)

        def worker():
            try:
                inserted = ingest_and_process(
                    universe,
                    max_expiries=max_exp,
                    r=r,
                    q=q,
                    underlying_lookback_days=underlying_lookback_days,
                )
                coverage_report = get_last_coverage_report()
                # On success, schedule UI updates
                def done():
                    lines = [
                        f"Ingested rows: {inserted}",
                        f"Tickers: {', '.join(universe)}",
                    ]
                    if coverage_report:
                        lines.append("")
                        lines.append("Coverage report:")
                        lines.append("ticker | provider | requested | fetched | stored | missing shared")
                        for row in coverage_report:
                            lines.append(
                                f"{row.get('ticker')} | "
                                f"{len(row.get('provider_expiries') or [])} | "
                                f"{len(row.get('requested_expiries') or [])} | "
                                f"{len(row.get('fetched_expiries') or [])} | "
                                f"{len(row.get('stored_expiries') or [])} | "
                                f"{', '.join(row.get('missing_shared_expiries') or []) or '-'}"
                            )
                    messagebox.showinfo("Download complete", "\n".join(lines))
                    self.status.config(text=f"Downloaded data for {', '.join(universe)}")
                    # Refresh available dates now that new data may be present
                    self._on_target_change()
                    self.inputs.btn_download.config(state=tk.NORMAL)
                self.after(0, done)
            except Exception as e:
                def handle(exc=e):
                    messagebox.showerror("Download error", str(exc))
                    self.status.config(text="Download failed")
                    self.inputs.btn_download.config(state=tk.NORMAL)
                self.after(0, handle)

        threading.Thread(target=worker, daemon=True).start()

    def _refresh_plot(self):
        settings = self.inputs.get_settings()
        if not settings.get("target") or not settings.get("asof"):
            self.status.config(text="Enter target and date to plot")
            return

        self.status.config(text="Loading...")

        def worker():
            try:
                # Computation (DB queries, weight calc) stays on the worker thread.
                self.plot_mgr.plot(self.ax, settings)
                self._attach_selected_feature_health(settings)
                # canvas.draw() and all Tk widget mutations go back to the main thread.
                def finish():
                    self.canvas.draw()
                    self.status.config(text="Ready")
                    self._update_nav_buttons()
                    self._sync_plot_status()
                self.after(0, finish)

            except Exception as e:
                def handle_err(exc=e):
                    messagebox.showerror("Plot error", str(exc))
                    self.status.config(text="Plot failed")
                    self._update_nav_buttons()
                self.after(0, handle_err)

        threading.Thread(target=worker, daemon=True).start()

    # ---------- helpers ----------
    def _prev_expiry(self):
        self.plot_mgr.prev_expiry()
        self.canvas.draw()
        self._sync_plot_status()

    def _next_expiry(self):
        self.plot_mgr.next_expiry()
        self.canvas.draw()
        self._sync_plot_status()

    def _sync_plot_status(self):
        self.lbl_desc.config(text=self.plot_mgr.last_description)
        self.tab_params.update(self.plot_mgr.last_fit_info)
        self.tab_health.update(self.plot_mgr.last_fit_info)

    def _on_tab_changed(self, _event=None):
        try:
            if self.notebook.select() != str(self.tab_status):
                return
            settings = self.inputs.get_settings()
        except Exception:
            return

        def worker():
            info = dict(self.plot_mgr.last_fit_info or {})
            if not info:
                info = {
                    "ticker": settings.get("target", ""),
                    "asof": settings.get("asof", ""),
                    "fit_by_expiry": {},
                }
            feature_health = self._compute_selected_feature_health(settings)
            if feature_health:
                info["feature_health"] = feature_health
            self.after(0, lambda i=info: self.tab_health.update(i))

        threading.Thread(target=worker, daemon=True).start()

    def _compute_selected_feature_health(self, settings: dict):
        from analysis.feature_health import build_feature_construction_result

        target = (settings.get("target") or "").upper()
        peers = [str(p).upper() for p in settings.get("peers") or [] if str(p).strip()]
        asof = settings.get("asof")
        if not target or not peers or not asof:
            return None
        weight_method = settings.get("weight_method", "corr")
        feature_mode = settings.get("feature_mode", "iv_atm")
        weight_mode = "oi" if weight_method == "oi" else f"{weight_method}_{feature_mode}"
        result = build_feature_construction_result(
            target=target,
            peers=peers,
            asof=asof,
            weight_mode=weight_mode,
            atm_band=settings.get("atm_band"),
            max_expiries=settings.get("max_expiries"),
        )
        return result.feature_health

    def _attach_selected_feature_health(self, settings: dict):
        try:
            feature_health = self._compute_selected_feature_health(settings)
            if not feature_health:
                return
            if not isinstance(self.plot_mgr.last_fit_info, dict):
                target = (settings.get("target") or "").upper()
                self.plot_mgr.last_fit_info = {
                    "ticker": target,
                    "asof": settings.get("asof"),
                    "fit_by_expiry": {},
                }
            self.plot_mgr.last_fit_info["feature_health"] = feature_health
        except Exception:
            pass

    def _open_health_expiry(self, row):
        if not row:
            self.notebook.select(self.tab_browser)
            return
        dte = row.get("DTE")
        try:
            days = float(dte)
            self.inputs.set_T_days(days)
        except Exception:
            pass
        try:
            self.inputs.cmb_plot.set("Smile (K/S vs IV)")
            self.inputs._sync_settings()
            self.inputs._refresh_visibility()
        except Exception:
            pass
        self.notebook.select(self.tab_browser)
        self._refresh_plot()

    def _open_health_signals(self, _row=None):
        if hasattr(self, "tab_rv_signals"):
            self.notebook.select(self.tab_rv_signals)

    def _update_nav_buttons(self):
        state = tk.NORMAL if self.plot_mgr.is_smile_active() else tk.DISABLED
        self.btn_prev.config(state=state)
        self.btn_next.config(state=state)

    def _load_tickers(self):
        try:
            return available_tickers()
        except Exception:
            return []


def main():
        parser = argparse.ArgumentParser(description="Vol Browser")
        parser.add_argument(
            "--overlay-synth",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Overlay synthetic curves",
        )
        parser.add_argument("--overlay-peers", action="store_true", help="Overlay peer curves")
        parser.add_argument("--ci", type=float, default=68.0,
                            help="Confidence interval percentage (e.g. 95 for 95%)")
        args = parser.parse_args()
        app = BrowserApp(overlay_synth=args.overlay_synth,
                         overlay_peers=args.overlay_peers,
                         ci_percent=args.ci)
        app.mainloop()


if __name__ == "__main__":
    main()
