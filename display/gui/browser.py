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

from analysis.analysis_pipeline import available_tickers, available_dates, ingest_and_process
from display.gui.gui_input import InputPanel
from display.gui.gui_plot_manager import PlotManager, plot_id
from display.gui.spillover_gui import SpilloverFrame
from display.gui.parameters_tab import ParametersTab


class BrowserApp(tk.Tk):
    def __init__(self, *, overlay_synth: bool = True, overlay_peers: bool = True,
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
        self.notebook.add(self.tab_browser, text="IV Explorer")

        # Inputs
        self.inputs = InputPanel(self.tab_browser, overlay_synth=overlay_synth,
                                 overlay_peers=overlay_peers,
                                 ci_percent=ci_percent)
        # Bind events
        self.inputs.bind_download(self._on_download)
        self.inputs.bind_plot(self._refresh_plot)
        self.inputs.bind_target_change(self._on_target_change)

        # Expiry navigation and animation controls
        nav = ttk.Frame(self.tab_browser)
        nav.pack(side=tk.TOP, fill=tk.X, pady=(0, 4))

        # Expiry navigation (existing)
        self.btn_prev = ttk.Button(nav, text="Prev Expiry", command=self._prev_expiry)
        self.btn_prev.pack(side=tk.LEFT, padx=4)
        self.btn_next = ttk.Button(nav, text="Next Expiry", command=self._next_expiry)
        self.btn_next.pack(side=tk.LEFT, padx=4)

        # Animation controls (new)
        ttk.Separator(nav, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        self.var_animated = tk.BooleanVar(value=False)
        self.chk_animated = ttk.Checkbutton(nav, text="Animate", variable=self.var_animated,
                                            command=self._toggle_animation_mode)
        self.chk_animated.pack(side=tk.LEFT, padx=4)

        self.btn_play_pause = ttk.Button(nav, text="Play", command=self._toggle_animation)
        self.btn_play_pause.pack(side=tk.LEFT, padx=2)

        self.btn_stop = ttk.Button(nav, text="Stop", command=self._stop_animation)
        self.btn_stop.pack(side=tk.LEFT, padx=2)

        ttk.Label(nav, text="Speed:").pack(side=tk.LEFT, padx=(8, 2))
        self.speed_var = tk.IntVar(value=500)  # Default speed
        self.speed_scale = ttk.Scale(nav, from_=100, to=2000, variable=self.speed_var,
                                     orient=tk.HORIZONTAL, length=100,
                                     command=self._on_speed_change)
        self.speed_scale.pack(side=tk.LEFT, padx=2)


        # Canvas
        self.fig = plt.Figure(figsize=(11.2, 6.6))
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.tab_browser)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.plot_mgr = PlotManager()
        self.plot_mgr.attach_canvas(self.canvas)

        # ---- Parameter summary tab ----
        self.tab_params = ParametersTab(self.notebook)
        self.notebook.add(self.tab_params, text="Parameter Summary")

        # ---- Spillover tab ----
        self.tab_spillover = SpilloverFrame(self.notebook, input_panel=self.inputs)
        self.notebook.add(self.tab_spillover, text="Spillover")

        # Status bar for user feedback
        self.status = ttk.Label(self, text="Ready", anchor="w")
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

        # Default target suggestion
        tickers = self._load_tickers()
        if tickers and not self.inputs.get_target():
            self.inputs.ent_target.insert(0, tickers[0])
            # Perform initial date load asynchronously
            self._on_target_change()

        self._update_nav_buttons()
        self._update_animation_buttons()


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
        r, q = self.inputs.get_rates()

        # Provide immediate feedback and disable download button
        self.status.config(text="Downloading data...")
        self.inputs.btn_download.config(state=tk.DISABLED)

        def worker():
            try:
                inserted = ingest_and_process(universe, max_expiries=max_exp, r=r, q=q)
                # On success, schedule UI updates
                def done():
                    messagebox.showinfo("Download complete", f"Ingested rows: {inserted}\nTickers: {', '.join(universe)}")
                    self.status.config(text=f"Downloaded data for {', '.join(universe)}")
                    # Refresh available dates now that new data may be present
                    self._on_target_change()
                    self.inputs.btn_download.config(state=tk.NORMAL)
                self.after(0, done)
            except Exception as e:
                def handle():
                    messagebox.showerror("Download error", str(e))
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
                want_anim = (
                    self.var_animated.get()
                    and self.plot_mgr.has_animation_support(plot_id(settings.get("plot_type", "")))
                )

                if want_anim:
                    # FuncAnimation registers a timer with Tk — must run on main thread.
                    def render_animated():
                        try:
                            if self.plot_mgr.plot_animated(self.ax, settings):
                                self.status.config(text="Animated plot created")
                            else:
                                self.plot_mgr.plot(self.ax, settings)
                                self.status.config(text="Animation failed — using static plot")
                            self.canvas.draw()
                            self._update_nav_buttons()
                            self._update_animation_buttons()
                            self.tab_params.update(self.plot_mgr.last_fit_info)
                        except Exception as exc:
                            messagebox.showerror("Plot error", str(exc))
                            self.status.config(text="Plot failed")
                            self._update_nav_buttons()
                            self._update_animation_buttons()
                    self.after(0, render_animated)
                else:
                    # Computation (DB queries, weight calc) stays on the worker thread.
                    self.plot_mgr.stop_animation()
                    self.plot_mgr.plot(self.ax, settings)
                    # canvas.draw() and all Tk widget mutations go back to the main thread.
                    def finish():
                        self.canvas.draw()
                        self.status.config(text="Ready")
                        self._update_nav_buttons()
                        self._update_animation_buttons()
                        self.tab_params.update(self.plot_mgr.last_fit_info)
                    self.after(0, finish)

            except Exception as e:
                def handle_err(exc=e):
                    messagebox.showerror("Plot error", str(exc))
                    self.status.config(text="Plot failed")
                    self._update_nav_buttons()
                    self._update_animation_buttons()
                self.after(0, handle_err)

        threading.Thread(target=worker, daemon=True).start()

    # ---------- helpers ----------
    def _prev_expiry(self):
        self.plot_mgr.prev_expiry()
        self.canvas.draw()

    def _next_expiry(self):
        self.plot_mgr.next_expiry()
        self.canvas.draw()

    def _update_nav_buttons(self):
        state = tk.NORMAL if self.plot_mgr.is_smile_active() else tk.DISABLED
        self.btn_prev.config(state=state)
        self.btn_next.config(state=state)

    def _update_animation_buttons(self):
        """Update animation control button states."""
        plot_type = self.inputs.get_plot_type()
        has_anim_support = self.plot_mgr.has_animation_support(plot_id(plot_type))
        is_animated = self.var_animated.get()
        is_anim_active = self.plot_mgr.is_animation_active()

        # Enable/disable animation checkbox based on plot type support
        anim_state = tk.NORMAL if has_anim_support else tk.DISABLED
        self.chk_animated.config(state=anim_state)

        # Enable/disable animation controls based on animation state
        control_state = tk.NORMAL if (is_animated and is_anim_active) else tk.DISABLED
        self.btn_play_pause.config(state=control_state)
        self.btn_stop.config(state=control_state)
        self.speed_scale.config(state=control_state)

        # Update play/pause button text
        if is_anim_active and not self.plot_mgr._animation_paused:
            self.btn_play_pause.config(text="Pause")
        else:
            self.btn_play_pause.config(text="Play")

    def _toggle_animation_mode(self):
        """Handle animation checkbox toggle."""
        # Refresh plot when animation mode changes
        self._refresh_plot()

    def _toggle_animation(self):
        """Toggle animation play/pause."""
        if self.plot_mgr.is_animation_active():
            if self.plot_mgr._animation_paused:
                self.plot_mgr.start_animation()
            else:
                self.plot_mgr.pause_animation()
            self._update_animation_buttons()

    def _stop_animation(self):
        """Stop animation."""
        self.plot_mgr.stop_animation()
        self._update_animation_buttons()

    def _on_speed_change(self, value):
        """Handle animation speed change."""
        try:
            speed_ms = int(2100 - float(value))  # Invert scale (higher value = faster)
            self.plot_mgr.set_animation_speed(speed_ms)
        except Exception:
            pass

    def _load_tickers(self):
        try:
            return available_tickers()
        except Exception:
            return []


def main():
        parser = argparse.ArgumentParser(description="Vol Browser")
        parser.add_argument("--overlay-synth", action="store_true", help="Overlay synthetic curves")
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
