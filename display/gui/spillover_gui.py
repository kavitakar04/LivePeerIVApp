import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.spillover.vol_spillover import run_spillover
from analysis.analysis_pipeline import get_daily_iv_for_spillover

try:
    from analysis.spillover.network_graph import build_spillover_digraph, compute_graph_metrics
    _HAVE_NETWORKX = True
except ImportError:
    _HAVE_NETWORKX = False


class SpilloverFrame(ttk.Frame):
    """Spillover analysis panel, integrated with the IV browser's InputPanel."""

    def __init__(self, master, input_panel=None):
        super().__init__(master)
        self.pack(fill=tk.BOTH, expand=True)
        self._input_panel = input_panel

        # ---- Controls ----
        ctrl = ttk.Frame(self)
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        ttk.Label(ctrl, text="Tickers:").grid(row=0, column=0, sticky=tk.W)
        self.ent_tickers = ttk.Entry(ctrl, width=40)
        self.ent_tickers.grid(row=0, column=1, sticky=tk.W, padx=(0, 4))

        if input_panel is not None:
            ttk.Button(ctrl, text="Sync from Browser", command=self._sync_from_browser).grid(
                row=0, column=2, padx=4
            )

        ttk.Label(ctrl, text="Lookback:").grid(row=1, column=0, sticky=tk.W)
        self.ent_lookback = ttk.Entry(ctrl, width=5)
        self.ent_lookback.insert(0, "60")
        self.ent_lookback.grid(row=1, column=1, sticky=tk.W)

        ttk.Label(ctrl, text="Threshold (%):").grid(row=0, column=3, sticky=tk.W, padx=(8, 0))
        self.ent_threshold = ttk.Entry(ctrl, width=5)
        self.ent_threshold.insert(0, "10")
        self.ent_threshold.grid(row=0, column=4, sticky=tk.W)

        ttk.Label(ctrl, text="Horizons:").grid(row=1, column=3, sticky=tk.W, padx=(8, 0))
        self.ent_horizons = ttk.Entry(ctrl, width=10)
        self.ent_horizons.insert(0, "1,3,5")
        self.ent_horizons.grid(row=1, column=4, sticky=tk.W)

        btn = ttk.Button(ctrl, text="Run", command=self._run_async)
        btn.grid(row=0, column=5, rowspan=2, padx=8)

        self._status_var = tk.StringVar(value="")
        ttk.Label(ctrl, textvariable=self._status_var, foreground="gray").grid(
            row=0, column=6, columnspan=2, sticky=tk.W
        )

        # ---- Paned area: tables top, plot bottom ----
        pane = ttk.PanedWindow(self, orient=tk.VERTICAL)
        pane.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        tables_frame = ttk.Frame(pane)
        pane.add(tables_frame, weight=1)

        plot_frame = ttk.Frame(pane)
        pane.add(plot_frame, weight=1)

        # Event table
        ev_lf = ttk.LabelFrame(tables_frame, text="Recent IV Events (top 20)")
        ev_lf.pack(side=tk.TOP, fill=tk.X, pady=(0, 4))
        self.tree = ttk.Treeview(
            ev_lf, columns=("date", "ticker", "chg"), show="headings", height=5
        )
        self.tree.heading("date", text="Date")
        self.tree.heading("ticker", text="Ticker")
        self.tree.heading("chg", text="Rel Change")
        self.tree.column("date", width=100)
        self.tree.column("ticker", width=80)
        self.tree.column("chg", width=90)
        ev_sb = ttk.Scrollbar(ev_lf, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=ev_sb.set)
        self.tree.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ev_sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.bind("<<TreeviewSelect>>", self._on_event_select)
        self._event_rows: dict[str, pd.Series] = {}

        # Summary table
        sum_cols = ("ticker", "peer", "h", "hit", "sign", "resp", "elast", "n")
        sum_lf = ttk.LabelFrame(tables_frame, text="Spillover Summary (top 50 by hit rate)")
        sum_lf.pack(side=tk.TOP, fill=tk.X, pady=(0, 4))
        self.tree_sum = ttk.Treeview(sum_lf, columns=sum_cols, show="headings", height=6)
        headings = {
            "ticker": "Trigger", "peer": "Peer", "h": "H",
            "hit": "Hit %", "sign": "Sign %", "resp": "Med Resp",
            "elast": "Med Elast", "n": "N",
        }
        widths = {"ticker": 70, "peer": 70, "h": 40, "hit": 60,
                  "sign": 60, "resp": 80, "elast": 80, "n": 50}
        for col in sum_cols:
            self.tree_sum.heading(col, text=headings[col])
            self.tree_sum.column(col, width=widths[col])
        sum_sb = ttk.Scrollbar(sum_lf, orient=tk.VERTICAL, command=self.tree_sum.yview)
        self.tree_sum.configure(yscrollcommand=sum_sb.set)
        self.tree_sum.pack(side=tk.LEFT, fill=tk.X, expand=True)
        sum_sb.pack(side=tk.RIGHT, fill=tk.Y)

        # Graph centrality table (only shown when networkx is available)
        if _HAVE_NETWORKX:
            gm_lf = ttk.LabelFrame(tables_frame, text="Network Centrality (horizon 1)")
            gm_lf.pack(side=tk.TOP, fill=tk.X)
            gm_cols = ("node", "out_strength", "in_strength", "betweenness", "degree")
            self.tree_graph = ttk.Treeview(gm_lf, columns=gm_cols, show="headings", height=4)
            gm_headings = {
                "node": "Ticker", "out_strength": "Out-Strength",
                "in_strength": "In-Strength", "betweenness": "Betweenness",
                "degree": "Degree",
            }
            gm_widths = {"node": 70, "out_strength": 100, "in_strength": 100,
                         "betweenness": 100, "degree": 60}
            for col in gm_cols:
                self.tree_graph.heading(col, text=gm_headings[col])
                self.tree_graph.column(col, width=gm_widths[col])
            self.tree_graph.pack(fill=tk.X, expand=True)
        else:
            self.tree_graph = None

        # Plot area
        self.fig = plt.Figure(figsize=(6, 4))
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.results = None

        # Auto-populate tickers if browser is attached
        if input_panel is not None:
            self._sync_from_browser()

    # ---- Sync from browser ----

    def _sync_from_browser(self):
        if self._input_panel is None:
            return
        tgt = self._input_panel.get_target()
        peers = self._input_panel.get_peers()
        all_tickers = [t for t in [tgt] + peers if t]
        self.ent_tickers.delete(0, tk.END)
        self.ent_tickers.insert(0, ",".join(all_tickers))

    def on_browser_selection_changed(self):
        """Called by BrowserApp when the target or peers change."""
        self._sync_from_browser()

    # ---- Run ----

    def _run_async(self):
        tickers_raw = self.ent_tickers.get()
        tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]
        if not tickers:
            messagebox.showerror("No tickers", "Enter at least one ticker")
            return
        try:
            thr = float(self.ent_threshold.get()) / 100.0
            lookback = int(self.ent_lookback.get())
            horizons = [int(h) for h in self.ent_horizons.get().split(",") if h.strip()]
        except ValueError:
            messagebox.showerror("Input error", "Invalid numeric input")
            return

        self._status_var.set("Loading IV data…")

        def worker():
            try:
                df = get_daily_iv_for_spillover(tickers)
                if df.empty:
                    self.after(0, lambda: (
                        self._status_var.set(""),
                        messagebox.showerror(
                            "No data",
                            "No ATM IV data found for the selected tickers.\n"
                            "Download data first via the Parameter Explorer tab.",
                        ),
                    ))
                    return

                results = run_spillover(
                    df,
                    tickers=tickers,
                    threshold=thr,
                    lookback=lookback,
                    horizons=horizons,
                    events_path=str(ROOT / "data" / "spill_events.parquet"),
                    summary_path=str(ROOT / "data" / "spill_summary.parquet"),
                )

                def update():
                    self.results = results
                    self._status_var.set(
                        f"{len(results['events'])} events · "
                        f"{len(results['summary'])} summary rows"
                    )
                    self._populate_events()
                    self._populate_summary()
                    self._populate_graph_metrics()
                    self._plot_response()

                self.after(0, update)

            except Exception as exc:
                self.after(0, lambda: (
                    self._status_var.set(""),
                    messagebox.showerror("Spillover error", str(exc)),
                ))

        threading.Thread(target=worker, daemon=True).start()

    # ---- Populate tables ----

    def _populate_events(self):
        self._event_rows.clear()
        for i in self.tree.get_children():
            self.tree.delete(i)
        events = self.results["events"].sort_values("date", ascending=False).head(20)
        for _, row in events.iterrows():
            iid = self.tree.insert(
                "", tk.END,
                values=(row["date"].date(), row["ticker"], f"{row['rel_change']:.2%}"),
            )
            self._event_rows[iid] = row

    def _populate_summary(self):
        for i in self.tree_sum.get_children():
            self.tree_sum.delete(i)
        summary = self.results["summary"].copy()
        if summary.empty:
            return
        summary = summary.sort_values("hit_rate", ascending=False).head(50)
        for _, row in summary.iterrows():
            self.tree_sum.insert(
                "", tk.END,
                values=(
                    row["ticker"], row["peer"], row["h"],
                    f"{row['hit_rate']:.0%}", f"{row['sign_concord']:.0%}",
                    f"{row['median_resp']:.2%}", f"{row['median_elasticity']:.2f}",
                    int(row["n"]),
                ),
            )

    def _populate_graph_metrics(self):
        if not _HAVE_NETWORKX or self.tree_graph is None:
            return
        for i in self.tree_graph.get_children():
            self.tree_graph.delete(i)
        summary = self.results["summary"]
        if summary.empty:
            return
        try:
            G = build_spillover_digraph(summary, horizon=1, min_n=3)
            if len(G) == 0:
                return
            metrics = compute_graph_metrics(G).sort_values("out_strength", ascending=False)
            for _, row in metrics.iterrows():
                self.tree_graph.insert(
                    "", tk.END,
                    values=(
                        row["node"],
                        f"{row['out_strength']:.3f}",
                        f"{row['in_strength']:.3f}",
                        f"{row['betweenness_centrality']:.3f}",
                        int(row["degree"]),
                    ),
                )
        except Exception:
            pass  # graph metrics are best-effort

    # ---- Plots ----

    def _plot_response(self):
        self.ax.clear()
        summary = self.results["summary"]
        if summary.empty:
            self.canvas.draw()
            return
        grp = summary.groupby("h")["median_resp"].mean()
        self.ax.plot(grp.index, grp.values, marker="o", label="Median resp (avg across pairs)")
        self.ax.set_xlabel("Horizon (days)")
        self.ax.set_ylabel("Peer IV change")
        self.ax.set_title("Average spillover response by horizon")
        self.ax.axhline(0, color="black", linewidth=0.5)
        self.ax.legend()
        self.fig.tight_layout()
        self.canvas.draw()

    def _on_event_select(self, _):
        sel = self.tree.selection()
        if not sel:
            return
        row = self._event_rows.get(sel[0])
        if row is not None:
            self._plot_event_response(row["ticker"], row["date"])

    def _plot_event_response(self, ticker: str, date):
        self.ax.clear()
        df = self.results["responses"]
        mask = (df["ticker"] == ticker) & (df["t0"] == date)
        subset = df.loc[mask]
        if subset.empty:
            self.canvas.draw()
            return
        for peer, grp in subset.groupby("peer"):
            self.ax.plot(grp["h"], grp["peer_pct"], marker="o", label=peer)
        self.ax.set_xlabel("Horizon (days)")
        self.ax.set_ylabel("Peer IV change")
        self.ax.set_title(f"{ticker} event on {pd.Timestamp(date).date()} — peer responses")
        self.ax.axhline(0, color="black", linewidth=0.5)
        self.ax.legend()
        self.fig.tight_layout()
        self.canvas.draw()


class SpilloverApp(tk.Tk):
    """Standalone application wrapper for :class:`SpilloverFrame`."""

    def __init__(self):
        super().__init__()
        self.title("IV Spillover Explorer")
        self.geometry("900x700")
        SpilloverFrame(self)


def launch_spillover(parent=None, input_panel=None):
    """Launch the spillover analysis window.

    Parameters
    ----------
    parent : tk.Widget, optional
        Parent window.  If None, creates a standalone app.
    input_panel : InputPanel, optional
        Browser's InputPanel for ticker/peer synchronisation.
    """
    if parent is None:
        return SpilloverApp()
    window = tk.Toplevel(parent)
    window.title("IV Spillover Explorer")
    window.geometry("900x700")
    SpilloverFrame(window, input_panel=input_panel)
    return window


SpilloverPanel = SpilloverFrame  # backward compat alias


if __name__ == "__main__":
    launch_spillover().mainloop()
