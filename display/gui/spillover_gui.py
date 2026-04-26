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
from analysis.data_availability_service import get_daily_iv_for_spillover, get_daily_hv_for_spillover

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

        desc = ttk.Label(
            self,
            text=(
                "Volatility spillover / propagation across peers for RV context. "
                "A trigger is a large daily IV/HV move in one ticker. "
                "Frequency of response is the share of trigger events where a peer also moves beyond the threshold; "
                "same-direction probability is the share moving in the trigger direction; "
                "median IV change is the typical peer response over the horizon."
            ),
            wraplength=980,
            justify=tk.LEFT,
        )
        desc.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(6, 2))

        hint = ttk.Label(
            self,
            text=(
                "Interpretation hint: strongest relationships combine high response frequency, "
                "high same-direction probability, a large abnormal response, and low p/q values."
            ),
            foreground="gray",
            wraplength=980,
            justify=tk.LEFT,
        )
        hint.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0, 4))

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

        ttk.Label(ctrl, text="Mode:").grid(row=0, column=5, sticky=tk.W, padx=(8, 0))
        self._mode_var = tk.StringVar(value="HV")
        mode_cb = ttk.Combobox(ctrl, textvariable=self._mode_var,
                               values=["IV", "HV"], state="readonly", width=4)
        mode_cb.grid(row=0, column=6, sticky=tk.W)
        mode_cb.bind("<<ComboboxSelected>>", self._on_mode_change)

        ttk.Label(ctrl, text="HV window:").grid(row=1, column=5, sticky=tk.W, padx=(8, 0))
        self.ent_hv_window = ttk.Entry(ctrl, width=5)
        self.ent_hv_window.insert(0, "20")
        self.ent_hv_window.grid(row=1, column=6, sticky=tk.W)

        btn = ttk.Button(ctrl, text="Run", command=self._run_async)
        btn.grid(row=0, column=7, rowspan=2, padx=8)

        self._status_var = tk.StringVar(value="")
        ttk.Label(ctrl, textvariable=self._status_var, foreground="gray").grid(
            row=0, column=8, columnspan=2, sticky=tk.W
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
        sum_cols = ("ticker", "peer", "h", "hit", "sign", "resp", "abn", "ci", "p", "q", "strength", "n")
        sum_lf = ttk.LabelFrame(tables_frame, text="Spillover Summary (top 50 by hit rate)")
        sum_lf.pack(side=tk.TOP, fill=tk.X, pady=(0, 4))
        self.tree_sum = ttk.Treeview(sum_lf, columns=sum_cols, show="headings", height=6)
        headings = {
            "ticker": "Trigger", "peer": "Peer", "h": "H",
            "hit": "Frequency of response", "sign": "Same-direction probability",
            "resp": "Median IV change (%)", "abn": "Abnormal IV change (%)",
            "ci": "95% CI", "p": "Perm p", "q": "FDR q",
            "strength": "Strength", "n": "N",
        }
        widths = {"ticker": 70, "peer": 70, "h": 40, "hit": 150,
                  "sign": 180, "resp": 150, "abn": 150, "ci": 140,
                  "p": 70, "q": 70, "strength": 100, "n": 50}
        for col in sum_cols:
            self.tree_sum.heading(col, text=headings[col])
            self.tree_sum.column(col, width=widths[col])
        self.tree_sum.tag_configure("strong", background="#fff2cc")
        sum_sb = ttk.Scrollbar(sum_lf, orient=tk.VERTICAL, command=self.tree_sum.yview)
        self.tree_sum.configure(yscrollcommand=sum_sb.set)
        self.tree_sum.pack(side=tk.LEFT, fill=tk.X, expand=True)
        sum_sb.pack(side=tk.RIGHT, fill=tk.Y)

        # Graph centrality table (only shown when networkx is available)
        if _HAVE_NETWORKX:
            gm_lf = ttk.LabelFrame(tables_frame, text="Network Centrality (horizon 1)")
            gm_lf.pack(side=tk.TOP, fill=tk.X)
            gm_note = ttk.Label(
                gm_lf,
                text="Leader sends stronger spillovers to peers; Follower receives stronger spillovers from peers.",
                foreground="gray",
                wraplength=980,
                justify=tk.LEFT,
            )
            gm_note.pack(side=tk.TOP, fill=tk.X, padx=4, pady=(2, 2))
            gm_cols = ("node", "role", "out_strength", "in_strength", "betweenness", "degree")
            self.tree_graph = ttk.Treeview(gm_lf, columns=gm_cols, show="headings", height=4)
            gm_headings = {
                "node": "Ticker", "role": "Role", "out_strength": "Out-Strength",
                "in_strength": "In-Strength", "betweenness": "Betweenness",
                "degree": "Degree",
            }
            gm_widths = {"node": 70, "role": 80, "out_strength": 100, "in_strength": 100,
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

    def _on_mode_change(self, _=None):
        state = tk.NORMAL if self._mode_var.get() == "HV" else tk.DISABLED
        self.ent_hv_window.configure(state=state)

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
            hv_window = int(self.ent_hv_window.get())
        except ValueError:
            messagebox.showerror("Input error", "Invalid numeric input")
            return

        mode = self._mode_var.get()
        self._status_var.set(f"Loading {mode} data…")

        def worker():
            try:
                if mode == "HV":
                    df = get_daily_hv_for_spillover(tickers, hv_window=hv_window)
                    no_data_msg = (
                        "No underlying price data found for the selected tickers.\n"
                        "Download data first via the Parameter Explorer tab."
                    )
                else:
                    df = get_daily_iv_for_spillover(tickers)
                    no_data_msg = (
                        "No ATM IV data found for the selected tickers.\n"
                        "Download data first via the Parameter Explorer tab."
                    )
                if df.empty:
                    self.after(0, lambda: (
                        self._status_var.set(""),
                        messagebox.showerror("No data", no_data_msg),
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
                self.after(0, lambda e=exc: (
                    self._status_var.set(""),
                    messagebox.showerror("Spillover error", str(e)),
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
        abs_resp_cutoff = summary["median_resp"].abs().quantile(0.75)
        if pd.isna(abs_resp_cutoff):
            abs_resp_cutoff = 0.0
        for _, row in summary.iterrows():
            strength = str(row.get("strength", ""))
            is_strong = strength == "Strong" or (
                float(row["hit_rate"]) >= 0.70
                and float(row["sign_concord"]) >= 0.70
                and abs(float(row["median_resp"])) >= float(abs_resp_cutoff)
            )
            ci_low = row.get("median_resp_ci_low", pd.NA)
            ci_high = row.get("median_resp_ci_high", pd.NA)
            ci_text = (
                f"[{ci_low:.2%}, {ci_high:.2%}]"
                if pd.notna(ci_low) and pd.notna(ci_high)
                else ""
            )
            p_value = row.get("p_value", pd.NA)
            q_value = row.get("q_value", pd.NA)
            self.tree_sum.insert(
                "", tk.END,
                values=(
                    row["ticker"], row["peer"], row["h"],
                    f"{row['hit_rate']:.0%}", f"{row['sign_concord']:.0%}",
                    f"{row['median_resp']:.2%}",
                    f"{row.get('median_abnormal_resp', pd.NA):.2%}" if pd.notna(row.get("median_abnormal_resp", pd.NA)) else "",
                    ci_text,
                    f"{p_value:.3f}" if pd.notna(p_value) else "",
                    f"{q_value:.3f}" if pd.notna(q_value) else "",
                    strength,
                    int(row["n"]),
                ),
                tags=("strong",) if is_strong else (),
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
                out_strength = float(row["out_strength"])
                in_strength = float(row["in_strength"])
                if out_strength > in_strength:
                    role = "Leader"
                elif in_strength > out_strength:
                    role = "Follower"
                else:
                    role = "Balanced"
                self.tree_graph.insert(
                    "", tk.END,
                    values=(
                        row["node"],
                        role,
                        f"{out_strength:.3f}",
                        f"{in_strength:.3f}",
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
