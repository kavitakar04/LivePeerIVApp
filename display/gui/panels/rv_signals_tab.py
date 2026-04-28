"""
RV Signals tab for the main GUI browser.

This tab is the final synthesis layer for relative-value research.  It turns
target-vs-peer IV, model quality, spillover, and surface-comparability context
into separated trade opportunities and market anomalies rather than exposing a
raw spread table.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import tkinter as tk
from tkinter import ttk

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.config.settings import DEFAULT_MAX_EXPIRIES

_TRADE_COLUMNS = (
    "rank",
    "title",
    "judgment",
    "trade_type",
    "direction",
    "target",
    "hedge_or_peer",
    "maturity",
    "confidence",
    "trade_score",
    "substitutability",
    "buy_legs",
    "sell_legs",
    "net_premium",
    "estimated_delta_after_hedge",
    "horizon",
    "rationale",
)

_TRADE_HEADERS = {
    "rank": ("Rank", 54),
    "title": ("Trade Opportunity", 340),
    "judgment": ("Judgment", 104),
    "trade_type": ("Type", 130),
    "direction": ("Direction", 240),
    "target": ("Target", 76),
    "hedge_or_peer": ("Hedge / Peer", 150),
    "maturity": ("Maturity", 82),
    "confidence": ("Confidence", 94),
    "trade_score": ("Score", 76),
    "substitutability": ("Substitutability", 140),
    "buy_legs": ("Buy", 300),
    "sell_legs": ("Sell", 300),
    "net_premium": ("Net Premium", 110),
    "estimated_delta_after_hedge": ("Net Delta", 110),
    "horizon": ("Horizon", 150),
    "rationale": ("Rationale", 520),
}

_ANOMALY_COLUMNS = (
    "rank",
    "title",
    "judgment",
    "anomaly_type",
    "group_size",
    "affected_names",
    "likely_driver",
    "systemic_or_idiosyncratic",
    "trade_score",
    "spillover_relevance",
    "impact_on_trade_confidence",
)

_ANOMALY_HEADERS = {
    "rank": ("Rank", 54),
    "title": ("Market Anomaly", 340),
    "judgment": ("Judgment", 104),
    "anomaly_type": ("Type", 180),
    "group_size": ("Signals", 72),
    "affected_names": ("Affected", 150),
    "likely_driver": ("Likely Driver", 260),
    "systemic_or_idiosyncratic": ("Context", 120),
    "trade_score": ("Score", 76),
    "spillover_relevance": ("Spillover", 260),
    "impact_on_trade_confidence": ("Trade Impact", 360),
}


class RVSignalsFrame(ttk.Frame):
    """Ranked opportunity dashboard for the current target/peer group."""

    def __init__(self, master, input_panel=None, **kwargs):
        super().__init__(master, **kwargs)
        self._input_panel = input_panel
        self._last_opportunities = None
        self._last_trades = None
        self._last_anomalies = None
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

        tables = ttk.PanedWindow(main, orient=tk.VERTICAL)
        main.add(tables, weight=3)

        trade_frame = ttk.LabelFrame(tables, text="Trade Opportunities - What might we do?")
        anomaly_frame = ttk.LabelFrame(tables, text="Market Anomalies - What is the market telling us?")
        tables.add(trade_frame, weight=1)
        tables.add(anomaly_frame, weight=1)

        self.trade_tree = self._build_tree(
            trade_frame,
            _TRADE_COLUMNS,
            _TRADE_HEADERS,
            stretch_cols={"title", "direction", "buy_legs", "sell_legs", "rationale"},
        )
        self.anomaly_tree = self._build_tree(
            anomaly_frame,
            _ANOMALY_COLUMNS,
            _ANOMALY_HEADERS,
            stretch_cols={"title", "likely_driver", "spillover_relevance", "impact_on_trade_confidence"},
        )
        self.tree = self.trade_tree
        self.trade_tree.bind("<<TreeviewSelect>>", self._on_trade_selected)
        self.anomaly_tree.bind("<<TreeviewSelect>>", self._on_anomaly_selected)

        detail = ttk.LabelFrame(main, text="Signal Detail")
        main.add(detail, weight=2)
        self.detail_text = tk.Text(detail, height=10, wrap="word", relief="flat", padx=8, pady=6)
        self.detail_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        detail_vsb = ttk.Scrollbar(detail, orient="vertical", command=self.detail_text.yview)
        self.detail_text.configure(yscrollcommand=detail_vsb.set)
        detail_vsb.pack(side=tk.LEFT, fill=tk.Y)
        self.detail_text.configure(state=tk.DISABLED)

    def _build_tree(self, parent, columns, headers, *, stretch_cols):
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)
        tree = ttk.Treeview(parent, columns=columns, show="headings", selectmode="browse", height=6)
        for col in columns:
            header, width = headers[col]
            tree.heading(col, text=header)
            anchor = "w" if col in stretch_cols else "center"
            tree.column(col, width=width, anchor=anchor, stretch=(col in stretch_cols))
        vsb = ttk.Scrollbar(parent, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(parent, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        tree.tag_configure("rich", background="#ffe8e8")
        tree.tag_configure("cheap", background="#e8f0ff")
        tree.tag_configure("neutral", background="#f5f5f5")
        tree.tag_configure("warning", background="#fff6d9")
        tree.tag_configure("high", font=("", 10, "bold"))
        return tree

    def on_browser_selection_changed(self):
        """Called by BrowserApp when target / peers change."""
        self._last_opportunities = None
        self._last_trades = None
        self._last_anomalies = None
        self._clear_dashboard("Press Refresh to synthesize RV opportunities.")

    def _clear_dashboard(self, status: str):
        for tree in (self.trade_tree, self.anomaly_tree):
            for item in tree.get_children():
                tree.delete(item)
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
            max_expiries = int(settings.get("max_expiries", DEFAULT_MAX_EXPIRIES) or DEFAULT_MAX_EXPIRIES)
        except (TypeError, ValueError):
            max_expiries = DEFAULT_MAX_EXPIRIES

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

        self.lbl_status.config(text="Classifying RV signals...")
        self.btn_refresh.config(state=tk.DISABLED)

        def worker():
            try:
                from analysis.rv.rv_analysis import generate_rv_opportunity_dashboard

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
        import pandas as pd

        opportunities = payload.get("opportunities")
        trades = payload.get("trade_opportunities")
        anomalies = payload.get("market_anomalies")
        if trades is None:
            trades = pd.DataFrame()
        if anomalies is None:
            anomalies = pd.DataFrame()
        self._last_opportunities = opportunities
        self._last_trades = trades
        self._last_anomalies = anomalies
        self._set_summary(payload.get("executive_summary") or [])

        cards = payload.get("context_cards") or {}
        for key, label in self.card_labels.items():
            label.config(text=str(cards.get(key, "Unavailable")))

        for tree in (self.trade_tree, self.anomaly_tree):
            for item in tree.get_children():
                tree.delete(item)

        if trades.empty:
            self.trade_tree.insert(
                "",
                "end",
                values=tuple("" for _ in _TRADE_COLUMNS[:-1]) + ("No signal passed the trade-score threshold.",),
                tags=("neutral",),
            )
        else:
            for idx, row in trades.iterrows():
                values = [self._format_table_value(row.get(col, "")) for col in _TRADE_COLUMNS]
                direction = str(row.get("direction", ""))
                judgment = str(row.get("judgment", ""))
                if judgment == "Conditional":
                    tag = "warning"
                else:
                    tag = (
                        "rich"
                        if direction.startswith("Sell ")
                        else "cheap" if direction.startswith("Buy ") else "neutral"
                    )
                tags = (tag, "high") if str(row.get("confidence")) == "High" else (tag,)
                self.trade_tree.insert("", "end", iid=str(idx), values=values, tags=tags)

        if anomalies.empty:
            self.anomaly_tree.insert(
                "",
                "end",
                values=tuple("" for _ in _ANOMALY_COLUMNS[:-1]) + ("No anomaly classified.",),
                tags=("neutral",),
            )
        else:
            for idx, row in anomalies.iterrows():
                values = [self._format_table_value(row.get(col, "")) for col in _ANOMALY_COLUMNS]
                context = str(row.get("systemic_or_idiosyncratic", ""))
                tag = "warning" if context in {"Systemic", "Cluster", "Unknown"} else "neutral"
                self.anomaly_tree.insert("", "end", iid=str(idx), values=values, tags=(tag,))

        first_trade = self.trade_tree.get_children()
        if first_trade and not trades.empty:
            self.trade_tree.selection_set(first_trade[0])
            self.trade_tree.focus(first_trade[0])
            self._render_detail_for_index(int(first_trade[0]), section="trade")
        elif not anomalies.empty:
            first_anomaly = self.anomaly_tree.get_children()
            self.anomaly_tree.selection_set(first_anomaly[0])
            self.anomaly_tree.focus(first_anomaly[0])
            self._render_detail_for_index(int(first_anomaly[0]), section="anomaly")
        else:
            warnings = payload.get("warnings") or []
            self._set_detail("\n".join(warnings) if warnings else "No selected signal.")

        warning_count = len(payload.get("warnings") or [])
        self.lbl_status.config(
            text=(
                f"{len(trades)} trade opportunity/opportunities; "
                f"{len(anomalies)} market anomaly/anomalies; {warning_count} warning(s)."
            )
        )

    def _format_table_value(self, value):
        import math
        import pandas as pd

        if isinstance(value, list):
            return ", ".join(str(v) for v in value)
        if isinstance(value, dict):
            return str(value)
        try:
            if pd.isna(value):
                return "-"
        except Exception:
            pass
        if isinstance(value, float):
            return f"{value:.3f}" if math.isfinite(value) else "-"
        return str(value) if value != "" else "-"

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

    def _on_trade_selected(self, _event=None):
        selected = self.trade_tree.selection()
        if not selected:
            return
        try:
            self._render_detail_for_index(int(selected[0]), section="trade")
        except ValueError:
            self._set_detail("No selected signal.")

    def _on_anomaly_selected(self, _event=None):
        selected = self.anomaly_tree.selection()
        if not selected:
            return
        try:
            self._render_detail_for_index(int(selected[0]), section="anomaly")
        except ValueError:
            self._set_detail("No selected signal.")

    def _render_detail_for_index(self, idx: int, section: str = "legacy"):
        pass

        if section == "trade":
            if self._last_trades is None or idx not in self._last_trades.index:
                self._set_detail("No selected trade opportunity.")
                return
            row = self._last_trades.loc[idx]
            source = row.get("source_signal", {}) if hasattr(row, "get") else {}
            trade = row.get("trade", {}) if isinstance(row.get("trade", {}), dict) else {}
            exposures = trade.get("exposures", {}) if isinstance(trade.get("exposures", {}), dict) else {}
            hedge_package = trade.get("hedge_package", {}) if isinstance(trade.get("hedge_package", {}), dict) else {}
            risks = row.get("risks", [])
            if isinstance(risks, list):
                risks_text = "\n".join(f"- {r}" for r in risks) if risks else "None"
            else:
                risks_text = str(risks or "None")
            detail = (
                f"Trade Opportunity\n"
                f"{row.get('title', '')}\n\n"
                f"Judgment: {row.get('judgment', '')}\n"
                f"Trade Score: {self._format_table_value(row.get('trade_score', ''))}\n"
                f"Substitutability: {row.get('substitutability', '')}\n"
                f"Trade Type: {row.get('trade_type', '')}\n"
                f"Direction: {row.get('direction', '')}\n"
                f"Target: {row.get('target', '')}\n"
                f"Hedge / Peer: {row.get('hedge_or_peer', '')}\n"
                f"Maturity: {row.get('maturity', '')}\n"
                f"Confidence: {row.get('confidence', '')}\n"
                f"Horizon: {row.get('horizon', '')}\n\n"
                f"Trade\n"
                f"Buy: {row.get('buy_legs', '')}\n"
                f"Sell: {row.get('sell_legs', '')}\n"
                f"Gross premium paid: {self._format_table_value(trade.get('gross_premium_paid', ''))}\n"
                f"Gross premium received: {self._format_table_value(trade.get('gross_premium_received', ''))}\n"
                f"Net premium: {self._format_table_value(row.get('net_premium', ''))}"
                f" {trade.get('net_premium_label', '')}\n\n"
                f"Sizing / Hedge\n"
                f"Executable package: {self._format_table_value(hedge_package.get('target_contracts', ''))} "
                f"target contract(s) vs {self._format_table_value(hedge_package.get('peer_contracts', ''))} "
                f"peer hedge contract(s)\n"
                f"Executable hedge ratio: {self._format_table_value(trade.get('hedge_ratio', ''))}\n"
                f"Continuous hedge ratio: {self._format_table_value(trade.get('continuous_hedge_ratio', ''))}\n"
                f"Ratio tolerance: {self._format_table_value(hedge_package.get('tolerance', ''))}\n"
                f"Ratio error: {self._format_table_value(hedge_package.get('relative_error', ''))}\n"
                f"Rounding status: {hedge_package.get('status', '')}\n"
                f"Hedge ratio source: {trade.get('hedge_ratio_source', '')}\n"
                f"Raw target delta per 1%: {self._format_table_value(exposures.get('raw_delta_target_per_1pct', ''))}\n"
                f"Raw peer delta per 1%: {self._format_table_value(exposures.get('raw_delta_peer_per_1pct', ''))}\n"
                f"Spillover beta: {self._format_table_value(exposures.get('spillover_beta', ''))}\n"
                f"Estimated net delta after hedge per 1%: "
                f"{self._format_table_value(exposures.get('estimated_net_delta_after_hedge_per_1pct', ''))}\n\n"
                f"Greek Exposure\n"
                f"Net vega: {self._format_table_value(exposures.get('net_vega', ''))}\n"
                f"Net gamma: {self._format_table_value(exposures.get('net_gamma', ''))}\n"
                f"Net theta/day: {self._format_table_value(exposures.get('net_theta_per_day', ''))}\n\n"
                f"Rationale\n{row.get('rationale', '')}\n\n"
                f"Risks\n{risks_text}\n\n"
                f"{self._format_source_signal_detail(source)}"
            )
            self._set_detail(detail)
            return

        if section == "anomaly":
            if self._last_anomalies is None or idx not in self._last_anomalies.index:
                self._set_detail("No selected market anomaly.")
                return
            row = self._last_anomalies.loc[idx]
            source = row.get("source_signal", {}) if hasattr(row, "get") else {}
            reasons = row.get("classification_reasons", [])
            if isinstance(reasons, list):
                reasons_text = "\n".join(f"- {r}" for r in reasons) if reasons else "None"
            else:
                reasons_text = str(reasons or "None")
            detail = (
                f"Market Anomaly\n"
                f"{row.get('title', '')}\n\n"
                f"Judgment: {row.get('judgment', '')}\n"
                f"Grouped Signals: {row.get('group_size', '')}\n"
                f"Trade Score: {self._format_table_value(row.get('trade_score', ''))}\n"
                f"Substitutability: {row.get('substitutability', '')}\n"
                f"Type: {row.get('anomaly_type', '')}\n"
                f"Affected: {self._format_table_value(row.get('affected_names', []))}\n"
                f"Likely Driver: {row.get('likely_driver', '')}\n"
                f"Context: {row.get('systemic_or_idiosyncratic', '')}\n"
                f"Spillover Relevance: {row.get('spillover_relevance', '')}\n\n"
                f"Why It Matters\n{row.get('why_it_matters', '')}\n\n"
                f"Impact On Trade Confidence\n{row.get('impact_on_trade_confidence', '')}\n\n"
                f"Classification Reasons\n{reasons_text}\n\n"
                f"{self._format_source_signal_detail(source)}"
            )
            self._set_detail(detail)
            return

        if self._last_opportunities is None or idx not in self._last_opportunities.index:
            self._set_detail("No selected signal.")
            return
        row = self._last_opportunities.loc[idx]
        detail = self._format_source_signal_detail(row)
        self._set_detail(detail)

    def _format_source_signal_detail(self, row):
        import math

        signal = row.get("signal", {}) if hasattr(row, "get") else {}
        if not isinstance(signal, dict):
            signal = {}
        narrative = signal.get("narrative", {}) if isinstance(signal.get("narrative", {}), dict) else {}
        significance = signal.get("significance", {}) if isinstance(signal.get("significance", {}), dict) else {}
        calculation = signal.get("calculation", {}) if isinstance(signal.get("calculation", {}), dict) else {}
        structure = signal.get("structure", {}) if isinstance(signal.get("structure", {}), dict) else {}
        dynamics = signal.get("dynamics", {}) if isinstance(signal.get("dynamics", {}), dict) else {}
        data_quality = signal.get("data_quality", {}) if isinstance(signal.get("data_quality", {}), dict) else {}
        classification = signal.get("classification", {}) if isinstance(signal.get("classification", {}), dict) else {}
        substitutability = (
            signal.get("substitutability", {}) if isinstance(signal.get("substitutability", {}), dict) else {}
        )
        contracts = signal.get("supporting_contracts", [])
        contract_lines = []
        for c in contracts[:10] if isinstance(contracts, list) else []:

            def fmt(v, spec):
                try:
                    f = float(v)
                    return format(f, spec) if math.isfinite(f) else "-"
                except Exception:
                    return "-"

            contract_lines.append(
                f"- {c.get('expiry', '-')} {c.get('call_put', '-')} "
                f"K={fmt(c.get('strike'), '.2f')} K/S={fmt(c.get('moneyness'), '.2f')} "
                f"IV={fmt(c.get('iv'), '.2%')} bid/ask={fmt(c.get('bid'), '.2f')}/{fmt(c.get('ask'), '.2f')} "
                f"vol={fmt(c.get('volume'), '.0f')} OI={fmt(c.get('open_interest'), '.0f')}"
            )
        contracts_text = (
            "\n".join(contract_lines) if contract_lines else "No contract-level quotes available for this signal."
        )
        warnings = str(row.get("warnings", "") or "None")
        detail = (
            f"Explanation\n"
            f"{narrative.get('headline', row.get('opportunity', ''))}\n"
            f"{narrative.get('what_differs', row.get('what_differs', ''))}\n"
            f"{narrative.get('why_matters', row.get('why_matters', ''))}\n\n"
            f"Decision\n"
            f"Judgment: {classification.get('judgment', row.get('judgment', '-'))}\n"
            f"Trade score: {fmt(classification.get('trade_score', row.get('trade_score')), '.3f')}\n"
            f"Reason: {' '.join(classification.get('reasons', []))}\n\n"
            f"Signal Calculation\n"
            f"{calculation.get('target_label', 'Target')}: {fmt(calculation.get('target_value'), '.4f')}\n"
            f"{calculation.get('synthetic_label', 'Weighted peer synthetic')}: "
            f"{fmt(calculation.get('synthetic_value'), '.4f')}\n"
            f"Spread: {fmt(calculation.get('spread'), '+.4f')}\n"
            f"{calculation.get('display', '')}\n\n"
            f"Statistical Context\n"
            f"Z-score: {significance.get('z_score', row.get('z_score', '-'))}\n"
            f"Percentile: {significance.get('percentile', row.get('percentile', '-'))}\n"
            f"{row.get('statistical_read', '')}\n\n"
            f"Structural Validation\n"
            f"Surface consistency: {structure.get('surface_vs_surface_grid_consistency', '')}\n"
            f"Similarity score: {structure.get('similarity_score', '')}\n"
            f"Substitutability: {substitutability.get('label', '-')} ({fmt(substitutability.get('score'), '.3f')})\n"
            f"{substitutability.get('prior', '')}\n"
            f"{row.get('comparability', '')}\n\n"
            f"Dynamics\n"
            f"Spillover: {dynamics.get('label', row.get('spillover_support', ''))}\n"
            f"Same-direction probability: {dynamics.get('same_direction_probability', '-')}\n"
            f"Lag profile: {dynamics.get('lag_profile', '-')}\n\n"
            f"Data Quality\n"
            f"Fit quality: {data_quality.get('fit_quality', row.get('data_quality', ''))}\n"
            f"RMSE: {data_quality.get('rmse', '-')}\n"
            f"Coverage: {data_quality.get('coverage', '-')}\n"
            f"Warnings: {warnings}\n\n"
            f"Supporting Contracts\n"
            f"{narrative.get('contracts', '')}\n"
            f"{contracts_text}"
        )
        return detail
