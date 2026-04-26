# display/plotting/term_plot.py
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Dict, Mapping

from analysis.confidence_bands import Bands
from analysis.term_view import (
    compute_term_fit_curve,
    term_ci_error,
    term_x_values,
)


def plot_atm_term_structure(
    ax: plt.Axes,
    atm_df: pd.DataFrame,
    x_units: str = "years",   # "years" or "days"
    fit: bool = False,
    show_ci: bool = False,    # draw CI bars if present
    show_quote_dispersion: bool = True,
    degree: int = 2,
    target_label: str = "Target ATM",
) -> None:
    ax.clear()
    if atm_df is None or atm_df.empty:
        ax.text(0.5, 0.5, "No ATM data", ha="center", va="center")
        return

    x = pd.to_numeric(atm_df["T"], errors="coerce").to_numpy(float)
    y = pd.to_numeric(atm_df["atm_vol"], errors="coerce").to_numpy(float)

    if x_units == "days":
        x_plot = x * 365.25
        x_label = "Time to Expiry (days)"
    else:
        x_plot = x
        x_label = "Time to Expiry (years)"

    yerr = term_ci_error(atm_df) if show_ci else None
    line = ax.plot(
        x_plot,
        y,
        marker="o",
        ms=5.0,
        lw=2.2,
        alpha=0.95,
        color="black",
        label=target_label,
        zorder=4,
    )[0]
    line.term_tooltip = _term_tooltip_text(target_label, atm_df, x_plot, y)
    line.set_picker(5)
    if show_quote_dispersion:
        _plot_quote_dispersion_lines(
            ax,
            x_plot,
            y,
            atm_df,
            np.isfinite(x_plot) & np.isfinite(y),
            color="black",
            label=f"{target_label} quote dispersion",
            zorder=3,
        )
    if yerr is not None:
        ax.errorbar(
            x_plot,
            y,
            yerr=yerr,
            fmt="none",
            capsize=3,
            alpha=0.45,
            color="black",
            label="_nolegend_",
            zorder=4,
        )

    if fit:
        grid_plot, fit_y = compute_term_fit_curve(atm_df, x_units=x_units, degree=degree)
        if grid_plot.size and fit_y.size:
            ax.plot(grid_plot, fit_y, linestyle="--", alpha=0.6, label="Term fit")

    ax.set_xlabel(x_label)
    ax.set_ylabel("ATM IV")
    ax.legend(loc="best", fontsize=8)
    _install_term_hover(ax)

def plot_peer_composite_term_structure(
    ax: plt.Axes,
    bands: Bands,
    *,
    label: str = "Peer composite ATM",
    line_kwargs: Optional[Dict] = None,
) -> Bands:
    """Plot peer-composite ATM term structure using pre-computed bands."""

    ax.fill_between(bands.x, bands.lo, bands.hi, alpha=0.20, label=f"CI ({int(bands.level*100)}%)")

    line_kwargs = dict(line_kwargs or {})
    line_kwargs.setdefault("lw", 1.8)
    ax.plot(bands.x, bands.mean, label=label, **line_kwargs)

    ax.set_xlabel("Pillar (days)")
    ax.set_ylabel("Implied Vol (ATM)")
    handles, labels = ax.get_legend_handles_labels()
    if handles and labels:
        order = {"Target ATM": 0, "ATM IV": 0, "Peer composite": 1, "Term fit": 2}
        pairs = sorted(zip(handles, labels), key=lambda item: order.get(item[1], 99))
        handles, labels = zip(*pairs)
        ax.legend(handles, labels, loc="best", fontsize=8)

    return bands


def plot_term_structure_comparison(
    ax: plt.Axes,
    atm_df: pd.DataFrame,
    *,
    peer_curves: Optional[Mapping[str, pd.DataFrame]] = None,
    synth_curve: Optional[pd.DataFrame] = None,
    weights: Optional[pd.Series | Mapping[str, float]] = None,
    x_units: str = "years",
    fit: bool = False,
    show_ci: bool = False,
    show_quote_dispersion: bool = True,
    title: Optional[str] = None,
    warning: Optional[str] = None,
    alignment_status: str = "",
    target_label: str = "Target ATM",
) -> None:
    """Plot target ATM term structure with peer, synthetic, and spread context."""
    ax.clear()

    plot_atm_term_structure(
        ax,
        atm_df,
        x_units=x_units,
        fit=fit,
        show_ci=show_ci,
        show_quote_dispersion=show_quote_dispersion,
        target_label=target_label,
    )

    peer_curves = dict(peer_curves or {})
    target_x = term_x_values(atm_df, x_units=x_units)
    target_mask = np.isfinite(target_x)
    if target_mask.any():
        target_min = float(np.nanmin(target_x[target_mask]))
        target_max = float(np.nanmax(target_x[target_mask]))
    else:
        target_min = target_max = np.nan

    if peer_curves:
        weight_map = pd.Series(weights, dtype=float) if weights is not None and len(weights) else pd.Series(dtype=float)
        for peer, curve in peer_curves.items():
            if curve is None or curve.empty or not {"T", "atm_vol"}.issubset(curve.columns):
                continue
            x = term_x_values(curve, x_units=x_units)
            y = pd.to_numeric(curve["atm_vol"], errors="coerce").to_numpy(float)
            mask = np.isfinite(x) & np.isfinite(y)
            if np.isfinite(target_min) and np.isfinite(target_max):
                mask &= (x >= target_min) & (x <= target_max)
            if not mask.any():
                continue
            w = weight_map.get(peer, np.nan)
            label = f"{peer} ({w:.1%})" if np.isfinite(w) else str(peer)
            _plot_term_band(ax, x, curve, mask, color=None, alpha=0.08)
            line = ax.plot(x[mask], y[mask], lw=1.25, alpha=0.48, label=label, zorder=2)[0]
            line.term_tooltip = _term_tooltip_text(str(peer), curve.loc[mask].copy(), x[mask], y[mask])
            line.set_picker(5)
            if show_quote_dispersion:
                _plot_quote_dispersion_lines(
                    ax,
                    x,
                    y,
                    curve,
                    mask,
                    color=line.get_color(),
                    label=f"{peer} quote dispersion",
                    zorder=1,
                )

    if synth_curve is not None and not synth_curve.empty and {"T", "atm_vol"}.issubset(synth_curve.columns):
        x_syn = term_x_values(synth_curve, x_units=x_units)
        y_syn = pd.to_numeric(synth_curve["atm_vol"], errors="coerce").to_numpy(float)
        mask_syn = np.isfinite(x_syn) & np.isfinite(y_syn)
        if np.isfinite(target_min) and np.isfinite(target_max):
            mask_syn &= (x_syn >= target_min) & (x_syn <= target_max)
        if mask_syn.any():
            if {"atm_lo", "atm_hi"}.issubset(synth_curve.columns):
                lo = pd.to_numeric(synth_curve["atm_lo"], errors="coerce").to_numpy(float)
                hi = pd.to_numeric(synth_curve["atm_hi"], errors="coerce").to_numpy(float)
                band_mask = mask_syn & np.isfinite(lo) & np.isfinite(hi)
                if band_mask.any():
                    ax.fill_between(x_syn[band_mask], lo[band_mask], hi[band_mask], color="tab:orange", alpha=0.16, label="_nolegend_")
            line = ax.plot(
                x_syn[mask_syn],
                y_syn[mask_syn],
                color="tab:orange",
                lw=1.25,
                alpha=0.9,
                label="Peer composite",
                zorder=3,
            )[0]
            line.term_tooltip = _term_tooltip_text("Peer composite", synth_curve.loc[mask_syn].copy(), x_syn[mask_syn], y_syn[mask_syn])
            line.set_picker(5)

    if title:
        ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    if handles and labels:
        order = {target_label: 0, "Peer composite": 1, "Term fit": 2}
        pairs = sorted(zip(handles, labels), key=lambda item: order.get(item[1], 99))
        handles, labels = zip(*pairs)
        ax.legend(handles, labels, loc="best", fontsize=8)
    _install_term_hover(ax)
    ax.grid(True, alpha=0.2)


def _plot_term_band(ax: plt.Axes, x: np.ndarray, curve: pd.DataFrame, mask: np.ndarray, *, color=None, alpha: float = 0.12) -> None:
    if curve is None or curve.empty or not {"atm_lo", "atm_hi"}.issubset(curve.columns):
        return
    lo = pd.to_numeric(curve["atm_lo"], errors="coerce").to_numpy(float)
    hi = pd.to_numeric(curve["atm_hi"], errors="coerce").to_numpy(float)
    band_mask = mask & np.isfinite(lo) & np.isfinite(hi)
    if band_mask.any():
        ax.fill_between(x[band_mask], lo[band_mask], hi[band_mask], color=color, alpha=alpha, label="_nolegend_")


def _plot_quote_dispersion_lines(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    curve: pd.DataFrame,
    mask: np.ndarray,
    *,
    color=None,
    label: str,
    zorder: int,
) -> None:
    if curve is None or curve.empty or "quote_dispersion" not in curve.columns:
        return
    disp = pd.to_numeric(curve["quote_dispersion"], errors="coerce").to_numpy(float)
    line_mask = mask & np.isfinite(disp) & (disp >= 0.0)
    if not line_mask.any():
        return
    lo = np.clip(y - disp, 0.0, None)
    hi = y + disp
    style = {"color": color, "lw": 0.75, "alpha": 0.45, "linestyle": ":", "zorder": zorder}
    ax.plot(x[line_mask], lo[line_mask], label=label, **style)
    ax.plot(x[line_mask], hi[line_mask], label="_nolegend_", **style)


def _term_tooltip_text(label: str, curve: pd.DataFrame, x_plot: np.ndarray, y: np.ndarray) -> list[str]:
    tips: list[str] = []
    if curve is None or curve.empty:
        return tips
    for i, (_, row) in enumerate(curve.iterrows()):
        expiry = str(row.get("expiry", "")).split(" ")[0]
        t_val = row.get("T", np.nan)
        n_obs = row.get("n_obs", row.get("count", np.nan))
        lo = row.get("atm_lo", np.nan)
        hi = row.get("atm_hi", np.nan)
        disp = row.get("atm_dispersion", np.nan)
        quote_disp = row.get("quote_dispersion", np.nan)
        band = ""
        if pd.notna(lo) and pd.notna(hi):
            band = f"\nband: [{float(lo):.2%}, {float(hi):.2%}]"
        if pd.notna(quote_disp):
            band += f"\nquote dispersion: {float(quote_disp):.2%}"
        elif pd.notna(disp):
            band += f"\ndispersion: {float(disp):.2%}"
        tips.append(
            f"{label}\nexpiry: {expiry or 'n/a'}\nT: {float(t_val):.4f}y\nATM IV: {float(y[i]):.2%}"
            f"\nN observations: {int(n_obs) if pd.notna(n_obs) else 'n/a'}{band}"
        )
    return tips


def _install_term_hover(ax: plt.Axes) -> None:
    canvas = ax.figure.canvas
    if canvas is None:
        return
    old_cid = getattr(ax, "_term_hover_cid", None)
    if old_cid is not None:
        try:
            canvas.mpl_disconnect(old_cid)
        except Exception:
            pass
    annot = ax.annotate(
        "",
        xy=(0, 0),
        xytext=(10, 10),
        textcoords="offset points",
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "0.7", "alpha": 0.95},
        fontsize=8,
    )
    annot.set_visible(False)

    def _on_motion(event):
        if event.inaxes is not ax:
            if annot.get_visible():
                annot.set_visible(False)
                canvas.draw_idle()
            return
        for line in ax.lines:
            tips = getattr(line, "term_tooltip", None)
            if not tips:
                continue
            contains, info = line.contains(event)
            if not contains:
                continue
            inds = info.get("ind", [])
            if len(inds) == 0:
                continue
            idx = int(inds[0])
            x = np.asarray(line.get_xdata(), dtype=float)
            y = np.asarray(line.get_ydata(), dtype=float)
            if idx >= len(x) or idx >= len(y) or idx >= len(tips):
                continue
            annot.xy = (x[idx], y[idx])
            annot.set_text(tips[idx])
            annot.set_visible(True)
            canvas.draw_idle()
            return
        if annot.get_visible():
            annot.set_visible(False)
            canvas.draw_idle()

    ax._term_hover_cid = canvas.mpl_connect("motion_notify_event", _on_motion)
