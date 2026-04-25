"""
Relative value plotting utilities.

Functions in this module accept pre-computed DataFrames (from
``analysis.rv_analysis``) and render them onto a provided
``matplotlib.Axes`` object so that the GUI can call them from a
background thread and hand the finished figure back to Tkinter.
"""

from __future__ import annotations

from typing import Dict, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


__all__ = [
    "plot_surface_residual_heatmap",
    "plot_skew_spread",
    "plot_rv_signals_table",
]


# ---------------------------------------------------------------------------
# Surface residual heatmap (Phase 1 output)
# ---------------------------------------------------------------------------

def plot_surface_residual_heatmap(
    ax: plt.Axes,
    residual_df: pd.DataFrame,
    *,
    title: Optional[str] = None,
    vmax: Optional[float] = None,
    annotate: bool = True,
) -> None:
    """Render a diverging heatmap of the surface residual.

    Blue cells indicate the target is *cheaper* than the synthetic
    (negative residual); red cells indicate it is *richer*.  Cell values
    are z-scores when the rolling history is available, or raw normalised
    residuals otherwise.

    Parameters
    ----------
    ax : plt.Axes
        Axis to draw on (will be cleared).
    residual_df : DataFrame
        Rows = moneyness bins, columns = tenor (days).
        Values are (z-scored) normalised residuals.
    title : str, optional
    vmax : float, optional
        Colour scale symmetry limit (|z| or |residual|).  Auto-detected
        from data when omitted.
    annotate : bool
        Whether to print numeric values inside each cell.
    """
    ax.clear()

    if residual_df is None or residual_df.empty:
        ax.text(0.5, 0.5, "No residual data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title or "Surface Residual")
        return

    arr = residual_df.to_numpy(float)
    finite_arr = arr[np.isfinite(arr)]
    if finite_arr.size == 0:
        ax.text(0.5, 0.5, "All NaN", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title or "Surface Residual")
        return

    if vmax is None:
        vmax = max(float(np.nanmax(np.abs(finite_arr))), 1e-6)

    n_rows, n_cols = arr.shape
    im = ax.imshow(
        arr,
        cmap="RdYlBu_r",
        vmin=-vmax,
        vmax=vmax,
        aspect="auto",
        origin="lower",
    )

    # Axis labels
    col_labels = [str(c) for c in residual_df.columns]
    row_labels = [str(r) for r in residual_df.index]
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_xlabel("Tenor (days)", fontsize=9)
    ax.set_ylabel("Moneyness", fontsize=9)

    if annotate:
        for i in range(n_rows):
            for j in range(n_cols):
                v = arr[i, j]
                if np.isfinite(v):
                    txt = f"{v:.1f}"
                    ax.text(
                        j, i, txt,
                        ha="center", va="center",
                        fontsize=6,
                        color="white" if abs(v) > vmax * 0.65 else "black",
                    )

    try:
        cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("z-score (rich→red, cheap→blue)", fontsize=8)
        cbar.ax.tick_params(labelsize=7)
    except Exception:
        pass

    ax.set_title(title or "Surface Residual Heatmap (target − synthetic)", fontsize=10)


# ---------------------------------------------------------------------------
# Skew and curvature spread plot (Phase 2 output)
# ---------------------------------------------------------------------------

def plot_skew_spread(
    ax: plt.Axes,
    skew_df: pd.DataFrame,
    *,
    title: Optional[str] = None,
    x_units: str = "days",
) -> None:
    """Plot target vs synthetic skew and curvature spread across tenors.

    Parameters
    ----------
    ax : plt.Axes
        Axis to draw on (will be cleared).
    skew_df : DataFrame
        Output of ``analysis.rv_analysis.compute_skew_spread``.
        Must contain columns: T, T_days, target_skew, synth_skew,
        skew_spread, target_curv, synth_curv, curv_spread.
    title : str, optional
    x_units : str
        ``"days"`` or ``"years"``.
    """
    ax.clear()

    required = {"T", "T_days", "target_skew", "synth_skew", "skew_spread",
                "target_curv", "synth_curv", "curv_spread"}
    if skew_df is None or skew_df.empty or not required.issubset(skew_df.columns):
        ax.text(0.5, 0.5, "No skew/curvature data", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title(title or "Skew & Curvature Spread")
        return

    x = (
        skew_df["T_days"].to_numpy(float)
        if x_units == "days"
        else skew_df["T"].to_numpy(float)
    )
    x_label = "Tenor (days)" if x_units == "days" else "Time to Expiry (years)"

    valid = np.isfinite(x)

    # Skew lines
    tgt_s = skew_df["target_skew"].to_numpy(float)
    syn_s = skew_df["synth_skew"].to_numpy(float)
    spread_s = skew_df["skew_spread"].to_numpy(float)

    m_tgt = valid & np.isfinite(tgt_s)
    m_syn = valid & np.isfinite(syn_s)
    m_spr = valid & np.isfinite(spread_s)

    if m_tgt.any():
        ax.plot(x[m_tgt], tgt_s[m_tgt], "o-", lw=1.6, label="Target skew", color="tab:blue")
    if m_syn.any():
        ax.plot(x[m_syn], syn_s[m_syn], "o--", lw=1.3, alpha=0.7, label="Synthetic skew",
                color="tab:orange")

    # Curvature lines on secondary axis
    ax2 = ax.twinx()
    tgt_c = skew_df["target_curv"].to_numpy(float)
    syn_c = skew_df["synth_curv"].to_numpy(float)
    m_tgt_c = valid & np.isfinite(tgt_c)
    m_syn_c = valid & np.isfinite(syn_c)
    if m_tgt_c.any():
        ax2.plot(x[m_tgt_c], tgt_c[m_tgt_c], "s-", lw=1.0, alpha=0.5,
                 label="Target curv", color="tab:green")
    if m_syn_c.any():
        ax2.plot(x[m_syn_c], syn_c[m_syn_c], "s--", lw=0.9, alpha=0.4,
                 label="Synth curv", color="tab:red")
    ax2.set_ylabel("Curvature", fontsize=8)
    ax2.tick_params(labelsize=7)

    # Spread as a bar fill around zero
    if m_spr.any():
        ax.bar(
            x[m_spr],
            spread_s[m_spr],
            width=max(1.0, float(np.diff(x[m_spr]).mean()) * 0.35) if m_spr.sum() > 1 else 1.5,
            bottom=0,
            alpha=0.22,
            color=np.where(spread_s[m_spr] > 0, "tab:red", "tab:blue"),
            label="Skew spread",
        )

    ax.axhline(0, color="black", lw=0.7, alpha=0.5)
    ax.set_xlabel(x_label, fontsize=9)
    ax.set_ylabel("Skew (target / synthetic)", fontsize=9)

    # Merge legends from both axes
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    if lines1 or lines2:
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="best")

    ax.grid(True, alpha=0.2)
    ax.set_title(title or "Skew & Curvature Spread (target vs synthetic)", fontsize=10)


# ---------------------------------------------------------------------------
# RV signals table (Phase 4 output — matplotlib fallback)
# ---------------------------------------------------------------------------

def plot_rv_signals_table(
    ax: plt.Axes,
    signals_df: pd.DataFrame,
    *,
    title: Optional[str] = None,
) -> None:
    """Render ranked RV signals as a color-coded matplotlib table.

    This is the fallback visualisation used in the main plot canvas when
    the "RV Signals" plot type is selected.  The dedicated GUI tab
    (``display/gui/rv_signals_tab.py``) shows the same data in a
    Treeview widget for better interactivity.

    Parameters
    ----------
    ax : plt.Axes
        Axis to draw on (will be cleared and set to axis-off).
    signals_df : DataFrame
        Output of ``analysis.rv_analysis.generate_rv_signals``.
    title : str, optional
    """
    ax.clear()
    ax.axis("off")
    ax.set_title(title or "Relative Value Signals (target vs synthetic)", fontsize=10)

    if signals_df is None or signals_df.empty:
        ax.text(0.5, 0.5, "No signals / insufficient data",
                ha="center", va="center", transform=ax.transAxes, fontsize=10)
        return

    display_cols = ["signal_type", "T_days", "value", "synth_value", "spread", "z_score", "pct_rank"]
    present_cols = [c for c in display_cols if c in signals_df.columns]
    col_labels = {
        "signal_type": "Type",
        "T_days": "T (d)",
        "value": "Target",
        "synth_value": "Synth",
        "spread": "Spread",
        "z_score": "Z",
        "pct_rank": "Pct",
    }

    df = signals_df.head(20).reset_index(drop=True)
    header = [col_labels.get(c, c) for c in present_cols]
    cell_data: list[list[str]] = []
    for _, row in df.iterrows():
        cells: list[str] = []
        for c in present_cols:
            v = row.get(c, "")
            if isinstance(v, float):
                if c in ("z_score", "pct_rank"):
                    cells.append(f"{v:.2f}" if np.isfinite(v) else "—")
                else:
                    cells.append(f"{v:.4f}" if np.isfinite(v) else "—")
            else:
                cells.append(str(v) if v != "" else "—")
        cell_data.append(cells)

    if not cell_data:
        ax.text(0.5, 0.5, "No signals", ha="center", va="center", transform=ax.transAxes)
        return

    # Build colour array for rows
    n_rows = len(cell_data)
    n_cols = len(present_cols)
    cell_colours: list[list] = []
    for _, row in df.iterrows():
        z = row.get("z_score", np.nan)
        z_f = float(z) if pd.notna(z) else np.nan
        if np.isfinite(z_f) and z_f > 1.5:
            bg = "#ffe8e8"
        elif np.isfinite(z_f) and z_f < -1.5:
            bg = "#e8f0ff"
        else:
            bg = "#f5f5f5"
        cell_colours.append([bg] * n_cols)

    tbl = ax.table(
        cellText=cell_data,
        colLabels=header,
        cellColours=cell_colours,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)

    col_widths = {
        "signal_type": 0.14,
        "T_days": 0.07,
        "value": 0.09,
        "synth_value": 0.09,
        "spread": 0.09,
        "z_score": 0.07,
        "pct_rank": 0.07,
    }
    for j, c in enumerate(present_cols):
        w = col_widths.get(c, 0.10)
        for i in range(n_rows + 1):
            tbl[i, j].set_width(w)
