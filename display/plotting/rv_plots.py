"""
Relative value plotting utilities.

Functions in this module accept pre-computed DataFrames (from
``analysis.rv_analysis``) and render them onto a provided
``matplotlib.Axes`` object so that the GUI can call them from a
background thread and hand the finished figure back to Tkinter.
"""

from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

__all__ = [
    "plot_surface_residual_heatmap",
    "plot_skew_spread",
]


# ---------------------------------------------------------------------------
# Surface residual heatmap (Phase 1 output)
# ---------------------------------------------------------------------------


def _clear_rv_heatmap_colorbar(fig: plt.Figure) -> None:
    """Remove RV heatmap helper axes from a prior redraw."""
    if hasattr(fig, "_rv_heatmap_colorbar"):
        try:
            fig._rv_heatmap_colorbar.remove()
        except Exception:
            pass
        try:
            delattr(fig, "_rv_heatmap_colorbar")
        except Exception:
            pass
    if hasattr(fig, "_rv_heatmap_colorbar_ax"):
        try:
            fig._rv_heatmap_colorbar_ax.remove()
        except Exception:
            pass
        try:
            delattr(fig, "_rv_heatmap_colorbar_ax")
        except Exception:
            pass


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
    fig = ax.figure
    if not hasattr(fig, "_rv_heatmap_orig_position"):
        fig._rv_heatmap_orig_position = ax.get_position().frozen()
    _clear_rv_heatmap_colorbar(fig)
    ax.set_position(fig._rv_heatmap_orig_position)
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
                        j,
                        i,
                        txt,
                        ha="center",
                        va="center",
                        fontsize=6,
                        color="white" if abs(v) > vmax * 0.65 else "black",
                    )

    try:
        bbox = ax.get_position()
        cax = fig.add_axes([bbox.x1 + 0.012, bbox.y0, 0.014, bbox.height])
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label("z-score (rich→red, cheap→blue)", fontsize=8)
        cbar.ax.tick_params(labelsize=7)
        fig._rv_heatmap_colorbar = cbar
        fig._rv_heatmap_colorbar_ax = cax
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

    required = {"T", "T_days", "target_skew", "synth_skew", "skew_spread", "target_curv", "synth_curv", "curv_spread"}
    if skew_df is None or skew_df.empty or not required.issubset(skew_df.columns):
        ax.text(0.5, 0.5, "No skew/curvature data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title or "Skew & Curvature Spread")
        return

    x = skew_df["T_days"].to_numpy(float) if x_units == "days" else skew_df["T"].to_numpy(float)
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
        ax.plot(x[m_syn], syn_s[m_syn], "o--", lw=1.3, alpha=0.7, label="Synthetic skew", color="tab:orange")

    # Curvature lines on secondary axis
    ax2 = ax.twinx()
    tgt_c = skew_df["target_curv"].to_numpy(float)
    syn_c = skew_df["synth_curv"].to_numpy(float)
    m_tgt_c = valid & np.isfinite(tgt_c)
    m_syn_c = valid & np.isfinite(syn_c)
    if m_tgt_c.any():
        ax2.plot(x[m_tgt_c], tgt_c[m_tgt_c], "s-", lw=1.0, alpha=0.5, label="Target curv", color="tab:green")
    if m_syn_c.any():
        ax2.plot(x[m_syn_c], syn_c[m_syn_c], "s--", lw=0.9, alpha=0.4, label="Synth curv", color="tab:red")
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
