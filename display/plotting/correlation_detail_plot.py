"""
Correlation plotting without pillars, with configurable weighting modes.

This module computes correlations across implied-volatility surfaces using
the first few expiries for each ticker (as opposed to fixed pillar days).
It then provides a heatmap and optional ETF weight annotations.  You can
specify how weights are computed via the ``weight_mode`` parameter.
"""

from __future__ import annotations

from typing import Iterable, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analysis.correlation_view import (
    CorrelationViewData,
    compute_atm_curve_simple,
    corr_by_expiry_rank,
    coverage_by_ticker,
    finite_cell_summary,
    maybe_compute_weights,
    overlap_counts,
    prepare_correlation_view,
    split_weight_mode,
)
from analysis.settings import (
    DEFAULT_ATM_BAND,
    DEFAULT_CLIP_NEGATIVE_WEIGHTS,
    DEFAULT_MAX_EXPIRIES,
    DEFAULT_WEIGHT_METHOD,
    DEFAULT_WEIGHT_POWER,
)

# ---------------------------------------------------------------------------
# Helpers to compute ATM curves and correlations without using fixed pillars
# ---------------------------------------------------------------------------


def _compute_atm_curve_simple(df: pd.DataFrame, atm_band: float = DEFAULT_ATM_BAND) -> pd.DataFrame:
    """Backward-compatible wrapper around analysis.correlation_view."""
    return compute_atm_curve_simple(df, atm_band=atm_band)


def _corr_by_expiry_rank(
    get_slice,
    tickers: Iterable[str],
    asof: str,
    max_expiries: int = DEFAULT_MAX_EXPIRIES,
    atm_band: float = DEFAULT_ATM_BAND,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Backward-compatible wrapper around analysis.correlation_view."""
    return corr_by_expiry_rank(
        get_slice=get_slice,
        tickers=tickers,
        asof=asof,
        max_expiries=max_expiries,
        atm_band=atm_band,
    )


def _maybe_compute_weights(
    target: Optional[str],
    peers: Optional[Iterable[str]],
    *,
    asof: str,
    weight_mode: str,
    weight_power: float,
    clip_negative: bool,
    **weight_config,
) -> Optional[pd.Series]:
    """Backward-compatible wrapper around analysis.correlation_view."""
    return maybe_compute_weights(
        target=target,
        peers=peers,
        asof=asof,
        weight_mode=weight_mode,
        weight_power=weight_power,
        clip_negative=clip_negative,
        **weight_config,
    )


# ---------------------------------------------------------------------------
# Correlation: compute and plot (optionally show weights)
# ---------------------------------------------------------------------------


def compute_and_plot_correlation(
    ax: plt.Axes,
    get_smile_slice,
    tickers: Iterable[str],
    asof: str,
    *,
    target: Optional[str] = None,
    peers: Optional[Iterable[str]] = None,
    atm_band: float = DEFAULT_ATM_BAND,
    show_values: bool = True,
    clip_negative: bool = DEFAULT_CLIP_NEGATIVE_WEIGHTS,
    weight_power: float = DEFAULT_WEIGHT_POWER,
    max_expiries: int = DEFAULT_MAX_EXPIRIES,
    weight_mode: str = DEFAULT_WEIGHT_METHOD,
    **weight_config,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.Series]]:
    """
    Compute a correlation matrix and draw a heatmap without relying on pillars.

    Parameters remain compatible with the upstream version but no longer accept
    pillar-related options.  The ``weight_mode`` is forwarded to
    :func:`analysis.unified_weights.compute_unified_weights`. Additional weight
    configuration such as ``weight_power`` and ``clip_negative`` can be
    supplied, along with any extra keyword arguments understood by the unified
    weight system.
    """
    view = prepare_correlation_view(
        get_smile_slice=get_smile_slice,
        tickers=tickers,
        asof=asof,
        target=target,
        peers=peers,
        atm_band=atm_band,
        clip_negative=clip_negative,
        weight_power=weight_power,
        max_expiries=max_expiries,
        weight_mode=weight_mode,
        **weight_config,
    )

    plot_correlation_details(
        ax,
        view.corr_df,
        weights=view.weights,
        show_values=show_values,
        atm_df=view.atm_df,
        view_data=view,
        target=target,
        asof=asof,
        weight_mode=weight_mode,
        max_expiries=max_expiries,
    )
    return view.atm_df, view.corr_df, view.weights


def _clear_corr_side_axes(fig: plt.Figure) -> None:
    """Remove side panels created by this module on prior redraws."""
    if hasattr(fig, "_correlation_colorbar"):
        try:
            fig._correlation_colorbar.remove()
        except Exception:
            pass
        try:
            delattr(fig, "_correlation_colorbar")
        except Exception:
            pass
    for attr in ("_corr_weight_ax", "_corr_coverage_ax", "_corr_colorbar_ax"):
        if hasattr(fig, attr):
            try:
                getattr(fig, attr).remove()
            except Exception:
                pass
            try:
                delattr(fig, attr)
            except Exception:
                pass


def _coverage_by_ticker(
    corr_df: pd.DataFrame,
    atm_df: Optional[pd.DataFrame],
) -> pd.Series:
    """Backward-compatible wrapper around analysis.correlation_view."""
    return coverage_by_ticker(corr_df, atm_df)


def _overlap_counts(
    corr_df: pd.DataFrame,
    atm_df: Optional[pd.DataFrame],
) -> Optional[pd.DataFrame]:
    """Backward-compatible wrapper around analysis.correlation_view."""
    return overlap_counts(corr_df, atm_df)


def _split_weight_mode(weight_mode: Optional[str]) -> tuple[str, str]:
    """Backward-compatible wrapper around analysis.correlation_view."""
    return split_weight_mode(weight_mode)


def _finite_cell_summary(corr_df: pd.DataFrame) -> tuple[int, int, float]:
    """Backward-compatible wrapper around analysis.correlation_view."""
    return finite_cell_summary(corr_df)


def _reserve_correlation_layout(
    ax: plt.Axes,
    *,
    has_weights: bool,
) -> tuple[plt.Axes, Optional[plt.Axes]]:
    """Lay out the heatmap and optional side panels in non-overlapping slots."""
    fig = ax.figure
    if not hasattr(fig, "_orig_position"):
        fig._orig_position = ax.get_position().bounds
        sp = fig.subplotpars
        fig._orig_subplotpars = (sp.left, sp.right, sp.bottom, sp.top)

    if has_weights:
        ax.set_position([0.08, 0.16, 0.58, 0.68])
        main = ax.get_position()
        side_x = main.x1 + 0.105
        side_w = max(0.12, 0.96 - side_x)
        weight_ax = fig.add_axes([side_x, main.y0 + main.height * 0.24, side_w, main.height * 0.52])
        return ax, weight_ax

    if hasattr(fig, "_orig_position"):
        ax.set_position(fig._orig_position)
    return ax, None


def plot_correlation_details(
    ax: plt.Axes,
    corr_df: pd.DataFrame,
    weights: Optional[pd.Series] = None,
    show_values: bool = True,
    *,
    atm_df: Optional[pd.DataFrame] = None,
    view_data: Optional[CorrelationViewData] = None,
    target: Optional[str] = None,
    asof: Optional[str] = None,
    weight_mode: Optional[str] = None,
    max_expiries: Optional[int] = None,
) -> None:
    """
    Heatmap of the correlation matrix with an optional peer-weight panel.

    The matrix is a diagnostic for pairwise similarity across expiry-rank ATM
    IV features.  Weights may come from a different method/basis, so they are
    shown as a separate side panel instead of being implied by heatmap color.
    """
    fig = ax.figure
    _clear_corr_side_axes(fig)
    ax.clear()
    if corr_df is None or corr_df.empty:
        ax.text(0.5, 0.5, "No correlation data", ha="center", va="center")
        return

    if view_data is not None:
        corr_df = view_data.corr_df
        weights = view_data.weights if weights is None else weights
        atm_df = view_data.atm_df if atm_df is None else atm_df
        target = view_data.context.get("target") or target
        asof = view_data.context.get("asof") or asof
        weight_mode = view_data.context.get("weight_mode") or weight_mode
        max_expiries = view_data.context.get("max_expiries") or max_expiries

    data = corr_df.to_numpy(dtype=float)
    if view_data is not None:
        finite_count = view_data.finite_count
        total_elements = view_data.total_cells
        data_quality = view_data.finite_ratio
    else:
        finite_count, total_elements, data_quality = _finite_cell_summary(corr_df)

    if finite_count == 0:
        ax.text(
            0.5,
            0.5,
            "No valid correlations\n(insufficient overlapping data)",
            ha="center",
            va="center",
            fontsize=12,
        )
        return

    view_data.coverage if view_data is not None else _coverage_by_ticker(corr_df, atm_df)
    overlap = view_data.overlap if view_data is not None else _overlap_counts(corr_df, atm_df)
    has_weights = bool(weights is not None and not weights.dropna().empty)
    _, weight_ax = _reserve_correlation_layout(
        ax,
        has_weights=has_weights,
    )
    min_overlap = None
    if overlap is not None and len(overlap) > 1:
        mask = ~np.eye(len(overlap), dtype=bool)
        vals = overlap.to_numpy(dtype=float)[mask]
        vals = vals[np.isfinite(vals)]
        if vals.size:
            min_overlap = int(np.nanmin(vals))

    if view_data is not None:
        method_label = view_data.method_label
        basis_label = view_data.basis_label
    else:
        method_label, basis_label = _split_weight_mode(weight_mode)
    max_exp_label = f"{int(max_expiries)}" if max_expiries else "available"
    feature_diag = view_data.context.get("feature_diagnostics", {}) if view_data is not None else {}
    coord = feature_diag.get("coordinate_system") or f"ATM IV expiry ranks, first {max_exp_label} expiries"
    display_method = view_data.context.get("similarity_display_method", "corr") if view_data is not None else "corr"
    context_parts = [str(coord)]
    context_parts.append(f"finite cells {finite_count}/{total_elements} ({data_quality:.0%})")
    if min_overlap is not None:
        context_parts.append(f"min overlap {min_overlap}")
    if has_weights:
        context_parts.append(f"weights {method_label} on {basis_label}")
    if asof:
        context_parts.append(str(asof))
    context_msg = " | ".join(context_parts)
    ax.text(
        0.0,
        1.02,
        context_msg,
        ha="left",
        va="bottom",
        fontsize=9,
        color="#444444",
        transform=ax.transAxes,
        clip_on=False,
    )

    im = ax.imshow(data, vmin=-1.0, vmax=1.0, cmap="RdBu_r", interpolation="nearest", aspect="auto")

    try:
        bbox = ax.get_position()
        cax = fig.add_axes([bbox.x1 + 0.01, bbox.y0, 0.012, bbox.height])
        cbar = fig.colorbar(im, cax=cax)
        cbar_labels = {
            "cosine": "Cosine similarity",
            "pca": "PCA score similarity",
        }
        cbar.set_label(cbar_labels.get(display_method, "Correlation"))
        cbar.ax.tick_params(labelsize=8)
        fig._correlation_colorbar = cbar
        fig._corr_colorbar_ax = cax
    except Exception:
        pass

    ax.set_xticks(range(len(corr_df.columns)))
    ax.set_yticks(range(len(corr_df.index)))
    ax.set_xticklabels(corr_df.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(corr_df.index, fontsize=9)
    title = f"{basis_label} {display_method.title()} Matrix"
    if target:
        title = f"{target.upper()} Peer Similarity - {title}"
    ax.set_title(title, pad=22)
    ax.set_xlabel("Ticker")
    ax.set_ylabel("Ticker")

    ax.set_xticks(np.arange(len(corr_df.columns) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(corr_df.index) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)

    if show_values:
        n, m = data.shape
        for i in range(n):
            for j in range(m):
                val = data[i, j]
                if np.isfinite(val):
                    cell_text = f"{val:.2f}"
                    if overlap is not None:
                        try:
                            cell_text = f"{val:.2f}\nn={int(overlap.iat[i, j])}"
                        except Exception:
                            pass
                    ax.text(
                        j,
                        i,
                        cell_text,
                        ha="center",
                        va="center",
                        fontsize=7.5 if overlap is not None else 8,
                        color=("white" if abs(val) > 0.5 else "black"),
                        weight="bold",
                    )
                else:
                    cell_text = "n<2" if overlap is not None else "N/A"
                    ax.text(
                        j,
                        i,
                        cell_text,
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="gray",
                        style="italic",
                    )

    if has_weights and weight_ax is not None:
        w = weights.dropna().astype(float)
        w = w[w.index.isin(corr_df.index) | w.index.isin(corr_df.columns)]
        if not w.empty:
            w_sorted = w.sort_values(ascending=True)
            colors = ["#2f6f9f" if str(k).upper() != str(target or "").upper() else "#444444" for k in w_sorted.index]
            weight_ax.barh(range(len(w_sorted)), w_sorted.to_numpy(float), color=colors, alpha=0.88)
            weight_ax.set_yticks(range(len(w_sorted)))
            weight_ax.set_yticklabels(w_sorted.index, fontsize=8)
            weight_ax.set_title("Peer Weights", fontsize=10)
            weight_ax.set_xlabel("Weight", fontsize=8)
            weight_ax.tick_params(axis="x", labelsize=8)
            weight_ax.grid(axis="x", alpha=0.25)
            for spine in ("top", "right"):
                weight_ax.spines[spine].set_visible(False)
            for i, v in enumerate(w_sorted.to_numpy(float)):
                weight_ax.text(v, i, f"{v:.2f}", va="center", ha="left", fontsize=8)
            fig._corr_weight_ax = weight_ax


def scatter_corr_matrix(
    df_or_path: pd.DataFrame | str,
    columns: Optional[Iterable[str]] = None,
    *,
    plot: bool = True,
) -> pd.DataFrame:
    """
    Compute pairwise correlation coefficients for numeric columns and
    optionally draw a scatter‑matrix plot.

    If ``plot`` is True, the scatter-matrix uses pandas.plotting.scatter_matrix().
    """
    if isinstance(df_or_path, str):
        df = pd.read_parquet(df_or_path)
    else:
        df = df_or_path.copy()

    if columns is not None:
        cols = [c for c in columns if c in df.columns]
        df = df[cols]

    num_df = df.select_dtypes(include=[np.number]).dropna()
    if num_df.shape[1] < 2:
        return pd.DataFrame()

    corr_df = num_df.corr()

    if plot:
        try:
            pd.plotting.scatter_matrix(num_df, figsize=(6, 6))
        except Exception:
            pass

    return corr_df
