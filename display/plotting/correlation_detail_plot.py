"""
Correlation plotting without pillars, with configurable weighting modes.

This module computes correlations across implied-volatility surfaces using
the first few expiries for each ticker (as opposed to fixed pillar days).
It then provides a heatmap and optional ETF weight annotations.  You can
specify how weights are computed via the ``weight_mode`` parameter.
"""

from __future__ import annotations

from typing import Iterable, Tuple, Optional, List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analysis.unified_weights import compute_unified_weights
from analysis.compute_or_load import compute_or_load


# ---------------------------------------------------------------------------
# Helpers to compute ATM curves and correlations without using fixed pillars
# ---------------------------------------------------------------------------


def _compute_atm_curve_simple(df: pd.DataFrame, atm_band: float = 0.05) -> pd.DataFrame:
    """
    Compute a simple ATM implied volatility per expiry using only the median
    of near‑ATM quotes.  This function avoids reliance on fixed pillar days.

    Parameters
    ----------
    df : pandas.DataFrame
        Slice of option quotes for a single ticker on a single date.
    atm_band : float, optional
        Relative band around ATM (moneyness ~ 1) used to compute medians.

    Returns
    -------
    pandas.DataFrame
        Columns ``T`` and ``atm_vol`` sorted by increasing maturity ``T``.
    """
    need_cols = {"T", "moneyness", "sigma"}
    if df is None or df.empty or not need_cols.issubset(df.columns):
        return pd.DataFrame(columns=["T", "atm_vol"])
    d = df.copy()
    d["T"] = pd.to_numeric(d["T"], errors="coerce")
    d["moneyness"] = pd.to_numeric(d["moneyness"], errors="coerce")
    d["sigma"] = pd.to_numeric(d["sigma"], errors="coerce")
    d = d.dropna(subset=["T", "moneyness", "sigma"])
    rows: List[dict[str, float]] = []
    for T_val, grp in d.groupby("T"):
        g = grp.dropna(subset=["moneyness", "sigma"])
        in_band = g.loc[(g["moneyness"] - 1.0).abs() <= atm_band]
        if not in_band.empty:
            atm_vol = float(in_band["sigma"].median())
        else:
            idx = int((g["moneyness"] - 1.0).abs().idxmin())
            atm_vol = float(g.loc[idx, "sigma"])
        rows.append({"T": float(T_val), "atm_vol": atm_vol})
    return pd.DataFrame(rows).sort_values("T").reset_index(drop=True)


def _corr_by_expiry_rank(
    get_slice,
    tickers: Iterable[str],
    asof: str,
    max_expiries: int = 6,
    atm_band: float = 0.05,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build an ATM matrix and correlation matrix without using pillars.

    The matrix rows correspond to tickers, columns to expiry ranks (0,1,…).
    Correlations are computed across expiry ranks.
    """
    rows: List[pd.Series] = []
    for t in tickers:
        try:
            df = get_slice(
                t, asof_date=asof, T_target_years=None, call_put=None, nearest_by="T"
            )
        except Exception:
            df = None
        if df is None or df.empty:
            continue
        atm_df = _compute_atm_curve_simple(df, atm_band=atm_band)
        values: Dict[int, float] = {}
        for i in range(max_expiries):
            if i < len(atm_df):
                v = atm_df.at[i, "atm_vol"]
                values[i] = float(v) if pd.notna(v) else np.nan
            else:
                values[i] = np.nan
        rows.append(pd.Series(values, name=t.upper()))
    atm_rank_df = pd.DataFrame(rows)
    if atm_rank_df.empty or len(atm_rank_df.index) < 2:
        corr_df = pd.DataFrame(
            index=atm_rank_df.index, columns=atm_rank_df.index, dtype=float
        )
    else:
        corr_df = atm_rank_df.transpose().corr(method="pearson", min_periods=2)
    return atm_rank_df, corr_df


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
    """Compute unified weights if ``target`` and ``peers`` are provided."""
    if not target or not peers:
        return None
    peers_list = list(peers)
    try:
        return compute_unified_weights(
            target=target,
            peers=peers_list,
            mode=weight_mode,
            asof=asof,
            clip_negative=clip_negative,
            power=weight_power,
            **weight_config,
        )
    except Exception:
        return None


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
    atm_band: float = 0.05,
    show_values: bool = True,
    clip_negative: bool = True,
    weight_power: float = 1.0,
    max_expiries: int = 6,
    weight_mode: str = "corr",
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
    tickers = [t.upper() for t in tickers]
    payload = {
        "tickers": sorted(tickers),
        "asof": pd.to_datetime(asof).floor("min").isoformat(),
        "atm_band": atm_band,
        "max_expiries": max_expiries,
    }

    def _builder() -> Tuple[pd.DataFrame, pd.DataFrame]:
        return _corr_by_expiry_rank(
            get_slice=get_smile_slice,
            tickers=tickers,
            asof=asof,
            max_expiries=max_expiries,
            atm_band=atm_band,
        )

    atm_df, corr_df = compute_or_load("corr", payload, _builder)

    weights = _maybe_compute_weights(
        target=target,
        peers=peers,
        asof=asof,
        weight_mode=weight_mode,
        weight_power=weight_power,
        clip_negative=clip_negative,
        **weight_config,
    )

    plot_correlation_details(
        ax,
        corr_df,
        weights=weights,
        show_values=show_values,
        atm_df=atm_df,
        target=target,
        asof=asof,
        weight_mode=weight_mode,
        max_expiries=max_expiries,
    )
    return atm_df, corr_df, weights


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
    """Count finite expiry-rank ATM observations available for each ticker."""
    if atm_df is None or atm_df.empty:
        return pd.Series(dtype=float)
    coverage = atm_df.apply(pd.to_numeric, errors="coerce").notna().sum(axis=1)
    return coverage.reindex(corr_df.index).fillna(0).astype(int)


def _overlap_counts(
    corr_df: pd.DataFrame,
    atm_df: Optional[pd.DataFrame],
) -> Optional[pd.DataFrame]:
    """Pairwise count of shared finite expiry-rank ATM observations."""
    if atm_df is None or atm_df.empty:
        return None
    aligned = atm_df.reindex(corr_df.index)
    valid = aligned.apply(pd.to_numeric, errors="coerce").notna().astype(int)
    counts = valid.to_numpy() @ valid.to_numpy().T
    return pd.DataFrame(counts, index=aligned.index, columns=aligned.index)


def _split_weight_mode(weight_mode: Optional[str]) -> tuple[str, str]:
    """Return user-facing method and basis labels from a canonical weight mode."""
    mode = str(weight_mode or "").strip().lower()
    if not mode:
        return "not selected", "expiry-rank ATM IV"
    if mode == "oi":
        return "open interest", "contracts"
    if "_" in mode:
        method, basis = mode.split("_", 1)
    else:
        method, basis = mode, "iv_atm"
    method_labels = {
        "corr": "correlation",
        "pca": "PCA",
        "cosine": "cosine",
        "equal": "equal",
        "oi": "open interest",
    }
    basis_labels = {
        "iv_atm": "ATM IV",
        "iv_atm_ranks": "ATM IV expiry ranks",
        "ul": "underlying returns",
        "surface": "IV surface",
        "surface_grid": "IV surface grid",
    }
    return method_labels.get(method, method), basis_labels.get(basis, basis)


def _finite_cell_summary(corr_df: pd.DataFrame) -> tuple[int, int, float]:
    data = corr_df.to_numpy(dtype=float)
    finite_count = int(np.sum(np.isfinite(data)))
    total_elements = int(data.size)
    pct = finite_count / total_elements if total_elements > 0 else 0.0
    return finite_count, total_elements, pct


def _reserve_correlation_layout(
    ax: plt.Axes,
    *,
    has_weights: bool,
    has_coverage: bool,
) -> tuple[plt.Axes, Optional[plt.Axes], Optional[plt.Axes]]:
    """Lay out the heatmap and optional side panels in non-overlapping slots."""
    fig = ax.figure
    if not hasattr(fig, "_orig_position"):
        fig._orig_position = ax.get_position().bounds
        sp = fig.subplotpars
        fig._orig_subplotpars = (sp.left, sp.right, sp.bottom, sp.top)

    if has_weights or has_coverage:
        ax.set_position([0.08, 0.16, 0.62, 0.68])
        main = ax.get_position()
        side_x = main.x1 + 0.045
        side_w = max(0.12, 0.96 - side_x)
        weight_ax = None
        coverage_ax = None
        if has_weights and has_coverage:
            weight_ax = fig.add_axes([side_x, main.y0 + main.height * 0.52, side_w, main.height * 0.36])
            coverage_ax = fig.add_axes([side_x, main.y0 + main.height * 0.04, side_w, main.height * 0.30])
        elif has_weights:
            weight_ax = fig.add_axes([side_x, main.y0 + main.height * 0.24, side_w, main.height * 0.52])
        else:
            coverage_ax = fig.add_axes([side_x, main.y0 + main.height * 0.24, side_w, main.height * 0.52])
        return ax, weight_ax, coverage_ax

    if hasattr(fig, "_orig_position"):
        ax.set_position(fig._orig_position)
    return ax, None, None


def plot_correlation_details(
    ax: plt.Axes,
    corr_df: pd.DataFrame,
    weights: Optional[pd.Series] = None,
    show_values: bool = True,
    *,
    atm_df: Optional[pd.DataFrame] = None,
    target: Optional[str] = None,
    asof: Optional[str] = None,
    weight_mode: Optional[str] = None,
    max_expiries: Optional[int] = None,
) -> None:
    """
    Heatmap of the correlation matrix with optional weight and coverage panels.

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

    data = corr_df.to_numpy(dtype=float)
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

    coverage = _coverage_by_ticker(corr_df, atm_df)
    overlap = _overlap_counts(corr_df, atm_df)
    has_weights = bool(weights is not None and not weights.dropna().empty)
    has_coverage = bool(not coverage.empty)
    _, weight_ax, cov_ax = _reserve_correlation_layout(
        ax,
        has_weights=has_weights,
        has_coverage=has_coverage,
    )
    min_overlap = None
    if overlap is not None and len(overlap) > 1:
        mask = ~np.eye(len(overlap), dtype=bool)
        vals = overlap.to_numpy(dtype=float)[mask]
        vals = vals[np.isfinite(vals)]
        if vals.size:
            min_overlap = int(np.nanmin(vals))

    method_label, basis_label = _split_weight_mode(weight_mode)
    max_exp_label = f"{int(max_expiries)}" if max_expiries else "available"
    context_parts = [f"ATM IV expiry ranks, first {max_exp_label} expiries"]
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
        cbar.set_label("Correlation")
        cbar.ax.tick_params(labelsize=8)
        fig._correlation_colorbar = cbar
        fig._corr_colorbar_ax = cax
    except Exception:
        pass

    ax.set_xticks(range(len(corr_df.columns)))
    ax.set_yticks(range(len(corr_df.index)))
    ax.set_xticklabels(corr_df.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(corr_df.index, fontsize=9)
    title = "Expiry-Rank ATM IV Correlation"
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

    if has_coverage and cov_ax is not None:
        cov = coverage.sort_values(ascending=True)
        cov_ax.barh(range(len(cov)), cov.to_numpy(float), color="#7f8c8d", alpha=0.8)
        cov_ax.set_yticks(range(len(cov)))
        cov_ax.set_yticklabels(cov.index, fontsize=8)
        cov_ax.set_xlim(0, max(float(cov.max()), float(max_expiries or cov.max() or 1)))
        cov_ax.set_title("ATM Coverage", fontsize=10)
        cov_ax.set_xlabel("expiry ranks", fontsize=8)
        cov_ax.tick_params(axis="x", labelsize=8)
        cov_ax.grid(axis="x", alpha=0.25)
        for spine in ("top", "right"):
            cov_ax.spines[spine].set_visible(False)
        for i, v in enumerate(cov.to_numpy(float)):
            cov_ax.text(v, i, f" {int(v)}", va="center", fontsize=8)
        fig._corr_coverage_ax = cov_ax


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
