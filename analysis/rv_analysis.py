"""
Relative Value analysis: surface residuals, skew/curvature spreads,
term-structure shape dislocation, ranked signal generation, and
weight stability tracking.

All public functions return plain pandas DataFrames or plain dicts so
callers are not forced into any custom class.  Heavy imports are delayed
inside functions to avoid circular-import issues with analysis_pipeline.
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional
import numpy as np
import pandas as pd


def _safe_float(val) -> float:
    """Convert *val* to float; return ``np.nan`` if not finite or not convertible."""
    try:
        f = float(val)
        return f if np.isfinite(f) else np.nan
    except (TypeError, ValueError):
        return np.nan


# ---------------------------------------------------------------------------
# Phase 1 — Surface residual
# ---------------------------------------------------------------------------

def compute_surface_residual(
    target_surfaces: Dict[pd.Timestamp, pd.DataFrame],
    synthetic_surfaces: Dict[pd.Timestamp, pd.DataFrame],
    lookback: int = 60,
) -> Dict[pd.Timestamp, pd.DataFrame]:
    """Compute the normalized per-cell residual between target and synthetic surfaces.

    For each date: raw_residual[K, T] = (target[K, T] - synth[K, T]) / synth[K, T].
    When more than one common date exists a rolling z-score over the previous
    ``lookback`` dates is computed per cell; otherwise the raw normalized residual
    is returned.

    Parameters
    ----------
    target_surfaces, synthetic_surfaces : dict[Timestamp -> DataFrame]
        Grids returned by ``build_surface_grids`` / ``combine_surfaces``.
        Rows = moneyness bins, columns = tenor (days).
    lookback : int
        Maximum number of past dates used for the rolling z-score.

    Returns
    -------
    dict[Timestamp -> DataFrame]
        Same moneyness × tenor shape as the inputs (common grid).
        Values are z-scored residuals where history is available, raw
        normalized residuals otherwise.
    """
    common_dates = sorted(
        set(target_surfaces.keys()).intersection(synthetic_surfaces.keys())
    )
    if not common_dates:
        return {}

    # Align all grids to the intersection of rows/cols on the first date
    def _align(tgt: pd.DataFrame, syn: pd.DataFrame):
        rows = tgt.index.intersection(syn.index)
        cols = tgt.columns.intersection(syn.columns)
        if len(rows) < 1 or len(cols) < 1:
            return None, None
        t = tgt.loc[rows, cols].astype(float)
        s = syn.loc[rows, cols].astype(float)
        return t, s

    # Compute raw normalized residual per date
    raw: Dict[pd.Timestamp, pd.DataFrame] = {}
    ref_index = None
    ref_columns = None
    for d in common_dates:
        t, s = _align(target_surfaces[d], synthetic_surfaces[d])
        if t is None:
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(
                np.isfinite(s.to_numpy()) & (s.to_numpy() != 0.0),
                (t.to_numpy() - s.to_numpy()) / s.to_numpy(),
                np.nan,
            )
        raw[d] = pd.DataFrame(r, index=t.index, columns=t.columns)
        if ref_index is None:
            ref_index = t.index
            ref_columns = t.columns

    if not raw:
        return {}

    dates = sorted(raw.keys())
    if len(dates) < 2:
        return dict(raw)

    # Z-score per cell using rolling history
    result: Dict[pd.Timestamp, pd.DataFrame] = {}
    for i, d in enumerate(dates):
        curr_df = raw[d].reindex(index=ref_index, columns=ref_columns)
        hist_start = max(0, i - lookback + 1)
        hist_dates = dates[hist_start : i + 1]
        hist_arrays = np.array(
            [raw[h].reindex(index=ref_index, columns=ref_columns).to_numpy(float)
             for h in hist_dates if h in raw],
            dtype=float,
        )
        if hist_arrays.shape[0] < 2:
            result[d] = curr_df
            continue
        mu = np.nanmean(hist_arrays, axis=0)
        sd = np.nanstd(hist_arrays, axis=0, ddof=1)
        sd = np.where(~np.isfinite(sd) | (sd <= 0.0), 1.0, sd)
        z_arr = (curr_df.to_numpy(float) - mu) / sd
        result[d] = pd.DataFrame(z_arr, index=curr_df.index, columns=curr_df.columns)

    return result


# ---------------------------------------------------------------------------
# Phase 2 — Skew and curvature spread
# ---------------------------------------------------------------------------

def compute_skew_spread(
    target: str,
    peers: Iterable[str],
    asof: str,
    weights: Optional[Mapping[str, float]] = None,
    atm_band: float = 0.05,
    max_expiries: int = 6,
) -> pd.DataFrame:
    """Compute target-vs-synthetic skew and curvature spread per expiry.

    Parameters
    ----------
    target : str
        Ticker of the target asset.
    peers : iterable of str
        Peer tickers used to build the synthetic composite.
    asof : str
        As-of date string (ISO format).
    weights : mapping, optional
        Per-peer weights.  Equal weights used if omitted.
    atm_band : float
        Moneyness band around 1.0 used when extracting ATM vol.
    max_expiries : int
        Maximum number of expiries to load per ticker.

    Returns
    -------
    DataFrame with columns:
        T, T_days,
        target_atm, synth_atm, atm_spread,
        target_skew, synth_skew, skew_spread,
        target_curv, synth_curv, curv_spread
    """
    from analysis.analysis_pipeline import get_smile_slice  # delayed import
    from analysis.pillars import compute_atm_by_expiry

    target = target.upper()
    peers = [p.upper() for p in peers]

    # Normalise weights
    if weights:
        w: Dict[str, float] = {
            k.upper(): float(v)
            for k, v in weights.items()
            if k.upper() in peers and float(v) > 0
        }
    else:
        w = {p: 1.0 for p in peers}
    total = sum(w.values())
    if total <= 0:
        return pd.DataFrame()
    w = {k: v / total for k, v in w.items()}

    # Target ATM curve (with skew/curv)
    df_tgt = get_smile_slice(target, asof, T_target_years=None, max_expiries=max_expiries)
    if df_tgt is None or df_tgt.empty:
        return pd.DataFrame()
    tgt_atm = compute_atm_by_expiry(df_tgt, atm_band=atm_band, method="fit", n_boot=0)
    if tgt_atm.empty or "skew" not in tgt_atm.columns:
        return pd.DataFrame()

    # Per-peer ATM curves
    tol_years = 10.0 / 365.25
    peer_atm: Dict[str, pd.DataFrame] = {}
    for p in peers:
        if w.get(p, 0.0) <= 0:
            continue
        df_p = get_smile_slice(p, asof, T_target_years=None, max_expiries=max_expiries)
        if df_p is None or df_p.empty:
            continue
        c = compute_atm_by_expiry(df_p, atm_band=atm_band, method="fit", n_boot=0)
        if not c.empty and "skew" in c.columns:
            peer_atm[p] = c

    if not peer_atm:
        return pd.DataFrame()

    rows = []
    for _, tgt_row in tgt_atm.iterrows():
        T_val = float(tgt_row["T"])
        target_atm_v = float(tgt_row["atm_vol"]) if pd.notna(tgt_row.get("atm_vol")) else np.nan
        target_skew_v = float(tgt_row["skew"]) if pd.notna(tgt_row.get("skew")) else np.nan
        target_curv_v = float(tgt_row["curv"]) if pd.notna(tgt_row.get("curv")) else np.nan

        sum_w = sum_atm = sum_skew = sum_curv = 0.0
        for p, patm in peer_atm.items():
            p_T = patm["T"].to_numpy(float)
            j = int(np.argmin(np.abs(p_T - T_val)))
            if abs(p_T[j] - T_val) > tol_years:
                continue
            p_row = patm.iloc[j]
            p_w = w.get(p, 0.0)
            if p_w <= 0:
                continue
            if pd.notna(p_row.get("atm_vol")):
                sum_atm += p_w * float(p_row["atm_vol"])
            if pd.notna(p_row.get("skew")):
                sum_skew += p_w * float(p_row["skew"])
            if pd.notna(p_row.get("curv")):
                sum_curv += p_w * float(p_row["curv"])
            sum_w += p_w

        if sum_w <= 0:
            continue

        synth_atm_v = sum_atm / sum_w
        synth_skew_v = sum_skew / sum_w
        synth_curv_v = sum_curv / sum_w

        rows.append({
            "T": T_val,
            "T_days": int(round(T_val * 365.25)),
            "target_atm": target_atm_v,
            "synth_atm": synth_atm_v,
            "atm_spread": (target_atm_v - synth_atm_v) if np.isfinite(target_atm_v) else np.nan,
            "target_skew": target_skew_v,
            "synth_skew": synth_skew_v,
            "skew_spread": (target_skew_v - synth_skew_v) if np.isfinite(target_skew_v) else np.nan,
            "target_curv": target_curv_v,
            "synth_curv": synth_curv_v,
            "curv_spread": (target_curv_v - synth_curv_v) if np.isfinite(target_curv_v) else np.nan,
        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("T").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Phase 3 — Term-structure shape dislocation
# ---------------------------------------------------------------------------

def compute_term_shape_dislocation(
    target: str,
    peers: Iterable[str],
    asof: str,
    weights: Optional[Mapping[str, float]] = None,
    max_expiries: int = 6,
    atm_band: float = 0.05,
) -> Dict[str, float]:
    """Fit a log-linear term structure to target and synthetic; return shape dislocation.

    The model is: ``atm_vol ≈ level + slope * log(T)`` where ``T`` is in years.
    Residuals after subtracting each fitted line are used to identify
    localised event-vol bumps.

    Returns
    -------
    dict with keys:
        target_level, target_slope,
        synth_level, synth_slope,
        level_spread, slope_spread,
        max_event_bump, event_bump_T_days
    Empty dict if there are fewer than 2 aligned expiries.
    """
    spread_df = compute_skew_spread(
        target, peers, asof,
        weights=weights, atm_band=atm_band, max_expiries=max_expiries,
    )
    if spread_df.empty or len(spread_df) < 2:
        return {}

    T = spread_df["T"].to_numpy(float)
    valid = np.isfinite(T) & (T > 0)
    if valid.sum() < 2:
        return {}

    log_T = np.log(T[valid])
    tgt_atm = spread_df["target_atm"].to_numpy(float)[valid]
    syn_atm = spread_df["synth_atm"].to_numpy(float)[valid]

    def _fit_log_linear(log_t, iv):
        finite = np.isfinite(log_t) & np.isfinite(iv)
        if finite.sum() < 2:
            return np.nan, np.nan, np.full(len(log_t), np.nan)
        X = np.column_stack([np.ones(finite.sum()), log_t[finite]])
        try:
            coef, _, _, _ = np.linalg.lstsq(X, iv[finite], rcond=None)
        except Exception:
            return np.nan, np.nan, np.full(len(log_t), np.nan)
        level, slope = float(coef[0]), float(coef[1])
        fitted_all = coef[0] + coef[1] * log_t
        residuals = np.where(np.isfinite(iv), iv - fitted_all, np.nan)
        return level, slope, residuals

    tgt_level, tgt_slope, tgt_resid = _fit_log_linear(log_T, tgt_atm)
    syn_level, syn_slope, syn_resid = _fit_log_linear(log_T, syn_atm)

    bump_diff = tgt_resid - syn_resid
    finite_bump = np.isfinite(bump_diff)
    max_event_bump = float(np.nanmax(np.abs(bump_diff))) if finite_bump.any() else np.nan
    if finite_bump.any() and np.isfinite(max_event_bump):
        bump_idx = int(np.nanargmax(np.abs(np.where(finite_bump, bump_diff, np.nan))))
        event_bump_T_days = float(T[valid][bump_idx] * 365.25)
    else:
        event_bump_T_days = np.nan

    return {
        "target_level": tgt_level,
        "target_slope": tgt_slope,
        "synth_level": syn_level,
        "synth_slope": syn_slope,
        "level_spread": (tgt_level - syn_level)
        if (np.isfinite(tgt_level) and np.isfinite(syn_level))
        else np.nan,
        "slope_spread": (tgt_slope - syn_slope)
        if (np.isfinite(tgt_slope) and np.isfinite(syn_slope))
        else np.nan,
        "max_event_bump": max_event_bump,
        "event_bump_T_days": event_bump_T_days,
    }


# ---------------------------------------------------------------------------
# Phase 4 — Ranked RV signals
# ---------------------------------------------------------------------------

def generate_rv_signals(
    target: str,
    peers: Iterable[str],
    asof: Optional[str] = None,
    weight_mode: str = "corr_iv_atm",
    lookback: int = 60,
    max_expiries: int = 6,
    min_abs_z: float = 1.0,
) -> pd.DataFrame:
    """Generate a ranked DataFrame of implied-volatility dislocation signals.

    Combines three signal sources:

    1. **ATM level** – per-pillar z-scores from the historical
       target-vs-synthetic spread, via the existing
       ``relative_value_atm_report_corrweighted`` infrastructure.
    2. **Skew / Curvature** – per-expiry spread between target and
       synthetic skew / curvature (no historical z-score; single-date).
    3. **Term structure shape** – level, slope, and max event-bump
       from the log-linear term-structure dislocation.

    Parameters
    ----------
    target : str
    peers : iterable of str
    asof : str, optional
        ISO date string.  Uses most-recent DB date when omitted.
    weight_mode : str
        Canonical weight-mode string (e.g. ``"corr_iv_atm"``).
    lookback : int
        Rolling window (days) for ATM-level z-score.
    max_expiries : int
    min_abs_z : float
        Only ATM-level signals with ``|z| >= min_abs_z`` are kept.
        Skew/curvature and shape signals always pass through.

    Returns
    -------
    DataFrame with columns:
        signal_type, asof_date, T_days, value, synth_value,
        spread, z_score, pct_rank, description
    Sorted descending by |z_score| (NaN z-score signals come last).
    """
    from analysis.analysis_pipeline import (  # delayed to avoid circular import
        compute_peer_weights,
        get_most_recent_date_global,
        relative_value_atm_report_corrweighted,
    )

    target = target.upper()
    peers = [p.upper() for p in peers]

    if asof is None:
        asof = get_most_recent_date_global()
    if not asof:
        return pd.DataFrame(columns=[
            "signal_type", "asof_date", "T_days", "value", "synth_value",
            "spread", "z_score", "pct_rank", "description",
        ])

    # Compute weights
    try:
        weights = compute_peer_weights(target, peers, weight_mode=weight_mode)
    except Exception:
        eq = 1.0 / max(len(peers), 1)
        weights = pd.Series(eq, index=peers, dtype=float)
    w_dict = weights.to_dict()

    signals: list[dict] = []

    # ---- 1. ATM level (z-scored via historical spread) ----
    try:
        rv_df, _ = relative_value_atm_report_corrweighted(
            target=target,
            peers=peers,
            mode=weight_mode,
            pillar_days=[7, 14, 30, 60, 90],
            lookback=lookback,
        )
        if not rv_df.empty:
            latest = rv_df.sort_values("asof_date").groupby("pillar_days").tail(1)
            for _, row in latest.iterrows():
                z_f = _safe_float(row.get("z", np.nan))
                if not np.isfinite(z_f):
                    continue
                spread_v = row.get("spread", np.nan)
                iv_tgt = row.get("iv_target", np.nan)
                iv_syn = row.get("iv_synth", np.nan)
                pct = row.get("pct_rank", np.nan)
                T_days = int(row.get("pillar_days", 0))
                signals.append({
                    "signal_type": "ATM Level",
                    "asof_date": str(row.get("asof_date", asof)),
                    "T_days": T_days,
                    "value": _safe_float(iv_tgt),
                    "synth_value": _safe_float(iv_syn),
                    "spread": _safe_float(spread_v),
                    "z_score": z_f,
                    "pct_rank": _safe_float(pct),
                    "description": f"{target} ATM vol vs synthetic at {T_days}d",
                })
    except Exception:
        pass

    # ---- 2. Skew and curvature ----
    try:
        skew_df = compute_skew_spread(
            target, peers, asof, weights=w_dict, max_expiries=max_expiries
        )
        if not skew_df.empty:
            for _, row in skew_df.iterrows():
                T_days = int(row.get("T_days", int(round(row.get("T", 0) * 365.25))))
                for sig_type, (tgt_col, syn_col, spread_col, desc_suffix) in {
                    "Skew": ("target_skew", "synth_skew", "skew_spread", "put/call skew"),
                    "Curvature": ("target_curv", "synth_curv", "curv_spread", "vol curvature"),
                }.items():
                    s_f = _safe_float(row.get(spread_col, np.nan))
                    if not np.isfinite(s_f):
                        continue
                    signals.append({
                        "signal_type": sig_type,
                        "asof_date": asof,
                        "T_days": T_days,
                        "value": _safe_float(row.get(tgt_col, np.nan)),
                        "synth_value": _safe_float(row.get(syn_col, np.nan)),
                        "spread": s_f,
                        "z_score": np.nan,
                        "pct_rank": np.nan,
                        "description": f"{target} {desc_suffix} vs synthetic at {T_days}d",
                    })
    except Exception:
        pass

    # ---- 3. Term structure shape ----
    try:
        shape = compute_term_shape_dislocation(
            target, peers, asof, weights=w_dict, max_expiries=max_expiries
        )
        if shape:
            for sig_type, val_key, syn_key, spread_key, desc in [
                ("TS Level", "target_level", "synth_level", "level_spread",
                 f"{target} term-structure level vs synthetic"),
                ("TS Slope", "target_slope", "synth_slope", "slope_spread",
                 f"{target} term-structure slope vs synthetic"),
            ]:
                sv = shape.get(spread_key, np.nan)
                if not np.isfinite(sv):
                    continue
                signals.append({
                    "signal_type": sig_type,
                    "asof_date": asof,
                    "T_days": 0,
                    "value": float(shape.get(val_key, np.nan)),
                    "synth_value": float(shape.get(syn_key, np.nan)),
                    "spread": float(sv),
                    "z_score": np.nan,
                    "pct_rank": np.nan,
                    "description": desc,
                })
            bump = shape.get("max_event_bump", np.nan)
            bump_T = shape.get("event_bump_T_days", np.nan)
            if np.isfinite(bump) and bump > 0:
                T_label = int(bump_T) if np.isfinite(bump_T) else "?"
                signals.append({
                    "signal_type": "Event Bump",
                    "asof_date": asof,
                    "T_days": int(bump_T) if np.isfinite(bump_T) else 0,
                    "value": float(bump),
                    "synth_value": 0.0,
                    "spread": float(bump),
                    "z_score": np.nan,
                    "pct_rank": np.nan,
                    "description": f"{target} event-vol bump vs synthetic at {T_label}d",
                })
    except Exception:
        pass

    if not signals:
        return pd.DataFrame(columns=[
            "signal_type", "asof_date", "T_days", "value", "synth_value",
            "spread", "z_score", "pct_rank", "description",
        ])

    df = pd.DataFrame(signals)

    # Separate z-scored from non-z-scored signals
    has_z = df["z_score"].apply(lambda z: pd.notna(z) and np.isfinite(float(z)))
    z_signals = df[has_z].copy()
    other_signals = df[~has_z].copy()

    # Filter z-scored by min_abs_z
    if not z_signals.empty:
        z_signals = z_signals[z_signals["z_score"].abs() >= min_abs_z]

    # Sort each group: z-scored by |z|, others by |spread|
    if not z_signals.empty:
        z_signals = z_signals.sort_values("z_score", key=lambda s: s.abs(), ascending=False)
    if not other_signals.empty:
        other_signals = other_signals.sort_values(
            "spread", key=lambda s: s.abs(), ascending=False
        )

    return pd.concat([z_signals, other_signals], ignore_index=True)


# ---------------------------------------------------------------------------
# Phase 5 — Weight stability
# ---------------------------------------------------------------------------

def compute_weight_stability(
    target: str,
    peers: Iterable[str],
    lookback: int = 30,
    asof: Optional[str] = None,
) -> pd.DataFrame:
    """Compute rolling IV correlation between target and each peer.

    Uses the daily ATM IV series stored in the database.  Returns a
    DataFrame (rows = peers) with columns ``rolling_corr`` and ``stable``
    (``True`` when ``rolling_corr >= 0.3``).

    Parameters
    ----------
    target : str
    peers : iterable of str
    lookback : int
        Number of most-recent trading days to use.
    asof : str, optional
        Ignored — function always uses the most-recent ``lookback`` dates.

    Returns
    -------
    DataFrame  (index = peer ticker)
    """
    from analysis.analysis_pipeline import get_daily_iv_for_spillover  # delayed

    target = target.upper()
    peers = [p.upper() for p in peers]
    all_tickers = [target] + peers

    empty = pd.DataFrame(
        {"rolling_corr": np.nan, "stable": False},
        index=peers,
    )

    try:
        iv_df = get_daily_iv_for_spillover(tickers=all_tickers)
    except Exception:
        return empty

    if iv_df.empty:
        return empty

    piv = iv_df.pivot(index="date", columns="ticker", values="atm_iv").sort_index()
    piv = piv.iloc[-lookback:] if len(piv) > lookback else piv

    if target not in piv.columns or len(piv) < 5:
        return empty

    tgt_series = piv[target]
    rows: dict = {}
    for p in peers:
        if p not in piv.columns:
            rows[p] = {"rolling_corr": np.nan, "stable": False}
            continue
        common = tgt_series.dropna().index.intersection(piv[p].dropna().index)
        if len(common) < 5:
            rows[p] = {"rolling_corr": np.nan, "stable": False}
            continue
        corr = float(tgt_series.loc[common].corr(piv[p].loc[common]))
        rows[p] = {"rolling_corr": corr, "stable": bool(corr >= 0.3)}

    return pd.DataFrame(rows).T


__all__ = [
    "compute_surface_residual",
    "compute_skew_spread",
    "compute_term_shape_dislocation",
    "generate_rv_signals",
    "compute_weight_stability",
]
