# analysis/peer_composite_builder.py
from __future__ import annotations
import sqlite3
from typing import Dict, Optional, Iterable, Mapping, Union, Tuple
import pandas as pd
import numpy as np

# For ATM pillar peer-composite IV
from data.db_utils import get_conn
from data.quote_quality import (
    ANALYTICS_MAX_MONEYNESS,
    ANALYTICS_MIN_MONEYNESS,
    filter_quotes,
)
from analysis.pillars import load_atm, nearest_pillars
from analysis.settings import (
    DEFAULT_MONEYNESS_BINS,
    DEFAULT_ATM_BAND,
    DEFAULT_MAX_EXPIRIES,
    DEFAULT_PILLAR_DAYS,
    DEFAULT_PILLAR_TOLERANCE_DAYS,
    DEFAULT_SURFACE_TENORS,
)


# Backward-compatible names used by callers/tests.
DEFAULT_TENORS = DEFAULT_SURFACE_TENORS
DEFAULT_MNY_BINS: Tuple[Tuple[float, float], ...] = DEFAULT_MONEYNESS_BINS


def _nearest_tenor(days: float, tenors: Iterable[int]) -> int:
    arr = np.asarray(list(tenors), dtype=float)
    return int(arr[np.argmin(np.abs(arr - days))])


def _mny_labels(bins: Tuple[Tuple[float, float], ...]) -> Tuple[pd.Interval, ...] | pd.Categorical:
    edges = [bins[0][0]] + [hi for (_, hi) in bins]
    labels = [f"{lo:.2f}-{hi:.2f}" for (lo, hi) in bins]
    return labels, edges


def build_surface_grids(
    tickers: Iterable[str] | None = None,
    tenors: Iterable[int] = DEFAULT_TENORS,
    mny_bins: Tuple[Tuple[float, float], ...] = DEFAULT_MNY_BINS,
    use_atm_only: bool = False,
    max_expiries: Optional[int] = None,
) -> Dict[str, Dict[pd.Timestamp, pd.DataFrame]]:
    """Return dict[ticker][date] -> DataFrame (rows=mny bins, cols=tenor bins) with IV means.

    If tickers is None, use all tickers in DB. If use_atm_only=True, only keep rows with is_atm=1.
    """
    conn = get_conn()

    if tenors is None:
        tenors = DEFAULT_TENORS
    if mny_bins is None:
        mny_bins = DEFAULT_MNY_BINS

    cols = "asof_date, ticker, ttm_years, moneyness, iv, is_atm"
    q = f"SELECT {cols} FROM options_quotes"
    params: list = []
    clauses = []
    tickers_list = list(tickers) if tickers else []
    if tickers_list:
        placeholders = ",".join(["?"] * len(tickers_list))
        clauses.append(f"ticker IN ({placeholders})")
        params.extend(tickers_list)
    if use_atm_only:
        clauses.append("is_atm = ?")
        params.append(1)
    if clauses:
        q += " WHERE " + " AND ".join(clauses)

    df = pd.read_sql_query(q, conn, params=params)
    if df.empty:
        return {}

    df = df.dropna(subset=["iv", "ttm_years", "moneyness"]).copy()
    df = filter_quotes(
        df,
        min_moneyness=ANALYTICS_MIN_MONEYNESS,
        max_moneyness=ANALYTICS_MAX_MONEYNESS,
        require_uncrossed=False,
    )
    if df.empty:
        return {}
    df["ttm_days"] = df["ttm_years"] * 365.25

    # Limit number of expiries if requested
    if max_expiries is not None and max_expiries > 0:
        # Group by ticker and asof_date, then limit expiries for each combination
        limited_dfs = []
        for (ticker, asof_date), group in df.groupby(['ticker', 'asof_date']):
            # Get the closest expiries to today (smallest ttm_years first)
            unique_expiries = group.groupby('ttm_years')['ttm_years'].first().sort_values()
            limited_expiries = unique_expiries.head(max_expiries).values
            limited_group = group[group['ttm_years'].isin(limited_expiries)]
            limited_dfs.append(limited_group)
        if limited_dfs:
            df = pd.concat(limited_dfs, ignore_index=True)
        else:
            return {}

    # Bin to nearest tenor (vectorized)
    tenor_arr = np.asarray(list(tenors), dtype=float)
    ttm_vals = df["ttm_days"].to_numpy(dtype=float)
    idx = np.abs(ttm_vals[:, None] - tenor_arr).argmin(axis=1)
    df["tenor_bin"] = tenor_arr[idx].astype(int)

    # Bin moneyness
    labels, edges = _mny_labels(mny_bins)
    df["mny_bin"] = pd.cut(df["moneyness"], bins=edges, labels=labels, include_lowest=True)
    df = df.dropna(subset=["mny_bin"])

    # Average IV per day/ticker cell
    cell = (
        df.groupby(["asof_date", "ticker", "mny_bin", "tenor_bin"], observed=True)  #
          ["iv"].mean().reset_index()
    )

    # Split into dicts
    out: Dict[str, Dict[pd.Timestamp, pd.DataFrame]] = {}
    for (ticker), g in cell.groupby("ticker"):
        sub: Dict[pd.Timestamp, pd.DataFrame] = {}
        for date, gd in g.groupby("asof_date"):
            grid = gd.pivot(index="mny_bin", columns="tenor_bin", values="iv").sort_index(axis=1)
            sub[pd.to_datetime(date)] = grid
        out[ticker] = sub
    return out
# ============================================================
# 1) Surface-based peer composite (grid combine)
# ============================================================

def combine_surfaces(
    surfaces: Dict[str, Dict[pd.Timestamp, pd.DataFrame]],
    rhos: Mapping[str, float],
    weight_grids: Optional[Dict[str, pd.DataFrame]] = None,
) -> Dict[pd.Timestamp, pd.DataFrame]:
    """
    Combine normalized volatility surfaces across peer tickers into a composite surface.

    Parameters
    ----------
    surfaces : dict
        {ticker -> {date -> DataFrame}} where each DataFrame has index=K (or moneyness)
        and columns=maturity (in days), with cell values = implied vol.
    rhos : mapping
        {ticker -> scalar weight}. These are the top-level per-ticker weights.
        They will be normalized automatically if they don't sum to 1.
    weight_grids : dict, optional
        {ticker -> DataFrame} of same shape as each surface grid to weight
        specific (K, T) points. If omitted, a grid of ones is used per ticker.

    Returns
    -------
    dict
        {date -> DataFrame} representing the peer-composite surface for each date.
    """
    # Normalize top-level weights defensively
    rhos = dict(rhos)
    total = float(sum(rhos.values())) if len(rhos) else 1.0
    if total <= 0:
        # fallback to equal weights
        tickers = list(surfaces.keys())
        rhos = {t: 1.0 / max(1, len(tickers)) for t in tickers}
    else:
        rhos = {k: v / total for k, v in rhos.items()}

    # All dates present across any ticker
    all_dates: set[pd.Timestamp] = set()
    for surf_by_date in surfaces.values():
        all_dates.update(surf_by_date.keys())

    result: Dict[pd.Timestamp, pd.DataFrame] = {}
    for date in sorted(all_dates):
        numerator = None
        denominator = None

        for ticker, surf_by_date in surfaces.items():
            if date not in surf_by_date:
                continue

            sigma = surf_by_date[date]  # DataFrame
            rho = float(rhos.get(ticker, 0.0))
            if rho == 0.0:
                continue

            wg = None
            if weight_grids is not None and ticker in weight_grids:
                wg = weight_grids[ticker]
            if wg is None:
                wg = pd.DataFrame(1.0, index=sigma.index, columns=sigma.columns)

            # Align shapes (defensive)
            wg = wg.reindex_like(sigma).fillna(0.0)

            contrib_num = rho * wg * sigma
            contrib_den = rho * wg

            if numerator is None:
                numerator = contrib_num
                denominator = contrib_den
            else:
                numerator = numerator.add(contrib_num, fill_value=0.0)
                denominator = denominator.add(contrib_den, fill_value=0.0)

        if numerator is not None and denominator is not None:
            # Avoid divide-by-zero: where denominator=0, leave NaN
            with np.errstate(divide="ignore", invalid="ignore"):
                combined = numerator / denominator
            result[date] = combined

    return result


# ============================================================
# 2) ATM-pillar peer-composite IV (series combine)
# ============================================================

def build_synthetic_iv(
    weights: Mapping[str, float],
    pillar_days: Union[int, Iterable[int]] = DEFAULT_PILLAR_DAYS,
    tolerance_days: float = DEFAULT_PILLAR_TOLERANCE_DAYS,
    conn: Optional["sqlite3.Connection"] = None,
) -> pd.DataFrame:
    """
    Build a peer-composite ATM volatility series by pillar from per-ticker ATM IVs.

    Uses your DB-backed ATM selection:
      - loads is_atm=1 quotes via analysis.pillars.load_atm()
      - picks nearest expiries to requested pillar_days via nearest_pillars()
      - combines by provided weights

    Parameters
    ----------
    weights : mapping
        {ticker -> weight}. Will be normalized if they don't sum to 1.
    pillar_days : int or iterable[int]
        Target pillars in days. Defaults to (7,30,60,90,180,365).
    tolerance_days : float
        Max allowed distance (in days) from target pillar to accept a match.
    conn : sqlite3.Connection, optional
        DB connection. If None, uses data.db_utils.get_conn().

    Returns
    -------
    DataFrame
        Columns:
          asof_date (str), pillar_days (int), iv (float),
          tickers_used (int), weighted_count (float),
          iv_constituents (dict ticker->iv), weights (dict ticker->w_normalized)
    """
    if isinstance(pillar_days, int):
        pillar_days = [pillar_days]

    # Normalize weights defensively
    weights = dict(weights)
    total = float(sum(weights.values())) if len(weights) else 1.0
    if total <= 0:
        raise ValueError("All provided weights are zero; supply at least one positive weight.")
    w_norm = {k: v / total for k, v in weights.items()}

    conn = conn or get_conn()
    df_atm = load_atm(conn)  # asof_date, ticker, expiry, ttm_years, iv, spot, moneyness, delta

    if df_atm.empty:
        raise RuntimeError("No ATM rows found. Did you run data.historical_saver?")

    # Keep only tickers we care about
    df_atm = df_atm[df_atm["ticker"].isin(w_norm.keys())].copy()
    if df_atm.empty:
        raise RuntimeError("No ATM rows for the requested weights' tickers.")

    # Pick nearest pillars per (asof_date, ticker)
    pillars = nearest_pillars(df_atm, pillars_days=list(pillar_days), tolerance_days=tolerance_days)
    if pillars.empty:
        raise RuntimeError("nearest_pillars returned no rows within tolerance.")

    # pillars columns: asof_date, ticker, pillar_days, pillar_diff_days, expiry, ttm_years, iv, spot, moneyness, delta
    # Combine by weights per (asof_date, pillar_days)
    out_rows = []
    for (asof, pday), g in pillars.groupby(["asof_date", "pillar_days"]):
        iv_map = {}
        w_sum = 0.0
        iv_wsum = 0.0
        n_used = 0

        for ticker, sub in g.groupby("ticker"):
            if ticker not in w_norm:
                continue
            # Use the row with minimum diff (group is already nearest, but be explicit)
            row = sub.iloc[(sub["pillar_diff_days"].abs()).argsort().values[0]]
            iv_t = float(row["iv"])
            w_t = float(w_norm[ticker])
            iv_map[ticker] = iv_t
            iv_wsum += w_t * iv_t
            w_sum += w_t
            n_used += 1

        if n_used == 0 or w_sum <= 0:
            continue

        out_rows.append(
            {
                "asof_date": asof,
                "pillar_days": int(pday),
                "iv": iv_wsum / w_sum,
                "tickers_used": int(n_used),
                "weighted_count": float(w_sum),
                "iv_constituents": iv_map,
                "weights": w_norm,
            }
        )

    return pd.DataFrame(out_rows).sort_values(["asof_date", "pillar_days"]).reset_index(drop=True)


def build_synthetic_iv_by_rank(
    weights: Mapping[str, float],
    asof: str,
    max_expiries: int = DEFAULT_MAX_EXPIRIES,
    atm_band: float = DEFAULT_ATM_BAND,
) -> pd.DataFrame:
    """Combine peer ATM vols by expiry rank into a composite curve for one date."""
    from analysis.analysis_pipeline import get_smile_slice
    from analysis.correlation_utils import compute_atm_corr_pillar_free

    weights = {k.upper(): float(v) for k, v in dict(weights).items()}
    total = sum(max(0.0, w) for w in weights.values())
    if total <= 0:
        raise ValueError("All weights are zero")
    w_norm = {k: w / total for k, w in weights.items()}
    tickers = list(w_norm.keys())

    atm_df, _ = compute_atm_corr_pillar_free(
        get_smile_slice=get_smile_slice,
        tickers=tickers,
        asof=asof,
        max_expiries=max_expiries,
        atm_band=atm_band,
    )
    if atm_df.empty:
        return pd.DataFrame(columns=["rank", "synth_iv"])

    out = []
    for r in atm_df.columns:
        ivs = []
        ws = []
        for t in tickers:
            if t in atm_df.index:
                v = atm_df.at[t, r]
                if pd.notna(v):
                    ivs.append(float(v))
                    ws.append(float(w_norm.get(t, 0.0)))
        if ws:
            s = float(np.dot(ws, ivs)) / float(np.sum(ws))
            out.append({"rank": int(r), "synth_iv": s})
    return pd.DataFrame(out)


# ============================================================
# __main__ smoke test (optional)
# ============================================================

if __name__ == "__main__":
    # Example: build a 30D peer-composite ATM IV from SPY/QQQ equal weights
    df_syn = build_synthetic_iv({"SPY": 0.5, "QQQ": 0.5}, pillar_days=30)
    print(df_syn.head())
