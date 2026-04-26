import pandas as pd
import numpy as np
import sqlite3
from typing import List, Dict, Iterable, Callable, Tuple, Optional
from data.ticker_groups import get_groups_for_target, load_ticker_group
from analysis.settings import (
    DEFAULT_SPILLOVER_EVENT_THRESHOLD,
    DEFAULT_SPILLOVER_HORIZONS,
    DEFAULT_SPILLOVER_LOOKBACK_DAYS,
    DEFAULT_SPILLOVER_REGRESSION_WINDOW_DAYS,
)

"""Tools to detect implied-volatility events and measure spillovers.

This module loads a daily IV dataset, flags events where a ticker's ATM IV
moves by a configurable percentage threshold and then measures how those shocks
propagate to its peers.
"""


def load_iv_data(path: str, use_raw: bool = False) -> pd.DataFrame:
    """Load IV data from a Parquet file.

    Parameters
    ----------
    path: str
        Location of the ``iv_daily`` Parquet file.
    use_raw: bool
        If ``True`` use the ``atm_iv_raw`` column, otherwise use
        ``atm_iv_synth``.
    """
    df = pd.read_parquet(path)
    col = "atm_iv_raw" if use_raw else "atm_iv_synth"
    df = df[["date", "ticker", col]].rename(columns={col: "atm_iv"})
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["ticker", "date"])


def detect_events(df: pd.DataFrame, threshold: float = DEFAULT_SPILLOVER_EVENT_THRESHOLD) -> pd.DataFrame:
    """Flag dates where a ticker's IV changes by ``threshold`` or more.

    Returns a DataFrame with columns ``ticker``, ``date``, ``rel_change`` and
    ``sign`` (1 or -1).
    """
    df = df.sort_values(["ticker", "date"]).copy()
    df["rel_change"] = df.groupby("ticker")["atm_iv"].pct_change()
    events = df.loc[df["rel_change"].abs() >= threshold,
                    ["ticker", "date", "rel_change"]].copy()
    events["sign"] = np.sign(events["rel_change"]).astype(int)
    return events.reset_index(drop=True)


def _load_peers_for_target(target: str, conn=None) -> List[str]:
    """Return peer tickers for a target using stored ticker groups."""
    groups = get_groups_for_target(target, conn)
    peers: List[str] = []
    for name in groups:
        grp = load_ticker_group(name, conn)
        if grp:
            peers.extend(grp.get("peer_tickers", []))
    # Deduplicate and exclude the target itself
    uniq = {p.upper() for p in peers if p and p.upper() != target.upper()}
    return sorted(uniq)


def compute_weights_and_regression(
    df: pd.DataFrame,
    target: str,
    window: int = DEFAULT_SPILLOVER_REGRESSION_WINDOW_DAYS,
    conn: Optional[sqlite3.Connection] = None,
) -> Tuple[pd.Series, pd.Series]:
    """Compute peer weights and regression betas from historical IV data.

    Parameters
    ----------
    df : pd.DataFrame
        Table with columns ``date``, ``ticker`` and ``atm_iv``.
    target : str
        Ticker for the target security. Peers are looked up via
        :func:`data.ticker_groups.get_groups_for_target`.
    window : int, default 90
        Number of trailing calendar days to use when computing statistics.
    conn : sqlite3.Connection, optional
        Connection to the ticker groups database. If ``None`` the default
        connection from :func:`data.ticker_groups.get_conn` is used.

    Returns
    -------
    tuple(pd.Series, pd.Series)
        ``weights`` : normalised non-negative weights for each peer based on
        correlation with the target.
        ``betas`` : regression slopes of peer IV returns on target IV returns.

    Notes
    -----
    ``atm_iv`` is converted to log returns before computing correlations and
    regressions.  Negative weights are clipped to zero prior to normalisation.
    """

    if df.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    end = df["date"].max()
    start = end - pd.Timedelta(days=window)
    df = df[(df["date"] > start) & (df["date"] <= end)]

    piv = df.pivot(index="date", columns="ticker", values="atm_iv").sort_index()
    piv = piv.replace([np.inf, -np.inf], np.nan)
    piv = piv.where(piv > 0.0)
    ivret = np.log(piv).diff().dropna(how="all")

    tgt = target.upper()
    peer_list = _load_peers_for_target(tgt, conn)
    if tgt not in ivret.columns:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    x = ivret[tgt]
    corrs = {}
    betas = {}
    for peer in peer_list:
        if peer not in ivret.columns:
            continue
        y = ivret[peer]
        pair = pd.concat([x, y], axis=1, keys=["x", "y"]).dropna()
        if len(pair) < 2:
            continue
        corrs[peer] = pair["x"].corr(pair["y"])
        denom = (pair["x"] ** 2).sum()
        betas[peer] = float(np.dot(pair["x"], pair["y"]) / denom) if denom > 0 else np.nan

    weights = pd.Series(corrs).clip(lower=0.0)
    if not weights.empty and float(weights.sum()) > 0:
        weights = weights / float(weights.sum())

    return weights, pd.Series(betas)


def compute_responses(df: pd.DataFrame,
                      events: pd.DataFrame,
                      peers: Dict[str, List[str]],
                      horizons: Iterable[int] = DEFAULT_SPILLOVER_HORIZONS) -> pd.DataFrame:
    """Compute peer responses for each event over given horizons.

    Response for peer j at horizon ``h`` is the percentage change in j's IV
    from ``t0-1`` to ``t0+h``.  This means ``h=1`` measures the response on the
    day *after* the trigger event and ``h=0`` would correspond to the trigger
    day itself.  The previous implementation incorrectly used ``t0+h-1`` which
    shifted all horizons one day earlier and was incompatible with other parts
    of the program that expect horizons to be offset from the event date.
    """
    panel = df.set_index(["date", "ticker"]).sort_index()
    dates = panel.index.get_level_values(0).unique()
    rows = []
    for _, e in events.iterrows():
        t0 = e["date"]
        i = e["ticker"]
        idx0 = dates.searchsorted(t0)
        if idx0 == 0:
            continue  # need t-1
        t_minus1 = dates[idx0 - 1]
        for j in peers.get(i, []):
            if (t_minus1, j) not in panel.index:
                continue
            base = panel.loc[(t_minus1, j), "atm_iv"]
            for h in horizons:
                # Use t0 + h to express the response h days after the event.
                idx_h = idx0 + h
                if idx_h >= len(dates):
                    continue
                d_h = dates[idx_h]
                if (d_h, j) not in panel.index:
                    continue
                resp = panel.loc[(d_h, j), "atm_iv"]
                pct = (resp - base) / base
                rows.append({
                    "ticker": i,
                    "peer": j,
                    "t0": t0,
                    "h": int(h),
                    "trigger_pct": e["rel_change"],
                    "peer_pct": pct,
                    "sign": e["sign"],
                })
    return pd.DataFrame(rows)


def _baseline_responses(
    df: pd.DataFrame,
    pairs: pd.DataFrame,
    horizons: Iterable[int],
) -> pd.DataFrame:
    """Build randomized-date peer responses for each trigger-peer pair.

    This keeps the same response definition as ``compute_responses`` but uses
    every available date as a pseudo trigger date.  It provides the null
    distribution for baseline-adjusted medians and permutation p-values.
    """
    if df.empty or pairs.empty:
        return pd.DataFrame(columns=["ticker", "peer", "h", "peer_pct"])

    panel = df.set_index(["date", "ticker"]).sort_index()
    dates = panel.index.get_level_values(0).unique()
    rows = []
    uniq_pairs = pairs[["ticker", "peer"]].drop_duplicates()
    for _, pair in uniq_pairs.iterrows():
        ticker = pair["ticker"]
        peer = pair["peer"]
        for idx0, _date in enumerate(dates):
            if idx0 == 0:
                continue
            t_minus1 = dates[idx0 - 1]
            if (t_minus1, peer) not in panel.index:
                continue
            base = panel.loc[(t_minus1, peer), "atm_iv"]
            if not np.isfinite(base) or base <= 0:
                continue
            for h in horizons:
                idx_h = idx0 + int(h)
                if idx_h >= len(dates):
                    continue
                d_h = dates[idx_h]
                if (d_h, peer) not in panel.index:
                    continue
                resp = panel.loc[(d_h, peer), "atm_iv"]
                if not np.isfinite(resp):
                    continue
                rows.append({
                    "ticker": ticker,
                    "peer": peer,
                    "h": int(h),
                    "peer_pct": float((resp - base) / base),
                })
    return pd.DataFrame(rows)


def _bootstrap_median_ci(values: np.ndarray, *, n_boot: int, rng: np.random.Generator) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    if values.size == 1 or n_boot <= 0:
        med = float(np.median(values))
        return med, med
    samples = rng.choice(values, size=(int(n_boot), values.size), replace=True)
    meds = np.median(samples, axis=1)
    return float(np.quantile(meds, 0.025)), float(np.quantile(meds, 0.975))


def _permutation_median_pvalue(
    actual_median: float,
    baseline_values: np.ndarray,
    *,
    n: int,
    n_perm: int,
    rng: np.random.Generator,
) -> float:
    baseline_values = np.asarray(baseline_values, dtype=float)
    baseline_values = baseline_values[np.isfinite(baseline_values)]
    if baseline_values.size == 0 or n <= 0 or n_perm <= 0 or not np.isfinite(actual_median):
        return np.nan
    baseline_median = float(np.median(baseline_values))
    actual_stat = abs(float(actual_median) - baseline_median)
    samples = rng.choice(baseline_values, size=(int(n_perm), int(n)), replace=True)
    null_stats = np.abs(np.median(samples, axis=1) - baseline_median)
    return float((1 + np.sum(null_stats >= actual_stat)) / (int(n_perm) + 1))


def _benjamini_hochberg(pvals: pd.Series) -> pd.Series:
    p = pd.to_numeric(pvals, errors="coerce")
    q = pd.Series(np.nan, index=p.index, dtype=float)
    valid = p.dropna().clip(lower=0.0, upper=1.0)
    m = len(valid)
    if m == 0:
        return q
    order = valid.sort_values().index
    ranked = valid.loc[order].to_numpy(float)
    adj = ranked * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    q.loc[order] = np.clip(adj, 0.0, 1.0)
    return q


def _strength_label(row: pd.Series) -> str:
    n = int(row.get("n", 0) or 0)
    if n < 5:
        return "Insufficient N"
    q = row.get("q_value", np.nan)
    p = row.get("p_value", np.nan)
    hit = float(row.get("hit_rate", 0.0) or 0.0)
    sign = float(row.get("sign_concord", 0.0) or 0.0)
    abnormal = abs(float(row.get("median_abnormal_resp", 0.0) or 0.0))
    if np.isfinite(q) and q <= 0.10 and hit >= 0.70 and sign >= 0.70 and abnormal > 0.0:
        return "Strong"
    if np.isfinite(p) and p <= 0.10 and hit >= 0.60 and sign >= 0.60 and abnormal > 0.0:
        return "Suggestive"
    return "Weak"


def summarise(
    responses: pd.DataFrame,
    threshold: float = DEFAULT_SPILLOVER_EVENT_THRESHOLD,
    *,
    baseline: pd.DataFrame | None = None,
    n_boot: int = 500,
    n_perm: int = 1000,
    random_state: int = 0,
) -> pd.DataFrame:
    """Summarise peer responses across events."""
    if responses.empty:
        return pd.DataFrame(
            columns=[
                "ticker", "peer", "h", "hit_rate", "sign_concord",
                "median_resp", "median_resp_ci_low", "median_resp_ci_high",
                "baseline_median_resp", "median_abnormal_resp", "p_value",
                "q_value", "strength", "median_elasticity", "n",
            ]
        )
    rng = np.random.default_rng(random_state)
    baseline = baseline if baseline is not None else pd.DataFrame(columns=["ticker", "peer", "h", "peer_pct"])

    rows = []
    for group_key, g in responses.groupby(["ticker", "peer", "h"]):
        hr = (g["peer_pct"].abs() >= threshold).mean()
        sc = (np.sign(g["peer_pct"]) == g["sign"]).mean()
        med_resp = g["peer_pct"].median()
        med_elast = (g["peer_pct"] / g["trigger_pct"]).median()
        b = baseline[
            (baseline["ticker"] == group_key[0])
            & (baseline["peer"] == group_key[1])
            & (baseline["h"] == group_key[2])
        ]["peer_pct"].to_numpy(float)
        baseline_med = float(np.nanmedian(b)) if b.size else np.nan
        ci_low, ci_high = _bootstrap_median_ci(g["peer_pct"].to_numpy(float), n_boot=n_boot, rng=rng)
        p_value = _permutation_median_pvalue(
            float(med_resp),
            b,
            n=len(g),
            n_perm=n_perm,
            rng=rng,
        )
        rows.append({
            "ticker": group_key[0],
            "peer": group_key[1],
            "h": group_key[2],
            "hit_rate": hr,
            "sign_concord": sc,
            "median_resp": med_resp,
            "median_resp_ci_low": ci_low,
            "median_resp_ci_high": ci_high,
            "baseline_median_resp": baseline_med,
            "median_abnormal_resp": med_resp - baseline_med if np.isfinite(baseline_med) else np.nan,
            "p_value": p_value,
            "median_elasticity": med_elast,
            "n": len(g),
        })
    summary = pd.DataFrame(rows)
    summary["q_value"] = _benjamini_hochberg(summary["p_value"])
    summary["strength"] = summary.apply(_strength_label, axis=1)
    return summary


def persist_events(events: pd.DataFrame, path: str) -> None:
    """Write event table to Parquet."""
    events.to_parquet(path)


def persist_summary(summary: pd.DataFrame, path: str) -> None:
    """Write summary metrics to Parquet."""
    summary.to_parquet(path)


def run_spillover(
    source: pd.DataFrame | Callable[[], pd.DataFrame],
    *,
    tickers: Iterable[str] | None = None,
    threshold: float = DEFAULT_SPILLOVER_EVENT_THRESHOLD,
    lookback: int = DEFAULT_SPILLOVER_LOOKBACK_DAYS,
    top_k: int = 3,
    horizons: Iterable[int] = DEFAULT_SPILLOVER_HORIZONS,
    events_path: str = "spill_events.parquet",
    summary_path: str = "spill_summary.parquet",
) -> Dict[str, pd.DataFrame]:
    """High level helper that runs the full spillover analysis.

    Parameters
    ----------
    source : DataFrame or callable returning DataFrame
        Either a preloaded IV data table or a function that yields one.
        This allows callers to supply data from any source (e.g. Parquet,
        database, API) without ``run_spillover`` needing to know the details.
    lookback : int, optional
        Retained for backward compatibility; peer selection now relies on
        pre-defined groups rather than historical correlations.
    top_k : int, optional
        Retained for backward compatibility and ignored.

    Peer relationships are obtained from ``data.ticker_groups`` via
    :func:`get_groups_for_target`.

    Returns
    -------
    dict
        Dictionary with keys ``events``, ``responses`` and ``summary``.
    """
    df = source() if callable(source) else source
    if tickers is not None:
        tickers = [t.upper() for t in tickers]
        df = df[df["ticker"].str.upper().isin(tickers)]
    events = detect_events(df, threshold=threshold)
    tick_set = events["ticker"].unique()
    peers = {t: _load_peers_for_target(t) for t in tick_set}
    responses = compute_responses(df, events, peers, horizons=horizons)
    baseline = _baseline_responses(df, responses[["ticker", "peer"]].drop_duplicates(), horizons)
    summary = summarise(responses, threshold=threshold, baseline=baseline)
    persist_events(events, events_path)
    persist_summary(summary, summary_path)
    return {"events": events, "responses": responses, "summary": summary}
