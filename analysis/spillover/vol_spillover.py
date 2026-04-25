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


def summarise(responses: pd.DataFrame, threshold: float = DEFAULT_SPILLOVER_EVENT_THRESHOLD) -> pd.DataFrame:
    """Summarise peer responses across events."""
    def _agg(g: pd.DataFrame) -> pd.Series:
        hr = (g["peer_pct"].abs() >= threshold).mean()
        sc = (np.sign(g["peer_pct"]) == g["sign"]).mean()
        med_resp = g["peer_pct"].median()
        med_elast = (g["peer_pct"] / g["trigger_pct"]).median()
        return pd.Series({
            "hit_rate": hr,
            "sign_concord": sc,
            "median_resp": med_resp,
            "median_elasticity": med_elast,
            "n": len(g),
        })
    return responses.groupby(["ticker", "peer", "h"]).apply(_agg).reset_index()


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
    summary = summarise(responses, threshold=threshold)
    persist_events(events, events_path)
    persist_summary(summary, summary_path)
    return {"events": events, "responses": responses, "summary": summary}
