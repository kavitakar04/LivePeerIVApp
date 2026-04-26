"""Graph adapters for spillover/correlation analytics.

This module intentionally sits on top of existing DataFrame outputs so callers can
adopt graph analysis incrementally without changing core event/correlation code.
"""

from __future__ import annotations

from typing import Any

import math

import pandas as pd

try:
    import networkx as nx
except ImportError as exc:  # pragma: no cover - exercised when dependency missing
    nx = None
    _NETWORKX_IMPORT_ERROR = exc
else:
    _NETWORKX_IMPORT_ERROR = None


SPILLOVER_REQUIRED_COLUMNS = {
    "ticker",
    "peer",
    "h",
    "n",
    "hit_rate",
    "sign_concord",
    "median_elasticity",
    "median_resp",
}


def _require_networkx() -> None:
    if nx is None:
        raise ImportError(
            "networkx is required for graph adapters. Install with `pip install networkx`."
        ) from _NETWORKX_IMPORT_ERROR


def _validate_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def build_spillover_digraph(
    summary_df: pd.DataFrame,
    horizon: int | None = None,
    min_n: int = 10,
    min_hit_rate: float = 0.0,
    weight_col: str = "median_elasticity",
    include_negative: bool = True,
):
    """Build a directed graph from spillover summary output.

    Parameters
    ----------
    summary_df
        DataFrame from :func:`analysis.spillover.vol_spillover.summarise`.
    horizon
        If provided, keep only rows for this horizon.
    min_n
        Minimum sample size per edge.
    min_hit_rate
        Minimum hit rate per edge.
    weight_col
        Column to map into the edge's ``weight`` attribute.
    include_negative
        If False, drop rows where ``weight_col`` is negative.
    """
    _require_networkx()
    _validate_columns(summary_df, SPILLOVER_REQUIRED_COLUMNS, "summary_df")
    if weight_col not in summary_df.columns:
        raise ValueError(f"weight_col '{weight_col}' not present in summary_df")

    df = summary_df.copy()
    if horizon is not None:
        df = df[df["h"] == int(horizon)]

    df[weight_col] = pd.to_numeric(df[weight_col], errors="coerce")
    df = df[
        (df["n"] >= int(min_n))
        & (df["hit_rate"] >= float(min_hit_rate))
        & df[weight_col].map(lambda x: math.isfinite(float(x)) if pd.notna(x) else False)
    ].copy()
    if not include_negative:
        df = df[df[weight_col] >= 0.0]

    G = nx.DiGraph()
    for _, row in df.iterrows():
        src = str(row["ticker"])
        dst = str(row["peer"])
        weight = float(row[weight_col])
        abs_weight = abs(weight)
        if abs_weight <= 0.0:
            continue
        attrs: dict[str, Any] = {
            "weight": weight,
            "abs_weight": abs_weight,
            # NetworkX shortest-path centrality treats this as a cost, not a
            # strength. Stronger spillovers should therefore be shorter paths.
            "distance": 1.0 / abs_weight,
            "h": int(row["h"]),
            "n": int(row["n"]),
            "hit_rate": float(row["hit_rate"]),
            "sign_concord": float(row["sign_concord"]),
            "median_elasticity": float(row["median_elasticity"]),
            "median_resp": float(row["median_resp"]),
        }
        G.add_edge(src, dst, **attrs)

    return G


def build_corr_graph(
    corr_df: pd.DataFrame,
    min_abs_corr: float = 0.3,
    include_diagonal: bool = False,
):
    """Build an undirected weighted graph from a correlation matrix."""
    _require_networkx()

    if corr_df.empty:
        return nx.Graph()
    if list(corr_df.index) != list(corr_df.columns):
        raise ValueError("corr_df must have matching index/columns ticker labels")

    labels = [str(x) for x in corr_df.index]
    G = nx.Graph()
    G.add_nodes_from(labels)

    for i, src in enumerate(labels):
        j_start = i if include_diagonal else i + 1
        for j in range(j_start, len(labels)):
            dst = labels[j]
            corr = corr_df.iloc[i, j]
            if pd.isna(corr):
                continue
            corr = float(corr)
            if abs(corr) < float(min_abs_corr):
                continue
            abs_corr = abs(corr)
            if abs_corr <= 0.0:
                continue
            G.add_edge(
                src,
                dst,
                weight=abs_corr,
                abs_weight=abs_corr,
                distance=1.0 / abs_corr,
                corr=corr,
                sign=1 if corr >= 0 else -1,
            )

    return G


def compute_graph_metrics(G) -> pd.DataFrame:
    """Return a DataFrame of basic node-level graph metrics."""
    _require_networkx()

    if len(G) == 0:
        return pd.DataFrame(
            columns=[
                "node",
                "degree",
                "degree_centrality",
                "betweenness_centrality",
                "in_degree",
                "out_degree",
                "in_strength",
                "out_strength",
                "in_signed_strength",
                "out_signed_strength",
            ]
        )

    degree_c = nx.degree_centrality(G)
    betweenness_weight = "distance" if nx.get_edge_attributes(G, "distance") else None
    betweenness = nx.betweenness_centrality(G, weight=betweenness_weight, normalized=True)

    rows = []
    directed = G.is_directed()
    for node in sorted(G.nodes()):
        row: dict[str, Any] = {
            "node": node,
            "degree": float(G.degree(node)),
            "degree_centrality": float(degree_c.get(node, 0.0)),
            "betweenness_centrality": float(betweenness.get(node, 0.0)),
        }

        if directed:
            row["in_degree"] = float(G.in_degree(node))
            row["out_degree"] = float(G.out_degree(node))
            row["in_strength"] = float(G.in_degree(node, weight="abs_weight"))
            row["out_strength"] = float(G.out_degree(node, weight="abs_weight"))
            row["in_signed_strength"] = float(G.in_degree(node, weight="weight"))
            row["out_signed_strength"] = float(G.out_degree(node, weight="weight"))
        else:
            strength = float(G.degree(node, weight="abs_weight"))
            row["in_degree"] = float("nan")
            row["out_degree"] = float("nan")
            row["in_strength"] = float("nan")
            row["out_strength"] = strength
            row["in_signed_strength"] = float("nan")
            row["out_signed_strength"] = float("nan")

        rows.append(row)

    return pd.DataFrame(rows)
