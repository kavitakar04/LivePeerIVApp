"""Market knowledge graph helpers for IV relative-value analysis.

The graph is an evidence layer over existing analysis outputs.  It does not
replace correlation, spillover, weighting, or model-fit code; it combines those
outputs into one typed network so peer selection and signal confidence can use
consistent relationship evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import math

import numpy as np
import pandas as pd

try:
    import networkx as nx
except ImportError as exc:  # pragma: no cover - dependency absent path
    nx = None
    _NETWORKX_IMPORT_ERROR = exc
else:
    _NETWORKX_IMPORT_ERROR = None


@dataclass(frozen=True)
class MarketGraphConfig:
    """Thresholds and scoring weights for market relationship graphs."""

    min_abs_corr: float = 0.30
    min_surface_similarity: float = 0.30
    min_spillover_n: int = 10
    min_spillover_hit_rate: float = 0.0
    surface_score_weight: float = 0.35
    corr_score_weight: float = 0.25
    spillover_score_weight: float = 0.20
    composite_weight_score_weight: float = 0.15
    quality_score_weight: float = 0.05


def _require_networkx() -> None:
    if nx is None:
        raise ImportError(
            "networkx is required for market graph analysis. Install with `pip install networkx`."
        ) from _NETWORKX_IMPORT_ERROR


def _ticker(value: Any) -> str:
    return str(value or "").upper().strip()


def _finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _clip01(value: Any, default: float = 0.0) -> float:
    out = _finite_float(value, default)
    if not math.isfinite(out):
        return default
    return float(np.clip(out, 0.0, 1.0))


def _quality_score(meta: Mapping[str, Any] | None) -> float:
    if not meta:
        return 0.50
    if bool(meta.get("degraded")):
        return 0.20
    status = str(meta.get("status") or meta.get("quality") or "").lower()
    if status in {"good", "ok", "pass", "passed"}:
        return 1.0
    if status in {"acceptable", "mixed"}:
        return 0.75
    if status in {"unknown", ""}:
        rmse = _finite_float(meta.get("rmse"), float("nan"))
        if math.isfinite(rmse):
            return float(np.clip(1.0 - rmse / 0.30, 0.10, 1.0))
        return 0.50
    if status in {"degraded", "poor", "fail", "failed"}:
        return 0.20
    return 0.50


def _add_ticker_node(G, ticker: str, *, target: str, peers: set[str]) -> None:
    if not ticker:
        return
    roles: list[str] = []
    if ticker == target:
        roles.append("target")
    if ticker in peers:
        roles.append("peer")
    G.add_node(ticker, node_type="ticker", ticker=ticker, roles=tuple(roles))


def _edge_key(relationship: str, layer: str, suffix: str = "") -> str:
    return f"{layer}:{relationship}:{suffix}" if suffix else f"{layer}:{relationship}"


def _add_bidirectional_edge(G, a: str, b: str, *, key: str, **attrs: Any) -> None:
    if not a or not b or a == b:
        return
    G.add_edge(a, b, key=key, **attrs)
    G.add_edge(b, a, key=key, **attrs)


def _matrix_labels(df: pd.DataFrame) -> list[str]:
    if df is None or df.empty:
        return []
    return [_ticker(x) for x in df.index]


def _add_similarity_matrix(
    G,
    df: pd.DataFrame | None,
    *,
    target: str,
    peers: set[str],
    relationship: str,
    layer: str,
    value_attr: str,
    min_abs_value: float,
) -> None:
    if df is None or df.empty:
        return
    labels = _matrix_labels(df)
    cols = [_ticker(x) for x in df.columns]
    if labels != cols:
        raise ValueError(f"{layer} matrix must have matching index/columns")
    for label in labels:
        _add_ticker_node(G, label, target=target, peers=peers)
    for i, src in enumerate(labels):
        for j in range(i + 1, len(labels)):
            dst = labels[j]
            value = _finite_float(df.iloc[i, j])
            if not math.isfinite(value):
                continue
            abs_value = abs(value)
            if abs_value < float(min_abs_value) or abs_value <= 0.0:
                continue
            _add_bidirectional_edge(
                G,
                src,
                dst,
                key=_edge_key(relationship, layer),
                relationship=relationship,
                layer=layer,
                weight=abs_value,
                abs_weight=abs_value,
                signed_weight=value,
                distance=1.0 / abs_value,
                **{value_attr: value},
            )


def _add_spillover_edges(
    G,
    spillover_summary: pd.DataFrame | None,
    *,
    target: str,
    peers: set[str],
    config: MarketGraphConfig,
    horizon: int | None,
) -> None:
    if spillover_summary is None or spillover_summary.empty:
        return
    required = {"ticker", "peer", "h", "n", "hit_rate", "median_elasticity"}
    missing = sorted(required - set(spillover_summary.columns))
    if missing:
        raise ValueError(f"spillover_summary is missing required columns: {missing}")
    df = spillover_summary.copy()
    if horizon is not None:
        df = df[pd.to_numeric(df["h"], errors="coerce") == int(horizon)]
    for _, row in df.iterrows():
        src = _ticker(row.get("ticker"))
        dst = _ticker(row.get("peer"))
        n = int(_finite_float(row.get("n"), 0.0))
        hit_rate = _clip01(row.get("hit_rate"))
        elasticity = _finite_float(row.get("median_elasticity"))
        if not src or not dst or src == dst:
            continue
        if n < config.min_spillover_n or hit_rate < config.min_spillover_hit_rate:
            continue
        if not math.isfinite(elasticity) or elasticity == 0.0:
            continue
        abs_weight = abs(elasticity)
        _add_ticker_node(G, src, target=target, peers=peers)
        _add_ticker_node(G, dst, target=target, peers=peers)
        G.add_edge(
            src,
            dst,
            key=_edge_key("spills_over_to", "spillover", str(row.get("h", ""))),
            relationship="spills_over_to",
            layer="spillover",
            weight=abs_weight,
            abs_weight=abs_weight,
            signed_weight=elasticity,
            distance=1.0 / abs_weight,
            horizon=int(_finite_float(row.get("h"), 0.0)),
            n=n,
            hit_rate=hit_rate,
            sign_concord=_clip01(row.get("sign_concord", np.nan), default=float("nan")),
        )


def _add_composite_weight_edges(
    G,
    weights: Mapping[str, Any] | pd.Series | None,
    *,
    target: str,
    peers: set[str],
) -> None:
    if weights is None:
        return
    series = pd.Series(weights, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if series.empty:
        return
    denom = float(series.abs().sum())
    if denom <= 0.0:
        return
    for peer_raw, value in series.items():
        peer = _ticker(peer_raw)
        weight = float(value)
        if not peer or peer == target:
            continue
        strength = abs(weight) / denom
        _add_ticker_node(G, peer, target=target, peers=peers)
        _add_ticker_node(G, target, target=target, peers=peers)
        G.add_edge(
            peer,
            target,
            key=_edge_key("explains_composite_for", "weights"),
            relationship="explains_composite_for",
            layer="weights",
            weight=strength,
            abs_weight=strength,
            signed_weight=weight,
            composite_weight=weight,
            normalized_weight=strength,
            distance=1.0 / strength if strength > 0 else float("inf"),
        )


def _add_quality_nodes(
    G,
    model_quality: Mapping[str, Mapping[str, Any]] | None,
    *,
    target: str,
    peers: set[str],
) -> None:
    if not model_quality:
        return
    for ticker_raw, meta in model_quality.items():
        ticker = _ticker(ticker_raw)
        if not ticker:
            continue
        _add_ticker_node(G, ticker, target=target, peers=peers)
        score = _quality_score(meta)
        G.nodes[ticker]["model_quality_score"] = score
        G.nodes[ticker]["model_quality"] = dict(meta or {})


def build_market_graph(
    *,
    target: str,
    peers: Iterable[str],
    corr: pd.DataFrame | None = None,
    surface_similarity: pd.DataFrame | None = None,
    spillover_summary: pd.DataFrame | None = None,
    weights: Mapping[str, Any] | pd.Series | None = None,
    model_quality: Mapping[str, Mapping[str, Any]] | None = None,
    themes: Mapping[str, Iterable[str]] | None = None,
    asof: str | None = None,
    horizon: int | None = 1,
    config: MarketGraphConfig | None = None,
):
    """Build a heterogeneous market graph for a target/peer analysis.

    Returns a ``networkx.MultiDiGraph``.  Ticker-to-ticker evidence is stored as
    typed edges.  Multiple evidence layers can coexist between the same nodes.
    """
    _require_networkx()
    cfg = config or MarketGraphConfig()
    target_up = _ticker(target)
    peer_set = {_ticker(p) for p in peers if _ticker(p)}

    G = nx.MultiDiGraph(
        graph_type="market_knowledge_graph",
        target=target_up,
        peers=tuple(sorted(peer_set)),
        asof=asof,
    )
    _add_ticker_node(G, target_up, target=target_up, peers=peer_set)
    for peer in sorted(peer_set):
        _add_ticker_node(G, peer, target=target_up, peers=peer_set)

    _add_similarity_matrix(
        G,
        corr,
        target=target_up,
        peers=peer_set,
        relationship="correlated_with",
        layer="correlation",
        value_attr="corr",
        min_abs_value=cfg.min_abs_corr,
    )
    _add_similarity_matrix(
        G,
        surface_similarity,
        target=target_up,
        peers=peer_set,
        relationship="similar_surface_to",
        layer="surface",
        value_attr="surface_similarity",
        min_abs_value=cfg.min_surface_similarity,
    )
    _add_spillover_edges(G, spillover_summary, target=target_up, peers=peer_set, config=cfg, horizon=horizon)
    _add_composite_weight_edges(G, weights, target=target_up, peers=peer_set)
    _add_quality_nodes(G, model_quality, target=target_up, peers=peer_set)

    if themes:
        for theme, tickers in themes.items():
            theme_id = f"theme:{str(theme).strip()}"
            G.add_node(theme_id, node_type="theme", label=str(theme))
            for ticker_raw in tickers:
                ticker = _ticker(ticker_raw)
                if not ticker:
                    continue
                _add_ticker_node(G, ticker, target=target_up, peers=peer_set)
                G.add_edge(
                    ticker,
                    theme_id,
                    key=_edge_key("shares_theme_with", "theme"),
                    relationship="shares_theme_with",
                    layer="theme",
                    weight=1.0,
                    abs_weight=1.0,
                )

    return G


def _edge_values_between(G, src: str, dst: str, relationship: str, attr: str) -> list[float]:
    values: list[float] = []
    if not G.has_edge(src, dst):
        return values
    for _key, data in G.get_edge_data(src, dst).items():
        if data.get("relationship") != relationship:
            continue
        value = _finite_float(data.get(attr))
        if math.isfinite(value):
            values.append(value)
    return values


def _best_bidirectional(G, a: str, b: str, relationship: str, attr: str) -> float:
    vals = _edge_values_between(G, a, b, relationship, attr)
    vals.extend(_edge_values_between(G, b, a, relationship, attr))
    if not vals:
        return 0.0
    return float(max(abs(v) for v in vals))


def _best_directional(G, src: str, dst: str, relationship: str, attr: str) -> float:
    vals = _edge_values_between(G, src, dst, relationship, attr)
    if not vals:
        return 0.0
    return float(max(abs(v) for v in vals))


def rank_peer_candidates(G, target: str | None = None, peers: Iterable[str] | None = None) -> pd.DataFrame:
    """Rank peers by graph evidence around the target."""
    _require_networkx()
    target_up = _ticker(target or G.graph.get("target"))
    peer_set = {_ticker(p) for p in (peers or G.graph.get("peers", ())) if _ticker(p)}
    cfg = MarketGraphConfig()

    rows: list[dict[str, Any]] = []
    for peer in sorted(peer_set):
        corr = _best_bidirectional(G, target_up, peer, "correlated_with", "corr")
        surface = _best_bidirectional(G, target_up, peer, "similar_surface_to", "surface_similarity")
        spill_from_target = _best_directional(G, target_up, peer, "spills_over_to", "abs_weight")
        spill_to_target = _best_directional(G, peer, target_up, "spills_over_to", "abs_weight")
        spillover = max(spill_from_target, spill_to_target)
        composite = _best_directional(G, peer, target_up, "explains_composite_for", "normalized_weight")
        quality = _clip01(G.nodes.get(peer, {}).get("model_quality_score", 0.50), default=0.50)
        score = (
            cfg.surface_score_weight * _clip01(surface)
            + cfg.corr_score_weight * _clip01(corr)
            + cfg.spillover_score_weight * _clip01(spillover)
            + cfg.composite_weight_score_weight * _clip01(composite)
            + cfg.quality_score_weight * quality
        )
        rows.append(
            {
                "peer": peer,
                "graph_score": float(score),
                "surface_similarity": surface,
                "abs_corr": corr,
                "spillover_strength": spillover,
                "composite_weight": composite,
                "model_quality_score": quality,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "peer",
                "graph_score",
                "surface_similarity",
                "abs_corr",
                "spillover_strength",
                "composite_weight",
                "model_quality_score",
            ]
        )
    return pd.DataFrame(rows).sort_values(["graph_score", "peer"], ascending=[False, True]).reset_index(drop=True)


def graph_confidence_features(G, target: str | None = None, peers: Iterable[str] | None = None) -> dict[str, Any]:
    """Summarize graph evidence for RV signal confidence scoring."""
    _require_networkx()
    ranks = rank_peer_candidates(G, target=target, peers=peers)
    if ranks.empty:
        return {
            "peer_count": 0,
            "mean_peer_score": 0.0,
            "min_peer_score": 0.0,
            "evidence_density": 0.0,
            "has_spillover_support": False,
            "has_surface_support": False,
        }
    target_up = _ticker(target or G.graph.get("target"))
    peer_set = {_ticker(p) for p in (peers or G.graph.get("peers", ())) if _ticker(p)}
    possible_layers = max(len(peer_set) * 4, 1)
    observed_layers = 0
    for peer in peer_set:
        observed_layers += int(_best_bidirectional(G, target_up, peer, "similar_surface_to", "surface_similarity") > 0)
        observed_layers += int(_best_bidirectional(G, target_up, peer, "correlated_with", "corr") > 0)
        observed_layers += int(_best_directional(G, target_up, peer, "spills_over_to", "abs_weight") > 0)
        observed_layers += int(_best_directional(G, peer, target_up, "explains_composite_for", "normalized_weight") > 0)
    return {
        "peer_count": int(len(ranks)),
        "mean_peer_score": float(ranks["graph_score"].mean()),
        "min_peer_score": float(ranks["graph_score"].min()),
        "evidence_density": float(observed_layers / possible_layers),
        "has_spillover_support": bool((ranks["spillover_strength"] > 0).any()),
        "has_surface_support": bool((ranks["surface_similarity"] > 0).any()),
    }


def explain_rv_signal_with_graph(
    G,
    *,
    target: str | None = None,
    peers: Iterable[str] | None = None,
    top_n: int = 3,
) -> list[str]:
    """Return compact explanation bullets from graph evidence."""
    ranks = rank_peer_candidates(G, target=target, peers=peers)
    if ranks.empty:
        return ["No graph evidence was available for the selected peer set."]
    top = ranks.head(int(top_n))
    bullets: list[str] = []
    bullets.append(
        "Strongest graph peers: "
        + ", ".join(f"{row.peer} ({row.graph_score:.2f})" for row in top.itertuples(index=False))
        + "."
    )
    if (top["surface_similarity"] > 0).any():
        best = top.sort_values("surface_similarity", ascending=False).iloc[0]
        bullets.append(f"{best['peer']} has the strongest surface-similarity evidence ({best['surface_similarity']:.2f}).")
    if (top["spillover_strength"] > 0).any():
        best = top.sort_values("spillover_strength", ascending=False).iloc[0]
        bullets.append(f"{best['peer']} has the strongest spillover evidence ({best['spillover_strength']:.2f}).")
    weak_quality = top[top["model_quality_score"] < 0.50]
    if not weak_quality.empty:
        bullets.append("Model-quality evidence is weak for: " + ", ".join(weak_quality["peer"].astype(str)) + ".")
    return bullets


__all__ = [
    "MarketGraphConfig",
    "build_market_graph",
    "rank_peer_candidates",
    "graph_confidence_features",
    "explain_rv_signal_with_graph",
]
