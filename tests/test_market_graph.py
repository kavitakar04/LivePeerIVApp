import pandas as pd

from analysis.market_graph import (
    build_market_graph,
    explain_rv_signal_with_graph,
    graph_confidence_features,
    rank_peer_candidates,
)


def _sample_inputs():
    labels = ["WULF", "IREN", "CORZ", "CIFR"]
    corr = pd.DataFrame(
        [
            [1.00, 0.72, 0.35, 0.55],
            [0.72, 1.00, 0.20, 0.40],
            [0.35, 0.20, 1.00, 0.50],
            [0.55, 0.40, 0.50, 1.00],
        ],
        index=labels,
        columns=labels,
    )
    surface = pd.DataFrame(
        [
            [1.00, 0.90, 0.20, 0.65],
            [0.90, 1.00, 0.25, 0.50],
            [0.20, 0.25, 1.00, 0.55],
            [0.65, 0.50, 0.55, 1.00],
        ],
        index=labels,
        columns=labels,
    )
    spillover = pd.DataFrame(
        [
            {
                "ticker": "WULF",
                "peer": "IREN",
                "h": 1,
                "n": 24,
                "hit_rate": 0.70,
                "sign_concord": 0.80,
                "median_elasticity": 0.45,
            },
            {
                "ticker": "CORZ",
                "peer": "WULF",
                "h": 1,
                "n": 24,
                "hit_rate": 0.60,
                "sign_concord": 0.70,
                "median_elasticity": 0.30,
            },
        ]
    )
    weights = {"IREN": 0.50, "CORZ": 0.20, "CIFR": 0.30}
    quality = {
        "IREN": {"status": "Good", "rmse": 0.04},
        "CORZ": {"status": "Degraded", "rmse": 0.25, "degraded": True},
        "CIFR": {"status": "Acceptable", "rmse": 0.10},
    }
    return corr, surface, spillover, weights, quality


def test_build_market_graph_combines_market_relationship_layers():
    corr, surface, spillover, weights, quality = _sample_inputs()

    G = build_market_graph(
        target="WULF",
        peers=["IREN", "CORZ", "CIFR"],
        corr=corr,
        surface_similarity=surface,
        spillover_summary=spillover,
        weights=weights,
        model_quality=quality,
        themes={"AI miners": ["WULF", "IREN", "CORZ", "CIFR"]},
        asof="2026-04-25",
    )

    assert G.graph["target"] == "WULF"
    assert G.nodes["WULF"]["node_type"] == "ticker"
    assert G.nodes["IREN"]["model_quality_score"] == 1.0
    assert "theme:AI miners" in G.nodes

    relationships = {data["relationship"] for _, _, data in G.edges(data=True)}
    assert {
        "correlated_with",
        "similar_surface_to",
        "spills_over_to",
        "explains_composite_for",
        "shares_theme_with",
    }.issubset(relationships)


def test_rank_peer_candidates_prefers_peer_with_multi_layer_support():
    corr, surface, spillover, weights, quality = _sample_inputs()
    G = build_market_graph(
        target="WULF",
        peers=["IREN", "CORZ", "CIFR"],
        corr=corr,
        surface_similarity=surface,
        spillover_summary=spillover,
        weights=weights,
        model_quality=quality,
    )

    ranked = rank_peer_candidates(G)

    assert list(ranked["peer"])[0] == "IREN"
    assert ranked.iloc[0]["surface_similarity"] == 0.90
    assert ranked.iloc[0]["abs_corr"] == 0.72
    assert ranked.iloc[0]["spillover_strength"] == 0.45
    assert ranked.iloc[0]["composite_weight"] == 0.50


def test_graph_confidence_features_and_explanations_are_analysis_ready():
    corr, surface, spillover, weights, quality = _sample_inputs()
    G = build_market_graph(
        target="WULF",
        peers=["IREN", "CORZ", "CIFR"],
        corr=corr,
        surface_similarity=surface,
        spillover_summary=spillover,
        weights=weights,
        model_quality=quality,
    )

    features = graph_confidence_features(G)
    explanations = explain_rv_signal_with_graph(G)

    assert features["peer_count"] == 3
    assert features["mean_peer_score"] > 0
    assert features["evidence_density"] > 0
    assert features["has_spillover_support"] is True
    assert features["has_surface_support"] is True
    assert any("Strongest graph peers" in item for item in explanations)
