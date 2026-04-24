import pandas as pd

from analysis.spillover.network_graph import (
    build_corr_graph,
    build_spillover_digraph,
    compute_graph_metrics,
)


def test_build_spillover_digraph_filters_and_attrs():
    summary = pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "peer": "MSFT",
                "h": 1,
                "n": 20,
                "hit_rate": 0.50,
                "sign_concord": 0.80,
                "median_elasticity": 0.30,
                "median_resp": 0.02,
            },
            {
                "ticker": "AAPL",
                "peer": "GOOGL",
                "h": 1,
                "n": 5,
                "hit_rate": 0.20,
                "sign_concord": 0.60,
                "median_elasticity": -0.10,
                "median_resp": -0.01,
            },
        ]
    )

    G = build_spillover_digraph(summary, horizon=1, min_n=10, min_hit_rate=0.4)

    assert set(G.nodes()) == {"AAPL", "MSFT"}
    assert set(G.edges()) == {("AAPL", "MSFT")}
    assert G["AAPL"]["MSFT"]["weight"] == 0.30


def test_build_corr_graph_signed_weighted_edges():
    corr = pd.DataFrame(
        [[1.0, 0.6, -0.2], [0.6, 1.0, -0.4], [-0.2, -0.4, 1.0]],
        index=["A", "B", "C"],
        columns=["A", "B", "C"],
    )

    G = build_corr_graph(corr, min_abs_corr=0.3)

    assert set(G.edges()) == {("A", "B"), ("B", "C")}
    assert G["A"]["B"]["weight"] == 0.6
    assert G["B"]["C"]["sign"] == -1


def test_compute_graph_metrics_directed_strengths_present():
    summary = pd.DataFrame(
        [
            {
                "ticker": "X",
                "peer": "Y",
                "h": 1,
                "n": 20,
                "hit_rate": 0.5,
                "sign_concord": 0.8,
                "median_elasticity": 0.4,
                "median_resp": 0.03,
            },
            {
                "ticker": "Y",
                "peer": "Z",
                "h": 1,
                "n": 20,
                "hit_rate": 0.5,
                "sign_concord": 0.8,
                "median_elasticity": 0.7,
                "median_resp": 0.04,
            },
        ]
    )
    G = build_spillover_digraph(summary, horizon=1, min_n=1)

    metrics = compute_graph_metrics(G).set_index("node")

    assert {"degree", "in_degree", "out_degree", "out_strength"}.issubset(metrics.columns)
    assert metrics.loc["Y", "in_degree"] == 1.0
    assert metrics.loc["Y", "out_degree"] == 1.0
