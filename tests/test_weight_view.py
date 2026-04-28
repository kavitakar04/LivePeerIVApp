import pandas as pd

from analysis.weights.weight_view import resolve_peer_weights


def test_weight_view_uses_matching_cached_correlation(monkeypatch):
    monkeypatch.setattr("analysis.weights.unified_weights.compute_unified_weights", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("skip")))
    corr_df = pd.DataFrame(
        [[1.0, 0.8], [0.8, 1.0]],
        index=["TARGET", "PEER"],
        columns=["TARGET", "PEER"],
    )
    meta = {
        "weight_mode": "corr_iv_atm",
        "clip_negative": True,
        "weight_power": 1.0,
        "pillars": [30],
        "asof": "2024-01-01",
        "tickers": ["TARGET", "PEER"],
    }

    weights = resolve_peer_weights(
        "TARGET",
        ["PEER"],
        "corr_iv_atm",
        asof="2024-01-01",
        pillars=[30],
        settings={"clip_negative": True, "weight_power": 1.0},
        last_corr_df=corr_df,
        last_corr_meta=meta,
    )

    assert weights.to_dict() == {"PEER": 1.0}


def test_weight_view_uses_pillars_as_surface_tenors(monkeypatch):
    captured = {}

    def fake_compute_unified_weights(**kwargs):
        captured.update(kwargs)
        return pd.Series({"PEER": 1.0})

    monkeypatch.setattr("analysis.weights.unified_weights.compute_unified_weights", fake_compute_unified_weights)

    weights = resolve_peer_weights(
        "TARGET",
        ["PEER"],
        "pca_surface_grid",
        asof="2024-01-01",
        pillars=[7, 14, 21],
        settings={
            "clip_negative": True,
            "weight_power": 1.0,
            "pillars": [7, 14, 21],
            "mny_bins": ((0.5, 0.6),),
        },
    )

    assert weights.to_dict() == {"PEER": 1.0}
    assert captured["tenors"] == [7, 14, 21]
    assert captured["mny_bins"] == ((0.5, 0.6),)
