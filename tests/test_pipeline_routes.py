import pandas as pd
import pytest

from analysis.weight_service import compute_peer_weights


def test_compute_peer_weights_dispatch(monkeypatch):
    called = {}

    def fake_compute_unified_weights(*, target, peers, mode, **kwargs):
        called["target"] = target
        called["peers"] = tuple(peers)
        called["mode"] = mode
        return pd.Series({"PEER": 1.0})

    monkeypatch.setattr(
        "analysis.weight_service.compute_unified_weights",
        fake_compute_unified_weights,
    )

    res = compute_peer_weights(target="SPY", peers=["QQQ"], weight_mode="cosine_ul")

    assert called == {"target": "SPY", "peers": ("QQQ",), "mode": "cosine_ul"}
    assert res.loc["PEER"] == pytest.approx(1.0)


def test_pipeline_compute_peer_weights_is_weight_service_facade():
    import analysis.analysis_pipeline as pipeline
    import analysis.weight_service as weight_service

    assert pipeline.compute_peer_weights is weight_service.compute_peer_weights
