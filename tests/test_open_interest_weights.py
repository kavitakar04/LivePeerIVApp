import sqlite3

import pytest

from data.db_utils import ensure_initialized, insert_quotes
from analysis.weights.unified_weights import compute_unified_weights
from analysis.weights.weight_service import compute_peer_weights


def test_open_interest_weighting(monkeypatch):
    conn = sqlite3.connect(":memory:")
    ensure_initialized(conn)

    quotes = [
        {
            "asof_date": "2024-01-01",
            "ticker": "AAA",
            "expiry": "2024-02-01",
            "K": 100,
            "call_put": "C",
            "open_interest": 100,
        },
        {
            "asof_date": "2024-01-01",
            "ticker": "BBB",
            "expiry": "2024-02-01",
            "K": 100,
            "call_put": "C",
            "open_interest": 300,
        },
    ]
    insert_quotes(conn, quotes)

    monkeypatch.setattr("data.db_utils.get_conn", lambda db_path=None: conn)

    weights = compute_unified_weights(
        target="TGT",
        peers=["AAA", "BBB"],
        mode="oi",
        asof="2024-01-01",
    )

    assert weights.loc["AAA"] == pytest.approx(0.25)
    assert weights.loc["BBB"] == pytest.approx(0.75)

    # Integration test through analysis pipeline
    weights_pipeline = compute_peer_weights(
        target="TGT",
        peers=["AAA", "BBB"],
        weight_mode="oi",
        asof="2024-01-01",
    )
    assert weights_pipeline.loc["AAA"] == pytest.approx(0.25)
    assert weights_pipeline.loc["BBB"] == pytest.approx(0.75)
