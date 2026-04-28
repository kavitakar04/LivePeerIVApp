import sqlite3
import numpy as np
import pandas as pd

from data.db_utils import ensure_initialized
from analysis.weights.unified_weights import compute_unified_weights
from analysis.weights.correlation_utils import corr_weights


def test_ul_weight_mode_exposes_concentrated_correlation_fallback(monkeypatch):
    conn = sqlite3.connect(":memory:")
    ensure_initialized(conn)

    prices = [
        ("2024-01-01", "TGT", 100),
        ("2024-01-02", "TGT", 101),
        ("2024-01-03", "TGT", 103),
        ("2024-01-04", "TGT", 106),
        ("2024-01-01", "AAA", 10),
        ("2024-01-02", "AAA", 10.5),
        ("2024-01-03", "AAA", 11.5),
        ("2024-01-04", "AAA", 13),
        ("2024-01-01", "BBB", 20),
        ("2024-01-02", "BBB", 21),
        ("2024-01-03", "BBB", 19),
        ("2024-01-04", "BBB", 20),
    ]
    conn.executemany(
        "INSERT INTO underlying_prices(asof_date, ticker, close) VALUES (?,?,?)",
        prices,
    )
    conn.commit()

    monkeypatch.setattr("data.db_utils.get_conn", lambda db_path=None: conn)

    weights = compute_unified_weights(
        target="TGT", peers=["AAA", "BBB"], mode="corr_ul", asof="2024-01-04"
    )

    df = pd.DataFrame(
        {
            "TGT": [100, 101, 103, 106],
            "AAA": [10, 10.5, 11.5, 13],
            "BBB": [20, 21, 19, 20],
        }
    )
    ret = np.log(df / df.shift(1)).dropna()
    raw_expected = corr_weights(ret.corr(), "TGT", ["AAA", "BBB"])

    pd.testing.assert_series_equal(
        raw_expected.rename_axis(None).rename(None), pd.Series({"AAA": 1.0, "BBB": 0.0})
    )
    pd.testing.assert_series_equal(
        weights.rename_axis(None), pd.Series({"AAA": 0.5, "BBB": 0.5})
    )
    assert "using equal weights" in weights.attrs["weight_warning"]
