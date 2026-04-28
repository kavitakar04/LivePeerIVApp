import sqlite3
import warnings

import numpy as np
import pandas as pd

from analysis.weights.unified_weights import (
    FeatureSet,
    UnifiedWeightComputer,
    WeightConfig,
    WeightMethod,
    corr_weights_from_matrix,
    cosine_similarity_weights_from_matrix,
    pca_regress_weights,
    _impute_col_median,
    _zscore_cols,
)
from display.gui.controls.gui_input import WEIGHT_METHOD_DISPLAY, WEIGHT_METHODS, weight_method_id, weight_method_label


def _features(noise: float = 0.0) -> pd.DataFrame:
    base = np.array([1.0, 2.0, 3.0, 4.0])
    return pd.DataFrame(
        {
            "f1": [base[0], base[0] + 0.10 + noise, -base[0], 1.0],
            "f2": [base[1], base[1] - 0.10 + noise, -base[1], 2.0],
            "f3": [base[2], base[2] + 0.10 + noise, -base[2], 2.0],
            "f4": [base[3], base[3] - 0.10 + noise, -base[3], 3.0],
        },
        index=["TGT", "P1", "P2", "P3"],
        dtype=float,
    )


def _assert_valid_simplex(weights: pd.Series, peers: list[str]) -> None:
    assert list(weights.index) == peers
    assert np.isfinite(weights.to_numpy(float)).all()
    assert (weights >= -1e-12).all()
    assert abs(float(weights.sum()) - 1.0) < 1e-9
    assert float(weights.max()) <= 0.98


def test_gui_weight_methods_match_engine_methods():
    assert list(WEIGHT_METHODS) == ["corr", "pca", "oi", "cosine", "equal"]
    assert list(WEIGHT_METHOD_DISPLAY) == [
        "Correlation",
        "PCA",
        "Open Interest",
        "Cosine Similarity",
        "Equal Weight",
    ]
    engine_methods = {m.value for m in WeightMethod}
    assert set(WEIGHT_METHODS) == engine_methods
    assert weight_method_label("corr") == "Correlation"
    assert weight_method_id("Correlation") == "corr"
    assert weight_method_id("corr") == "corr"


def test_nan_feature_helpers_do_not_emit_runtime_warnings():
    X = np.array(
        [
            [np.nan, 1.0, np.nan],
            [np.nan, 2.0, np.nan],
            [np.nan, np.nan, np.nan],
        ]
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        imputed = _impute_col_median(X)
        z, mu, sd = _zscore_cols(imputed)

    assert np.isfinite(imputed).all()
    assert np.isfinite(z).all()
    assert np.isfinite(mu).all()
    assert np.isfinite(sd).all()
    assert np.allclose(imputed[:, 0], 0.0)
    assert np.allclose(imputed[:, 2], 0.0)


def test_equal_mode_quality_gate(monkeypatch):
    computer = UnifiedWeightComputer()
    cfg = WeightConfig(method=WeightMethod.EQUAL, feature_set=FeatureSet.ATM)
    weights = computer.compute_weights("TGT", ["P1", "P2", "P3"], cfg)
    _assert_valid_simplex(weights, ["P1", "P2", "P3"])
    assert np.allclose(weights.to_numpy(float), 1.0 / 3.0)


def test_corr_mode_quality_gate(monkeypatch):
    computer = UnifiedWeightComputer()
    monkeypatch.setattr(
        computer,
        "_build_feature_matrix",
        lambda target, peers, asof, config: _features(),
    )
    cfg = WeightConfig(method=WeightMethod.CORRELATION, feature_set=FeatureSet.UNDERLYING_PX, corr_shrinkage=0.10)
    weights = computer.compute_weights("TGT", ["P1", "P2", "P3"], cfg)
    _assert_valid_simplex(weights, ["P1", "P2", "P3"])
    assert weights["P1"] > weights["P2"]


def test_corr_fallback_diagnostics_include_rejected_weights(monkeypatch, caplog):
    computer = UnifiedWeightComputer()
    feature_df = pd.DataFrame(
        {
            "f1": [1.0, 1.0, -1.0],
            "f2": [2.0, 2.0, -2.0],
            "f3": [3.0, 3.0, -3.0],
        },
        index=["TGT", "P1", "P2"],
        dtype=float,
    )
    monkeypatch.setattr(
        computer,
        "_build_feature_matrix",
        lambda target, peers, asof, config: feature_df,
    )
    cfg = WeightConfig(
        method=WeightMethod.CORRELATION,
        feature_set=FeatureSet.UNDERLYING_PX,
        corr_shrinkage=0.0,
    )

    with caplog.at_level("WARNING", logger="analysis.weights.unified_weights"):
        weights = computer.compute_weights("TGT", ["P1", "P2"], cfg)

    assert np.allclose(weights.to_numpy(float), [0.5, 0.5])
    messages = [record.getMessage() for record in caplog.records]
    diag_msg = next(msg for msg in messages if msg.startswith("weight diagnostics:"))
    assert "'fallback': 'equal'" in diag_msg
    assert "'rejected_weights': {'P1': 1.0, 'P2': 0.0}" in diag_msg


def test_cosine_mode_quality_gate(monkeypatch):
    computer = UnifiedWeightComputer()
    monkeypatch.setattr(
        computer,
        "_build_feature_matrix",
        lambda target, peers, asof, config: _features(),
    )
    cfg = WeightConfig(method=WeightMethod.COSINE, feature_set=FeatureSet.UNDERLYING_PX)
    weights = computer.compute_weights("TGT", ["P1", "P2", "P3"], cfg)
    _assert_valid_simplex(weights, ["P1", "P2", "P3"])
    assert weights["P1"] > weights["P2"]


def test_pca_mode_quality_gate(monkeypatch):
    computer = UnifiedWeightComputer()
    monkeypatch.setattr(
        computer,
        "_build_feature_matrix",
        lambda target, peers, asof, config: _features(),
    )
    cfg = WeightConfig(method=WeightMethod.PCA, feature_set=FeatureSet.UNDERLYING_PX, pca_ridge=1e-2)
    weights = computer.compute_weights("TGT", ["P1", "P2", "P3"], cfg)
    _assert_valid_simplex(weights, ["P1", "P2", "P3"])
    assert weights["P1"] > weights["P2"]


def test_open_interest_mode_quality_gate(monkeypatch):
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE options_quotes (asof_date TEXT, ticker TEXT, open_interest REAL)"
    )
    conn.executemany(
        "INSERT INTO options_quotes VALUES (?, ?, ?)",
        [
            ("2024-01-02", "P1", 100.0),
            ("2024-01-02", "P1", 200.0),
            ("2024-01-02", "P2", 300.0),
            ("2024-01-02", "P2", 400.0),
            ("2024-01-02", "P3", 500.0),
            ("2024-01-02", "P3", 600.0),
        ],
    )
    monkeypatch.setattr("data.db_utils.get_conn", lambda: conn)

    computer = UnifiedWeightComputer()
    cfg = WeightConfig(method=WeightMethod.OPEN_INTEREST, feature_set=FeatureSet.ATM, asof="2024-01-02")
    weights = computer.compute_weights("TGT", ["P1", "P2", "P3"], cfg)

    _assert_valid_simplex(weights, ["P1", "P2", "P3"])
    assert weights["P3"] > weights["P2"] > weights["P1"]


def test_corr_weights_are_regularized_and_normalized():
    weights = corr_weights_from_matrix(
        _features(),
        "TGT",
        ["P1", "P2", "P3"],
        shrinkage=0.10,
    )
    assert list(weights.index) == ["P1", "P2", "P3"]
    assert np.isfinite(weights.to_numpy(float)).all()
    assert (weights >= -1e-12).all()
    assert abs(float(weights.sum()) - 1.0) < 1e-9
    assert weights["P1"] > weights["P2"]


def test_cosine_weights_guard_near_zero_target():
    df = _features()
    df.loc["TGT"] = 1.0
    try:
        cosine_similarity_weights_from_matrix(df, "TGT", ["P1", "P2"])
    except ValueError as exc:
        assert "near zero" in str(exc)
    else:
        raise AssertionError("near-zero cosine target should fail")


def test_pca_weights_are_stable_under_small_noise():
    peers = ["P1", "P2", "P3"]
    base = _features(0.0)
    noisy = _features(1e-4)
    w0 = pd.Series(
        pca_regress_weights(
            base.loc[peers].to_numpy(float),
            base.loc["TGT"].to_numpy(float),
            ridge=1e-2,
        ),
        index=peers,
    )
    w1 = pd.Series(
        pca_regress_weights(
            noisy.loc[peers].to_numpy(float),
            noisy.loc["TGT"].to_numpy(float),
            ridge=1e-2,
        ),
        index=peers,
    )
    for weights in (w0, w1):
        assert list(weights.index) == peers
        assert np.isfinite(weights.to_numpy(float)).all()
        assert (weights >= -1e-12).all()
        assert abs(float(weights.sum()) - 1.0) < 1e-9
    assert float(np.linalg.norm((w0 - w1).to_numpy(float), ord=1)) < 0.05


def test_compute_pca_degenerate_concentration_falls_back(monkeypatch):
    computer = UnifiedWeightComputer()
    base = pd.DataFrame(
        {
            "f1": [1.0, 1.0, -1.0, 4.0],
            "f2": [2.0, 2.0, -2.0, 3.0],
            "f3": [3.0, 3.0, -3.0, 2.0],
            "f4": [4.0, 4.0, -4.0, 1.0],
        },
        index=["TGT", "P1", "P2", "P3"],
        dtype=float,
    )
    monkeypatch.setattr(
        computer,
        "_build_feature_matrix",
        lambda target, peers, asof, config: base,
    )
    cfg = WeightConfig(method=WeightMethod.PCA, feature_set=FeatureSet.UNDERLYING_PX)
    weights = computer.compute_weights("TGT", ["P1", "P2", "P3"], cfg)
    _assert_valid_simplex(weights, ["P1", "P2", "P3"])
    assert np.allclose(weights.to_numpy(float), [1 / 3, 1 / 3, 1 / 3])
    assert "using equal weights" in weights.attrs.get("weight_warning", "")


def test_compute_weights_falls_back_on_degenerate_cosine(monkeypatch):
    computer = UnifiedWeightComputer()
    degenerate = pd.DataFrame(
        [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
        index=["TGT", "P1", "P2"],
        columns=["a", "b"],
    )
    monkeypatch.setattr(
        computer,
        "_build_feature_matrix",
        lambda target, peers, asof, config: degenerate,
    )
    cfg = WeightConfig(method=WeightMethod.COSINE, feature_set=FeatureSet.UNDERLYING_PX)
    weights = computer.compute_weights("TGT", ["P1", "P2"], cfg)
    _assert_valid_simplex(weights, ["P1", "P2"])
    assert np.allclose(weights.to_numpy(float), [0.5, 0.5])
    assert "using equal weights" in weights.attrs.get("weight_warning", "")


def test_open_interest_mode_normalizes_and_rejects_extreme_concentration(monkeypatch):
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE options_quotes (asof_date TEXT, ticker TEXT, open_interest REAL)"
    )
    rows = [
        ("2024-01-01", "P1", 100.0),
        ("2024-01-01", "P1", 100.0),
        ("2024-01-01", "P2", 100.0),
        ("2024-01-01", "P2", 100.0),
    ]
    conn.executemany("INSERT INTO options_quotes VALUES (?, ?, ?)", rows)
    monkeypatch.setattr("data.db_utils.get_conn", lambda: conn)

    computer = UnifiedWeightComputer()
    cfg = WeightConfig(method=WeightMethod.OPEN_INTEREST, feature_set=FeatureSet.ATM)
    weights = computer.compute_weights("TGT", ["P1", "P2"], cfg)
    _assert_valid_simplex(weights, ["P1", "P2"])
    assert np.allclose(weights.to_numpy(float), [0.5, 0.5])

    conn.execute("DELETE FROM options_quotes")
    rows = [("2024-01-02", "P1", 10000.0), ("2024-01-02", "P2", 1.0)]
    conn.executemany("INSERT INTO options_quotes VALUES (?, ?, ?)", rows)
    weights = computer.compute_weights("TGT", ["P1", "P2"], cfg)
    _assert_valid_simplex(weights, ["P1", "P2"])
    assert np.allclose(weights.to_numpy(float), [0.5, 0.5])
    assert "using equal weights" in weights.attrs.get("weight_warning", "")
