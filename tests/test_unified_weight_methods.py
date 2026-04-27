import numpy as np
import pandas as pd
import pytest


from analysis.unified_weights import (
    UnifiedWeightComputer,
    WeightConfig,
    WeightMethod,
    FeatureSet,
)


def _patch_feature_matrix(monkeypatch, feature_df):
    monkeypatch.setattr(
        UnifiedWeightComputer,
        "_build_feature_matrix",
        lambda self, target, peers, asof, config: feature_df,
    )


def test_correlation_weights_success(monkeypatch):
    uwc = UnifiedWeightComputer()
    cfg = WeightConfig(
        method=WeightMethod.CORRELATION,
        feature_set=FeatureSet.ATM,
        asof="2024-01-01",
    )
    feature_df = pd.DataFrame(
        [[1, 2], [1, 2], [1, 2]],
        index=["TGT", "P1", "P2"],
        columns=["f1", "f2"],
    )
    _patch_feature_matrix(monkeypatch, feature_df)
    weights = uwc.compute_weights("TGT", ["P1", "P2"], cfg)
    assert weights.sum() == pytest.approx(1.0)
    assert weights.loc["P1"] == pytest.approx(0.5)
    assert weights.loc["P2"] == pytest.approx(0.5)
    assert "weight_diagnostics" in weights.attrs


def test_correlation_weights_zero_sum_raises(monkeypatch):
    uwc = UnifiedWeightComputer()
    cfg = WeightConfig(
        method=WeightMethod.CORRELATION,
        feature_set=FeatureSet.ATM,
        asof="2024-01-01",
    )
    feature_df = pd.DataFrame(
        [[1, 1], [-1, -1], [-2, -2]],
        index=["TGT", "P1", "P2"],
        columns=["f1", "f2"],
    )
    _patch_feature_matrix(monkeypatch, feature_df)
    weights = uwc.compute_weights("TGT", ["P1", "P2"], cfg)

    assert weights.to_dict() == {"P1": pytest.approx(0.5), "P2": pytest.approx(0.5)}
    assert "using equal weights" in weights.attrs["weight_warning"]


def test_cosine_weights_success(monkeypatch):
    uwc = UnifiedWeightComputer()
    cfg = WeightConfig(
        method=WeightMethod.COSINE,
        feature_set=FeatureSet.ATM,
        asof="2024-01-01",
    )
    feature_df = pd.DataFrame(
        [[1, 0], [1, 0], [1, 0]],
        index=["TGT", "P1", "P2"],
        columns=["f1", "f2"],
    )
    _patch_feature_matrix(monkeypatch, feature_df)
    weights = uwc.compute_weights("TGT", ["P1", "P2"], cfg)
    assert weights.sum() == pytest.approx(1.0)
    assert weights.loc["P1"] == pytest.approx(0.5)
    assert weights.loc["P2"] == pytest.approx(0.5)


def test_cosine_weights_zero_sum_raises(monkeypatch):
    uwc = UnifiedWeightComputer()
    cfg = WeightConfig(
        method=WeightMethod.COSINE,
        feature_set=FeatureSet.ATM,
        asof="2024-01-01",
    )
    feature_df = pd.DataFrame(
        [[1, 0], [0, 1], [0, -1]],
        index=["TGT", "P1", "P2"],
        columns=["f1", "f2"],
    )
    _patch_feature_matrix(monkeypatch, feature_df)
    weights = uwc.compute_weights("TGT", ["P1", "P2"], cfg)

    assert weights.to_dict() == {"P1": pytest.approx(0.5), "P2": pytest.approx(0.5)}
    assert "using equal weights" in weights.attrs["weight_warning"]


def test_pca_weights(monkeypatch):
    uwc = UnifiedWeightComputer()
    cfg = WeightConfig(
        method=WeightMethod.PCA,
        feature_set=FeatureSet.ATM,
        asof="2024-01-01",
    )
    feature_df = pd.DataFrame(
        [[1, 2], [2, 1], [3, 0]],
        index=["TGT", "P1", "P2"],
        columns=["f1", "f2"],
    )
    _patch_feature_matrix(monkeypatch, feature_df)
    monkeypatch.setattr(
        "analysis.unified_weights._impute_col_median", lambda arr: arr
    )
    monkeypatch.setattr(
        "analysis.unified_weights.pca_regress_weights",
        lambda Xp, y, k=None, nonneg=True: np.array([2.0, 1.0]),
    )
    weights = uwc.compute_weights("TGT", ["P1", "P2"], cfg)
    assert weights.sum() == pytest.approx(1.0)
    assert weights.loc["P1"] == pytest.approx(2.0 / 3.0)
    assert weights.loc["P2"] == pytest.approx(1.0 / 3.0)


def test_equal_weights():
    uwc = UnifiedWeightComputer()
    cfg = WeightConfig(method=WeightMethod.EQUAL, feature_set=FeatureSet.ATM)
    weights = uwc.compute_weights("TGT", ["P1", "P2", "P3"], cfg)
    assert weights.sum() == pytest.approx(1.0)
    assert all(weight == pytest.approx(1.0 / 3.0) for weight in weights)


def test_surface_feature_set_dispatch(monkeypatch):
    uwc = UnifiedWeightComputer()
    cfg = WeightConfig(
        method=WeightMethod.CORRELATION,
        feature_set=FeatureSet.SURFACE,
        asof="2024-01-01",
    )

    feature_df = pd.DataFrame(
        [[1, 2], [1, 2], [1, 2]],
        index=["TGT", "P1", "P2"],
        columns=["f1", "f2"],
    )

    called = {"surface": False}

    def fake_surface(self, tickers, asof, config):
        called["surface"] = True
        return feature_df

    monkeypatch.setattr(UnifiedWeightComputer, "_build_surface_features", fake_surface)
    monkeypatch.setattr(
        UnifiedWeightComputer,
        "_compute_weights_from_features",
        lambda self, df, target, peers, config: pd.Series(0.5, index=peers),
    )

    weights = uwc.compute_weights("TGT", ["P1", "P2"], cfg)
    assert called["surface"]
    assert weights.sum() == pytest.approx(1.0)
    assert weights.loc["P1"] == pytest.approx(0.5)
    assert weights.loc["P2"] == pytest.approx(0.5)


def test_surface_and_surface_grid_feature_sets_are_distinct(monkeypatch):
    uwc = UnifiedWeightComputer()
    native = pd.DataFrame([[1.0, 2.0]], index=["TGT"], columns=["rank0_0.90-1.00", "rank0_1.00-1.10"])

    monkeypatch.setattr(
        "analysis.unified_weights.native_surface_feature_matrix",
        lambda tickers, asof, max_expiries, mny_bins: (native, list(native.columns)),
    )
    monkeypatch.setattr(
        "analysis.unified_weights.surface_feature_matrix",
        lambda tickers, asof, tenors, mny_bins: ({ticker: {} for ticker in tickers}, np.array([[3.0, 4.0]]), ["T30_0.90-1.00", "T60_1.00-1.10"]),
    )

    surface = uwc._build_feature_matrix(
        "TGT",
        [],
        "2024-01-01",
        WeightConfig(method=WeightMethod.CORRELATION, feature_set=FeatureSet.SURFACE, asof="2024-01-01"),
    )
    grid = uwc._build_feature_matrix(
        "TGT",
        [],
        "2024-01-01",
        WeightConfig(method=WeightMethod.CORRELATION, feature_set=FeatureSet.SURFACE_VECTOR, asof="2024-01-01"),
    )

    assert surface.attrs["feature_diagnostics"]["coordinate_system"] == "native_expiry_rank_x_moneyness_bin"
    assert grid.attrs["feature_diagnostics"]["coordinate_system"] == "standardized_tenor_grid_x_moneyness_bin"
    assert list(surface.columns) != list(grid.columns)


def test_surface_grid_reindexes_mismatched_peer_shapes_before_stacking(monkeypatch):
    from analysis.unified_weights import surface_feature_matrix

    asof = pd.Timestamp("2024-01-01")
    grids = {
        "TGT": {
            asof: pd.DataFrame(
                [[0.20, 0.21], [0.22, 0.23]],
                index=["0.90-1.00", "1.00-1.10"],
                columns=[30, 60],
            )
        },
        "P1": {
            asof: pd.DataFrame(
                [[0.24], [0.25], [0.26]],
                index=["0.80-0.90", "0.90-1.00", "1.00-1.10"],
                columns=[30],
            )
        },
        "P2": {
            asof: pd.DataFrame(
                [[0.27, 0.28, 0.29]],
                index=["0.90-1.00"],
                columns=[30, 60, 90],
            )
        },
    }

    monkeypatch.setattr(
        "analysis.unified_weights.get_surface_grids_cached",
        lambda cfg, key: grids,
        raising=False,
    )
    monkeypatch.setattr(
        "analysis.analysis_pipeline.get_surface_grids_cached",
        lambda cfg, key: grids,
    )

    _grids, X, names = surface_feature_matrix(
        ["TGT", "P1", "P2"],
        "2024-01-01",
        tenors=(30, 60, 90),
        mny_bins=((0.80, 0.90), (0.90, 1.00), (1.00, 1.10)),
        standardize=False,
    )

    assert X.shape == (3, 9)
    assert len(names) == 9
    assert names == [
        "T30_0.80-0.90",
        "T30_0.90-1.00",
        "T30_1.00-1.10",
        "T60_0.80-0.90",
        "T60_0.90-1.00",
        "T60_1.00-1.10",
        "T90_0.80-0.90",
        "T90_0.90-1.00",
        "T90_1.00-1.10",
    ]


def test_surface_missing_policy_drops_sparse_cells():
    from analysis.unified_weights import _apply_surface_missing_policy

    features = pd.DataFrame(
        {
            "shared": [0.20, 0.21, 0.22],
            "two_of_three": [0.30, 0.31, np.nan],
            "sparse": [0.40, np.nan, np.nan],
        },
        index=["TGT", "P1", "P2"],
    )

    dropped, note = _apply_surface_missing_policy(
        features,
        policy="drop_sparse",
        min_coverage=0.70,
    )
    required, _ = _apply_surface_missing_policy(features, policy="require_shared")

    assert list(dropped.columns) == ["shared"]
    assert list(required.columns) == ["shared"]
    assert "drop_sparse kept 1/3" in note
