import numpy as np
import pandas as pd


from analysis.weights.beta_builder import cosine_similarity_weights, build_peer_weights


def test_cosine_ul_weights(monkeypatch):
    def fake_returns(conn_fn):
        return pd.DataFrame({
            'TGT': [0.1, 0.2],
            'P1': [0.1, 0.2],
            'P2': [-0.1, 0.0],
        })
    monkeypatch.setattr('analysis.weights.beta_builder._underlying_log_returns', fake_returns)
    w = cosine_similarity_weights(
        get_smile_slice=None,
        mode='cosine_ul',
        target='TGT',
        peers=['P1', 'P2'],
        asof='2024-01-01',
    )
    assert np.isclose(w.sum(), 1.0)
    assert (w >= 0).all()
    assert w.idxmax() == 'P1'


def test_cosine_surface_weights(monkeypatch):
    def fake_surface_feature_matrix(tickers, asof, tenors=None, mny_bins=None, standardize=True):
        ok = [t.upper() for t in tickers]
        grids = {t: None for t in ok}
        X = np.array([[1, 0], [1, 0], [0, 1]], dtype=float)
        names = ['f1', 'f2']
        return grids, X, names
    monkeypatch.setattr('analysis.weights.beta_builder.surface_feature_matrix', fake_surface_feature_matrix)
    w = cosine_similarity_weights(
        get_smile_slice=None,
        mode='cosine_surface',
        target='TGT',
        peers=['P1', 'P2'],
        asof='2024-01-01',
    )
    assert np.isclose(w.sum(), 1.0)
    assert (w >= 0).all()
    assert w.idxmax() == 'P1'


def test_cosine_ul_vol_weights(monkeypatch):
    def fake_vol_series(conn_fn, window=21, min_obs=10, demean=False):
        return pd.DataFrame({'TGT': [0.2, 0.3], 'P1': [0.2, 0.3], 'P2': [0.1, 0.0]})
    monkeypatch.setattr('analysis.weights.beta_builder._underlying_vol_series', fake_vol_series)
    w = build_peer_weights('cosine', 'ul_vol', 'TGT', ['P1', 'P2'])
    assert np.isclose(w.sum(), 1.0)
    assert (w >= 0).all()
    assert w.idxmax() == 'P1'


def test_corr_surface_vector_weights(monkeypatch):
    def fake_surface_feature_matrix(tickers, asof, tenors=None, mny_bins=None, standardize=True):
        ok = [t.upper() for t in tickers]
        grids = {t: None for t in ok}
        X = np.array([[1, 0], [1, 0], [0, 1]], dtype=float)
        names = ['f1', 'f2']
        return grids, X, names
    monkeypatch.setattr('analysis.weights.beta_builder.surface_feature_matrix', fake_surface_feature_matrix)
    w = build_peer_weights('corr', 'surface_vector', 'TGT', ['P1', 'P2'], asof='2024-01-01')
    assert np.isclose(w.sum(), 1.0)
    assert (w >= 0).all()
    assert w.idxmax() == 'P1'


def test_top_k_cosine(monkeypatch):
    def fake_returns(conn_fn):
        return pd.DataFrame({'TGT': [0.1, 0.2], 'P1': [0.1, 0.2], 'P2': [0.1, 0.2]})
    monkeypatch.setattr('analysis.weights.beta_builder._underlying_log_returns', fake_returns)
    w = build_peer_weights('cosine', 'ul_px', 'TGT', ['P1', 'P2'], k=1)
    assert np.isclose(w.sum(), 1.0)
    assert (w > 0).sum() == 1

