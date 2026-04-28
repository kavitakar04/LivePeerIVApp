import pandas as pd
from pathlib import Path
import tempfile
import analysis.weights.beta_builder as beta_builder
from analysis import analysis_pipeline

def test_save_correlations_writes_parquet(monkeypatch):
    sample = pd.Series({'AAA': 0.1, 'BBB': 0.2})
    monkeypatch.setattr(beta_builder, 'build_vol_betas', lambda **kwargs: sample)
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = beta_builder.save_correlations(mode='demo', benchmark='SPY', base_path=tmpdir)
        assert paths and Path(paths[0]).exists()
        df = pd.read_parquet(paths[0])
        assert set(df.columns) == {'ticker', 'beta'}
        assert set(df['ticker']) == {'AAA', 'BBB'}


def test_save_betas_surface_writes_parquet(monkeypatch):
    sample = pd.Series({'AAA': 0.3, 'BBB': 0.4})
    monkeypatch.setattr(analysis_pipeline, 'build_vol_betas', lambda *args, **kwargs: sample)
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = analysis_pipeline.save_betas(mode='surface', benchmark='SPY', base_path=tmpdir)
        assert paths and Path(paths[0]).exists()
        df = pd.read_parquet(paths[0])
        assert set(df.columns) == {'ticker', 'beta'}
        assert set(df['ticker']) == {'AAA', 'BBB'}
