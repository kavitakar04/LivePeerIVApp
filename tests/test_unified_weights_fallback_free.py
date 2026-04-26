import pandas as pd
import pytest

from analysis.unified_weights import UnifiedWeightComputer, WeightConfig, WeightMethod, FeatureSet


def test_missing_target_data_raises(monkeypatch):
    uwc = UnifiedWeightComputer()
    cfg = WeightConfig(method=WeightMethod.CORRELATION, feature_set=FeatureSet.ATM)
    # Patch _choose_asof directly so no date is found, bypassing DB and global lookups.
    monkeypatch.setattr(UnifiedWeightComputer, '_choose_asof', lambda self, *a, **kw: None)
    with pytest.raises(ValueError, match="no surface/ATM date available"):
        uwc.compute_weights('TGT', ['P1', 'P2'], cfg)


def test_empty_feature_matrix_falls_back_to_equal_weights(monkeypatch):
    uwc = UnifiedWeightComputer()
    cfg = WeightConfig(method=WeightMethod.CORRELATION, feature_set=FeatureSet.ATM)
    monkeypatch.setattr(UnifiedWeightComputer, '_choose_asof', lambda self, *a, **kw: '2024-01-01')
    monkeypatch.setattr(
        UnifiedWeightComputer,
        '_build_feature_matrix',
        lambda self, target, peers, asof, config: pd.DataFrame()
    )
    result = uwc.compute_weights('TGT', ['P1', 'P2'], cfg)
    assert set(result.index) == {'P1', 'P2'}
    assert result.sum() == pytest.approx(1.0)
    assert 'weight_warning' in result.attrs
    assert 'empty' in result.attrs['weight_warning'].lower()
