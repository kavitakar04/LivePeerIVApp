import sqlite3
from unittest.mock import patch

from data.db_utils import ensure_initialized, insert_quotes
from analysis.weights.unified_weights import (
    _weight_computer,
    WeightConfig,
    FeatureSet,
    WeightMethod,
)


def test_choose_asof_includes_target():
    conn = sqlite3.connect(':memory:', detect_types=sqlite3.PARSE_DECLTYPES)
    ensure_initialized(conn)
    quotes = [
        {
            'asof_date': '2024-01-01',
            'ticker': 'TGT',
            'expiry': '2024-02-01',
            'K': 100.0,
            'call_put': 'C',
            'sigma': 0.2,
            'S': 100.0,
            'T': 0.1,
            'moneyness': 1.0,
            'is_atm': 1,
        },
        {
            'asof_date': '2024-01-01',
            'ticker': 'P1',
            'expiry': '2024-02-01',
            'K': 100.0,
            'call_put': 'C',
            'sigma': 0.25,
            'S': 100.0,
            'T': 0.1,
            'moneyness': 1.0,
            'is_atm': 1,
        },
        {
            'asof_date': '2024-01-02',
            'ticker': 'P1',
            'expiry': '2024-02-01',
            'K': 100.0,
            'call_put': 'C',
            'sigma': 0.25,
            'S': 100.0,
            'T': 0.1,
            'moneyness': 1.0,
            'is_atm': 1,
        },
        {
            'asof_date': '2024-01-02',
            'ticker': 'P2',
            'expiry': '2024-02-01',
            'K': 100.0,
            'call_put': 'C',
            'sigma': 0.3,
            'S': 100.0,
            'T': 0.1,
            'moneyness': 1.0,
            'is_atm': 1,
        },
    ]
    insert_quotes(conn, quotes)
    cfg = WeightConfig(method=WeightMethod.CORRELATION, feature_set=FeatureSet.SURFACE)
    with patch('data.db_utils.get_conn', return_value=conn):
        chosen = _weight_computer._choose_asof('TGT', ['P1', 'P2'], cfg)
    assert chosen == '2024-01-01'
