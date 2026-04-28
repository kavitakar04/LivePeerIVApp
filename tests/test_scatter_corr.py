import sys
from pathlib import Path

import pandas as pd

# Ensure project root on path for module imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from display.plotting.charts.correlation_detail_plot import scatter_corr_matrix


def test_scatter_corr_subset_columns():
    df = pd.DataFrame({
        'a': [1, 2, 3, 4],
        'b': [2, 4, 6, 8],
        'c': [10, 11, 12, 13],
    })
    corr = scatter_corr_matrix(df, columns=['a', 'b'], plot=False)
    assert list(corr.columns) == ['a', 'b']
    assert corr.loc['a', 'b'] == 1.0
