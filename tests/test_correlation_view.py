import numpy as np
import pandas as pd

from analysis.correlation_view import (
    corr_by_expiry_rank,
    coverage_by_ticker,
    finite_cell_summary,
    overlap_counts,
)


def test_correlation_view_prepares_coverage_and_overlap():
    data = {
        "AAA": pd.DataFrame(
            {
                "T": [0.1, 0.2, 0.3],
                "moneyness": [1.0, 1.0, 1.0],
                "sigma": [0.20, 0.22, 0.24],
            }
        ),
        "BBB": pd.DataFrame(
            {
                "T": [0.1, 0.2, 0.3],
                "moneyness": [1.0, 1.0, 1.0],
                "sigma": [0.21, 0.23, 0.25],
            }
        ),
    }

    def get_slice(ticker, **_kwargs):
        return data[ticker]

    atm_df, corr_df = corr_by_expiry_rank(get_slice, ["AAA", "BBB"], "2024-01-01", max_expiries=3)
    coverage = coverage_by_ticker(corr_df, atm_df)
    overlap = overlap_counts(corr_df, atm_df)
    finite_count, total_cells, ratio = finite_cell_summary(corr_df)

    assert list(atm_df.index) == ["AAA", "BBB"]
    assert coverage.to_dict() == {"AAA": 3, "BBB": 3}
    assert overlap.loc["AAA", "BBB"] == 3
    assert np.isclose(corr_df.loc["AAA", "BBB"], 1.0)
    assert finite_count == total_cells == 4
    assert ratio == 1.0
