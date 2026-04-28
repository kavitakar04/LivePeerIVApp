import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from display.plotting.charts.correlation_detail_plot import plot_correlation_details


def test_coverage_summary_outside_axes():
    corr_df = pd.DataFrame(
        [[1.0, np.nan], [np.nan, 1.0]], columns=["A", "B"], index=["A", "B"]
    )
    fig, ax = plt.subplots()
    plot_correlation_details(ax, corr_df, show_values=False)
    # Find text artist created for matrix coverage and ensure it's above the axes
    found = False
    for txt in ax.texts:
        if "finite cells" in txt.get_text().lower():
            found = True
            assert txt.get_position()[1] > 1.0
    assert found


def test_weight_legend_separate_axis():
    corr_df = pd.DataFrame(
        [[1.0, 0.5], [0.5, 1.0]], columns=["A", "B"], index=["A", "B"]
    )
    weights = pd.Series({"A": 0.6, "B": 0.4})
    fig, ax = plt.subplots()
    plot_correlation_details(ax, corr_df, weights=weights, show_values=False)
    assert hasattr(ax.figure, "_corr_weight_ax")
    weight_ax = ax.figure._corr_weight_ax
    # Ensure weight axis sits to the right of the main plot
    assert weight_ax.get_position().x0 >= ax.get_position().x1
    assert "Peer Weights" in weight_ax.get_title()


def test_atm_coverage_does_not_create_side_bar_plot():
    corr_df = pd.DataFrame(
        [[1.0, 0.5], [0.5, 1.0]], columns=["A", "B"], index=["A", "B"]
    )
    atm_df = pd.DataFrame(
        [[0.2, 0.21, 0.22], [0.3, 0.31, np.nan]],
        columns=[1, 2, 3],
        index=["A", "B"],
    )
    fig, ax = plt.subplots()
    plot_correlation_details(ax, corr_df, show_values=False, atm_df=atm_df)

    assert hasattr(fig, "_corr_colorbar_ax")
    assert not hasattr(fig, "_corr_coverage_ax")
