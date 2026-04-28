import sys
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from display.plotting.charts.correlation_detail_plot import plot_correlation_details


def test_repeated_correlation_plot_keeps_axes_position():
    corr_df = pd.DataFrame(
        [[1.0, 0.5], [0.5, 1.0]],
        columns=["A", "B"],
        index=["A", "B"],
    )
    fig, ax = plt.subplots()
    plot_correlation_details(ax, corr_df, show_values=False)
    pos1 = ax.get_position().bounds
    plot_correlation_details(ax, corr_df, show_values=False)
    pos2 = ax.get_position().bounds
    assert pos1 == pos2
