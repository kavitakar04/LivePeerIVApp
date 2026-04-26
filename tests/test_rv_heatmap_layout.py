import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from display.plotting.rv_plots import plot_surface_residual_heatmap


def test_rv_heatmap_redraw_keeps_main_axes_position_stable():
    residual = pd.DataFrame(
        [[-1.0, 0.2, 1.1], [0.4, -0.3, 0.8]],
        index=["0.90-1.00", "1.00-1.10"],
        columns=[30, 60, 90],
    )
    fig, ax = plt.subplots()

    plot_surface_residual_heatmap(ax, residual, title="RV")
    first_bounds = ax.get_position().bounds
    first_axes = len(fig.axes)

    plot_surface_residual_heatmap(ax, residual, title="RV")
    second_bounds = ax.get_position().bounds
    second_axes = len(fig.axes)

    assert second_bounds == first_bounds
    assert second_axes == first_axes == 2
    assert hasattr(fig, "_rv_heatmap_colorbar")
    assert hasattr(fig, "_rv_heatmap_colorbar_ax")

    plt.close(fig)
