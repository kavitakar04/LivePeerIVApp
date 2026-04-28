import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from volModel.polyFit import fit_tps_slice
from analysis.surfaces.confidence_bands import tps_confidence_bands
from display.plotting.charts.smile_plot import fit_and_plot_smile


def test_tps_smile_plot_runs():
    S = 100.0
    K = S * np.array([0.9, 1.0, 1.1, 1.2])
    T = 0.5
    log_mny = np.log(K / S)
    iv = 0.2 + 0.05 * log_mny ** 2
    params = fit_tps_slice(S, K, T, iv)
    grid = np.linspace(0.8, 1.2, 21) * S
    bands = tps_confidence_bands(S, K, T, iv, grid, n_boot=5)
    fig, ax = plt.subplots()
    result = fit_and_plot_smile(
        ax,
        S=S,
        K=K,
        T=T,
        iv=iv,
        model="tps",
        params=params,
        bands=bands,
    )
    assert result["params"]
    assert bands.mean.shape == grid.shape
    plt.close(fig)

