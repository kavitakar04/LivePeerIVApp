import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from unittest.mock import patch

from display.gui.controllers.gui_plot_manager import PlotManager


def test_term_plot_populates_last_fit_info():
    pm = PlotManager()
    fig, ax = plt.subplots()
    try:
        atm_curve = pd.DataFrame({
            'T': [0.5, 1.0],
            'atm_vol': [0.2, 0.25],
            'skew': [0.01, 0.02],
            'curv': [0.0, -0.01],
            'expiry': ['2024-06-01', '2024-12-01'],
        })
        data = {'atm_curve': atm_curve}
        with patch('display.gui.controllers.gui_plot_manager.compute_or_load', return_value=data):
            settings = {
                'plot_type': 'Term ATM',
                'target': 'XYZ',
                'asof': '2024-01-01',
                'model': 'svi',
                'T_days': 30,
                'ci': 68,
                'x_units': 'years',
                'peers': [],
                'pillars': [30],
                'overlay_synth': False,
                'overlay_peers': False,
                'max_expiries': 2,
                'atm_band': 0.05,
                'weight_method': 'corr',
                'feature_mode': 'iv_atm',
            }
            pm.plot(ax, settings)
        info = pm.last_fit_info
        assert info['ticker'] == 'XYZ'
        assert 0.5 in info['fit_by_expiry']
        assert info['fit_by_expiry'][0.5]['sens']['atm_vol'] == 0.2
    finally:
        plt.close(fig)
