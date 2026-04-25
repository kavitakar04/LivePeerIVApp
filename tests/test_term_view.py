import numpy as np
import pandas as pd

from analysis.term_view import compute_term_fit_curve, compute_term_spread_curve, term_ci_error


def test_term_view_fit_and_spread_helpers():
    target = pd.DataFrame(
        {
            "T": [0.1, 0.2, 0.3, 0.4],
            "atm_vol": [0.20, 0.22, 0.24, 0.26],
            "atm_lo": [0.19, 0.21, 0.23, 0.25],
            "atm_hi": [0.21, 0.23, 0.25, 0.27],
        }
    )
    synth = pd.DataFrame({"T": [0.1, 0.2, 0.3, 0.4], "atm_vol": [0.19, 0.21, 0.23, 0.25]})

    fit_x, fit_y = compute_term_fit_curve(target, degree=2, points=20)
    spread_x, spread_y = compute_term_spread_curve(target, synth, points=20)
    yerr = term_ci_error(target)

    assert fit_x.size == fit_y.size == 20
    assert spread_x.size == spread_y.size == 20
    assert np.allclose(spread_y, 0.01)
    assert yerr.shape == (2, 4)
