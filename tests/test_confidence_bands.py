import numpy as np

from analysis.confidence_bands import bootstrap_bands, normalize_confidence_level, residual_bootstrap_bands


def test_normalize_confidence_level_accepts_percent_or_decimal():
    assert normalize_confidence_level(0.68) == 0.68
    assert normalize_confidence_level(68) == 0.68


def test_bootstrap_bands_normalizes_percent_level():
    x = np.arange(5, dtype=float)
    y = 0.2 + 0.01 * x
    grid = np.array([2.0])

    bands = bootstrap_bands(
        x,
        y,
        fit_fn=lambda xb, yb: {"mean": float(np.mean(yb))},
        pred_fn=lambda params, xq: np.full_like(xq, params["mean"], dtype=float),
        grid=grid,
        level=68,
        n_boot=20,
    )

    assert bands.level == 0.68
    assert np.isfinite(bands.lo).all()
    assert np.isfinite(bands.hi).all()


def test_residual_bootstrap_keeps_fixed_x_design():
    x = np.linspace(80.0, 120.0, 9)
    y = 0.25 + 0.001 * (x - 100.0) + 0.0002 * (x - 100.0) ** 2
    y = y + np.array([0.001, -0.001, 0.0, 0.0005, -0.0005, 0.001, -0.001, 0.0, 0.0005])
    grid = np.linspace(85.0, 115.0, 5)
    seen_x = []

    def fit_fn(xb, yb):
        seen_x.append(np.asarray(xb, dtype=float).copy())
        return {"coeff": np.polyfit(xb, yb, 2)}

    def pred_fn(params, xq):
        return np.polyval(params["coeff"], xq)

    bands = residual_bootstrap_bands(x, y, fit_fn, pred_fn, grid, level=0.68, n_boot=12)

    assert bands.level == 0.68
    assert len(seen_x) == 13
    assert all(np.array_equal(xb, x) for xb in seen_x)
    assert np.isfinite(bands.lo).all()
    assert np.isfinite(bands.hi).all()
