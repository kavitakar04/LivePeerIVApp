import numpy as np
import pandas as pd


def _synthetic_smile():
    S = 100.0
    T = 30.0 / 365.25
    K = S * np.linspace(0.8, 1.2, 9)
    k = np.log(K / S)
    iv = 0.22 - 0.04 * k + 0.25 * k * k
    return S, T, K, iv


def test_sabr_synthetic_smile_sanity():
    from volModel.sabrFit import fit_sabr_slice, sabr_smile_iv

    S, T, K, iv = _synthetic_smile()
    params = fit_sabr_slice(S, K, T, iv)
    pred = sabr_smile_iv(S, K, T, params)
    rmse = float(np.sqrt(np.mean((pred - iv) ** 2)))

    assert params["quality_ok"] is True
    assert np.isfinite(pred).all()
    assert np.all(pred > 0)
    assert rmse < 0.02


def test_volmodel_tps_explicit_dispatch():
    from volModel.models import SUPPORTED_MODELS
    from volModel.volModel import VolModel

    S, T, K, iv = _synthetic_smile()
    vm = VolModel(model="tps").fit(S, K, np.full_like(K, T), iv)
    pred = vm.smile(K, T)

    assert "tps" in SUPPORTED_MODELS
    assert vm.model == "tps"
    assert vm.available_expiries()
    assert np.isfinite(pred).all()
    assert np.all(pred > 0)


def test_pillar_svi_does_not_silently_fallback():
    from analysis.atm_extraction import fit_smile_get_atm

    S, T, K, iv = _synthetic_smile()
    g = pd.DataFrame(
        {
            "S": S,
            "K": K,
            "T": T,
            "moneyness": K / S,
            "sigma": iv,
        }
    )

    out = fit_smile_get_atm(g, model="svi", vega_weighted=False)

    assert out["model"] == "svi"
    assert np.isfinite(out["atm_vol"])
    assert np.isfinite(out["skew"])
    assert np.isfinite(out["curv"])
    assert out["rmse"] < 0.02


def test_auto_rejects_broken_sabr_and_uses_valid_fallback(monkeypatch):
    import analysis.atm_extraction as atm_extraction
    import analysis.pillars as pillars

    S, T, K, iv = _synthetic_smile()
    g = pd.DataFrame(
        {
            "S": S,
            "K": K,
            "T": T,
            "moneyness": K / S,
            "sigma": iv,
        }
    )

    monkeypatch.setattr(atm_extraction, "_HAS_SVI", False)
    monkeypatch.setattr(atm_extraction, "_HAS_SABR", True)

    def broken_fit_sabr(**_kwargs):
        return {"alpha": 1.0, "beta": 0.5, "rho": 0.0, "nu": 1.0, "rmse": 1e9, "n": len(K)}

    def broken_sabr_iv(_S, Kq, _T, _params):
        return -np.ones_like(np.asarray(Kq, dtype=float))

    monkeypatch.setattr(atm_extraction, "_fit_sabr_slice", broken_fit_sabr)
    monkeypatch.setattr(atm_extraction, "_sabr_iv", broken_sabr_iv)

    out = pillars._fit_smile_get_atm(g, model="auto", vega_weighted=False)

    assert out["model"] in {"tps", "poly2"}
    assert out["model"] != "sabr"
    assert np.isfinite(out["atm_vol"])
    assert out["rmse"] < 0.02


def test_quality_gate_rejects_absurd_positive_iv():
    from volModel.quality import validate_model_fit

    params = {"rmse": 0.01, "n": 4}
    quality = validate_model_fit(
        "sabr",
        params,
        lambda _p: np.array([0.2, 0.25, 6.0, 0.3]),
        iv_obs=np.array([0.2, 0.25, 0.28, 0.3]),
    )

    assert quality.ok is False
    assert "too high" in quality.reason
    assert quality.max_iv == 6.0


def test_repeated_model_fallback_warning_is_rate_limited(caplog):
    import logging
    import volModel.quality as quality_mod
    from volModel.quality import ModelQuality, warn_model_fallback

    quality_mod._FALLBACK_WARNING_KEYS.clear()
    q = ModelQuality(False, "RMSE too high: 0.3", rmse=0.3, n=10)

    with caplog.at_level(logging.DEBUG, logger="volModel.quality"):
        warn_model_fallback(
            requested_model="sabr",
            failed_model="sabr",
            fallback_model="none",
            message=q.reason,
            quality=q,
        )
        warn_model_fallback(
            requested_model="sabr",
            failed_model="sabr",
            fallback_model="none",
            message=q.reason,
            quality=q,
        )

    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    debug_records = [r for r in caplog.records if r.levelno == logging.DEBUG]

    assert len(warning_records) == 1
    assert len(debug_records) == 1
