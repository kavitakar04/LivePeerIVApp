import numpy as np
import pandas as pd

from analysis.services.term_data_service import _apply_term_feature_band_policy, prepare_term_data


def test_term_surface_policy_does_not_label_dispersion_as_ci():
    curve = pd.DataFrame(
        {
            "T": [30 / 365.25],
            "atm_vol": [0.25],
            "count": [5],
            "atm_dispersion": [0.04],
            "atm_lo": [np.nan],
            "atm_hi": [np.nan],
        }
    )

    out = _apply_term_feature_band_policy(curve, "surface_grid")

    assert np.isnan(out.loc[0, "atm_lo"])
    assert np.isnan(out.loc[0, "atm_hi"])
    assert out.loc[0, "atm_dispersion"] == 0.04
    assert out.loc[0, "band_source"] == "none_ci_unavailable"


def test_weighted_peer_ci_uses_variance_propagation(monkeypatch):
    import analysis.services.term_data_service as service

    def fake_compute(ticker, asof, *, atm_band, min_boot, ci, max_expiries, method="fit"):
        if ticker == "TGT":
            return pd.DataFrame(
                {
                    "T": [0.1],
                    "atm_vol": [0.20],
                    "count": [5],
                    "atm_lo": [0.19],
                    "atm_hi": [0.21],
                    "atm_dispersion": [0.02],
                }
            )
        return pd.DataFrame(
            {
                "T": [0.1],
                "atm_vol": [0.30],
                "count": [5],
                "atm_lo": [0.28],
                "atm_hi": [0.32],
                "atm_dispersion": [0.03],
            }
        )

    monkeypatch.setattr(service, "_compute_term_atm_curve", fake_compute)

    data = prepare_term_data(
        target="TGT",
        asof="2024-01-01",
        ci=0.68,
        overlay_synth=True,
        peers=["P1", "P2"],
        weights={"P1": 0.5, "P2": 0.5},
        feature_mode="surface_grid",
    )

    synth = data["synth_curve"].iloc[0]
    expected_half_width = np.sqrt((0.5 * 0.02) ** 2 + (0.5 * 0.02) ** 2)
    assert np.isclose(synth["atm_vol"], 0.30)
    assert np.isclose(synth["atm_lo"], 0.30 - expected_half_width)
    assert np.isclose(synth["atm_hi"], 0.30 + expected_half_width)
    assert synth["band_source"] == "weighted_peer_bootstrap_model_fit"
