from display.gui.parameters_tab import flatten_diagnostics_info, flatten_fit_info


def _sample_info():
    return {
        "ticker": "IONQ",
        "asof": "2026-04-25",
        "fit_by_expiry": {
            30 / 365.25: {
                "expiry": "2026-05-25 00:00:00",
                "sens": {"atm_vol": 0.76, "skew": -0.19, "curv": 0.04},
                "svi": {"a": 0.1, "b": 0.2, "rmse": 0.011, "n": 18},
                "sabr": {},
                "tps": {"coef0": 0.7, "rmse": 0.014, "n": 18},
                "quality": {
                    "svi": {"ok": True, "reason": "", "rmse": 0.011, "min_iv": 0.5, "max_iv": 1.2, "n": 18},
                    "sabr": {"ok": False, "reason": "RMSE too high", "rmse": 0.8, "min_iv": 0.4, "max_iv": 1.6, "n": 18},
                    "tps": {"ok": True, "reason": "", "rmse": 0.014, "min_iv": 0.5, "max_iv": 1.1, "n": 18},
                },
                "fallback": {"svi": "none", "sabr": "none", "tps": "none"},
            }
        },
    }


def test_parameter_summary_is_one_row_per_expiry():
    rows = flatten_fit_info(_sample_info())

    assert len(rows) == 1
    assert rows[0] == {
        "Ticker": "IONQ",
        "As Of": "2026-04-25",
        "DTE": 30,
        "Expiry": "2026-05-25",
        "ATM Vol": 0.76,
        "Skew": -0.19,
        "Curvature": 0.04,
        "Fit Quality": "Good",
    }


def test_parameter_summary_excludes_diagnostics_and_weights():
    rows = flatten_fit_info({
        "ticker": "IONQ",
        "asof": "2026-04-25",
        "fit_by_expiry": {},
        "status_events": [{"category": "model_fit", "status": "warning", "message": "RMSE high"}],
        "weight_info": {
            "target": "IONQ",
            "asof": "2026-04-25",
            "mode": "corr_iv_atm",
            "weights": {"RKLB": 0.7, "JOBY": 0.3},
        },
    })

    assert rows == []


def test_diagnostics_info_exposes_model_quality_and_fallback_status():
    rows = flatten_diagnostics_info(_sample_info())

    assert [row["Model"] for row in rows] == ["svi", "sabr", "tps"]
    assert rows[0]["Status"] == "ok"
    assert rows[0]["Params"] == "a=0.1, b=0.2"

    sabr = rows[1]
    assert sabr["Status"] == "rejected"
    assert sabr["Reason"] == "RMSE too high"
    assert sabr["RMSE"] == 0.8
    assert sabr["N"] == 18


def test_diagnostics_info_includes_peer_weights_and_status_events():
    rows = flatten_diagnostics_info({
        "ticker": "IONQ",
        "asof": "2026-04-25",
        "fit_by_expiry": {},
        "status_events": [
            {
                "category": "expiry_alignment",
                "status": "warning",
                "message": "Target and peer maturities do not overlap.",
                "detail": "peer=RGTI",
            }
        ],
        "weight_info": {
            "target": "IONQ",
            "asof": "2026-04-25",
            "mode": "pca_surface",
            "weights": {"RGTI": 1.0},
            "warning": "pca weights failed validation; using equal weights",
        },
    })

    assert rows[0]["Model"] == "expiry_alignment"
    assert rows[0]["Status"] == "warning"
    assert rows[0]["Reason"] == "Target and peer maturities do not overlap."
    assert rows[0]["Params"] == "peer=RGTI"
    assert rows[1]["Model"] == "weight"
    assert rows[1]["Fallback"] == "equal"
