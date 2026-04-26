import pandas as pd

from analysis.feature_health import summarize_feature_health
from display.gui.parameters_tab import (
    build_model_health_grid,
    flatten_diagnostics_info,
    flatten_fit_info,
    summarize_health_info,
)


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


def test_system_health_summary_surfaces_failures_and_reliable_models():
    health = summarize_health_info(_sample_info())

    assert health["overall"] == "Degraded"
    assert health["failures"] == 1
    assert health["primary_model"] == "SVI"
    assert "SVI" in health["reliable_models"]
    assert "TPS" in health["reliable_models"]
    assert any("failure" in msg for msg in health["messages"])


def test_model_health_grid_pivots_models_by_expiry():
    health = build_model_health_grid(_sample_info())

    assert [exp["label"] for exp in health["expiries"]] == ["30d"]
    rows = {row["Model"]: row for row in health["rows"]}
    assert rows["SVI"]["30d"] == "✅ ok"
    assert rows["SABR"]["30d"] == "⚠ degraded"
    assert rows["SABR"]["_reasons"]["30d"] == "RMSE too high"
    assert rows["TPS"]["30d"] == "✅ ok"
    assert health["primary_model"] == "SVI"


def test_feature_health_summarizes_grid_distribution_and_pairs():
    features = pd.DataFrame(
        [
            [0.20, 0.21, 0.22, 0.23],
            [0.19, 0.20, 0.21, 0.22],
            [0.80, 0.10, 0.80, 0.10],
        ],
        index=["TGT", "P1", "P2"],
        columns=["T30_0.90-1.00", "T30_1.00-1.10", "T60_0.90-1.00", "T60_1.00-1.10"],
    )
    features.attrs["feature_diagnostics"] = {
        "feature_set": "surface_grid",
        "coordinate_system": "standardized_tenor_grid_x_moneyness_bin",
        "normalization": "column_zscore",
        "missing_policy": "column median imputation before grid standardization",
    }

    health = summarize_feature_health(features, target="TGT")

    assert health["available"] is True
    assert health["summary"]["total_points"] == 4
    assert health["alignment"]["shared_grid"] is True
    assert health["distribution"][0]["ticker"] == "TGT"
    assert health["pairs"][0]["ticker"] == "P1"
    assert "normalized using column_zscore" in health["transformation_log"]
