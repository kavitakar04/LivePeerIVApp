import numpy as np
import pandas as pd

from display.gui.spillover_gui import (
    compute_rolling_spillover_signal,
    compute_trigger_event_response,
    prepare_spillover_summary_display,
    to_plot_percent,
)


def test_compute_trigger_event_response_uses_same_event_window_as_peers():
    dates = pd.date_range("2024-01-01", periods=5)
    df = pd.DataFrame({
        "date": list(dates) * 2,
        "ticker": ["AAA"] * 5 + ["BBB"] * 5,
        "atm_iv": [
            100.0, 120.0, 130.0, 150.0, 160.0,
            50.0, 51.0, 52.0, 53.0, 54.0,
        ],
    })

    out = compute_trigger_event_response(df, "AAA", dates[1], horizons=[1, 3])

    assert list(out["h"]) == [1, 3]
    assert np.allclose(out["response"], [0.30, 0.60])


def test_compute_trigger_event_response_rejects_missing_prior_base():
    dates = pd.date_range("2024-01-01", periods=3)
    df = pd.DataFrame({
        "date": dates,
        "ticker": ["AAA"] * 3,
        "atm_iv": [100.0, 120.0, 130.0],
    })

    out = compute_trigger_event_response(df, "AAA", dates[0], horizons=[1])

    assert list(out.columns) == ["h", "response"]
    assert out.empty


def test_compute_rolling_spillover_signal_uses_event_based_windows():
    dates = pd.to_datetime([
        "2024-01-10",
        "2024-01-12",
        "2024-02-20",
        "2024-04-01",
    ])
    responses = pd.DataFrame({
        "ticker": ["AAA"] * 4,
        "peer": ["BBB"] * 4,
        "t0": dates,
        "h": [5] * 4,
        "trigger_pct": [0.10, 0.10, 0.10, 0.10],
        "peer_pct": [0.02, 0.06, -0.04, 0.10],
        "sign": [1, 1, -1, 1],
    })
    summary = pd.DataFrame({
        "ticker": ["AAA"],
        "peer": ["BBB"],
        "h": [5],
        "baseline_median_resp": [0.01],
    })

    out = compute_rolling_spillover_signal(
        responses,
        summary,
        "AAA",
        "BBB",
        5,
        window=2,
    )

    assert list(out["date"]) == list(dates)
    assert list(out["event_count"]) == [1, 2, 2, 2]
    assert np.allclose(
        out["rolling_median_peer_response"],
        [0.02, 0.04, 0.01, 0.03],
    )
    assert np.allclose(
        out["rolling_abnormal_response"],
        [0.01, 0.03, 0.00, 0.02],
    )
    assert np.allclose(
        out["rolling_same_direction_probability"],
        [1.0, 1.0, 1.0, 1.0],
    )


def test_compute_rolling_spillover_signal_requires_baseline():
    responses = pd.DataFrame({
        "ticker": ["AAA"],
        "peer": ["BBB"],
        "t0": [pd.Timestamp("2024-01-10")],
        "h": [1],
        "trigger_pct": [0.10],
        "peer_pct": [0.02],
        "sign": [1],
    })
    summary = pd.DataFrame({
        "ticker": ["AAA"],
        "peer": ["BBB"],
        "h": [1],
        "baseline_median_resp": [np.nan],
    })

    out = compute_rolling_spillover_signal(responses, summary, "AAA", "BBB", 1)

    assert list(out.columns) == [
        "date",
        "rolling_median_peer_response",
        "rolling_abnormal_response",
        "rolling_same_direction_probability",
        "event_count",
    ]
    assert out.empty


def test_prepare_spillover_summary_display_does_not_truncate_rows():
    summary = pd.DataFrame({
        "ticker": ["AAA"] * 60,
        "peer": [f"P{i:02d}" for i in range(60)],
        "h": [1] * 60,
        "hit_rate": np.linspace(0.0, 0.59, 60),
    })

    out = prepare_spillover_summary_display(summary)

    assert len(out) == 60
    assert out.iloc[0]["hit_rate"] == summary["hit_rate"].max()
    assert out.iloc[-1]["hit_rate"] == summary["hit_rate"].min()


def test_to_plot_percent_matches_table_percentage_units():
    out = to_plot_percent(pd.Series([-0.0824, 0.125]))

    assert np.allclose(out, [-8.24, 12.5])
