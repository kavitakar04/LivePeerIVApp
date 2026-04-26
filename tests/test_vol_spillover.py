import pandas as pd
import numpy as np

from analysis.spillover.vol_spillover import (
    compute_responses,
    compute_weights_and_regression,
    run_spillover,
    summarise,
)


def test_compute_responses_horizon_offsets():
    dates = pd.date_range('2023-01-01', periods=4)
    df = pd.DataFrame({
        'date': list(dates) * 2,
        'ticker': ['AAA'] * 4 + ['BBB'] * 4,
        'atm_iv': [100, 120, 110, 115, 50, 55, 60, 65],
    })
    events = pd.DataFrame({
        'ticker': ['AAA'],
        'date': [dates[1]],
        'rel_change': [0.2],
        'sign': [1],
    })
    peers = {'AAA': ['BBB']}
    responses = compute_responses(df, events, peers, horizons=[1, 2])
    result = responses.sort_values('h')['peer_pct'].tolist()
    assert np.allclose(result, [0.2, 0.3])


def test_compute_responses_empty_has_expected_columns():
    dates = pd.date_range('2023-01-01', periods=2)
    df = pd.DataFrame({
        'date': list(dates) * 2,
        'ticker': ['AAA'] * 2 + ['BBB'] * 2,
        'atm_iv': [100, 101, 50, 51],
    })
    out = compute_responses(
        df,
        pd.DataFrame(columns=["ticker", "date", "rel_change", "sign"]),
        {"AAA": ["BBB"]},
        horizons=[1],
    )

    assert list(out.columns) == ["ticker", "peer", "t0", "h", "trigger_pct", "peer_pct", "sign"]
    assert out.empty


def test_run_spillover_uses_full_lookback_window_not_top_20(monkeypatch, tmp_path):
    dates = pd.date_range("2024-01-01", periods=40, freq="D")
    aaa = 100.0 * (1.02 ** np.arange(len(dates)))
    bbb = 50.0 * (1.01 ** np.arange(len(dates)))
    df = pd.DataFrame({
        "date": list(dates) * 2,
        "ticker": ["AAA"] * len(dates) + ["BBB"] * len(dates),
        "atm_iv": np.concatenate([aaa, bbb]),
    })

    monkeypatch.setattr(
        "analysis.spillover.vol_spillover.get_groups_for_target",
        lambda ticker, conn=None: ["grp"] if ticker == "AAA" else [],
    )
    monkeypatch.setattr(
        "analysis.spillover.vol_spillover.load_ticker_group",
        lambda name, conn=None: {"peer_tickers": ["BBB"]} if name == "grp" else None,
    )

    result = run_spillover(
        df,
        tickers=["AAA", "BBB"],
        threshold=0.01,
        lookback=30,
        horizons=[1],
        events_path=str(tmp_path / "events.parquet"),
        summary_path=str(tmp_path / "summary.parquet"),
    )

    aaa_events = result["events"][result["events"]["ticker"] == "AAA"]
    assert len(aaa_events) > 20
    assert aaa_events["date"].min() > dates.max() - pd.Timedelta(days=30)
    assert aaa_events["date"].max() == dates.max()


def test_run_spillover_explicit_universe_populates_peer_trigger_responses(monkeypatch, tmp_path):
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    df = pd.DataFrame({
        "date": list(dates) * 3,
        "ticker": ["AAA"] * 5 + ["BBB"] * 5 + ["CCC"] * 5,
        "atm_iv": [
            100, 100, 100, 100, 100,
            50, 60, 60, 60, 60,
            30, 33, 36, 39, 42,
        ],
    })

    def fake_get_groups(ticker, conn=None):
        return ["grp"] if ticker == "AAA" else []

    def fake_load_group(name, conn=None):
        return {"peer_tickers": ["BBB", "CCC"]} if name == "grp" else None

    monkeypatch.setattr(
        "analysis.spillover.vol_spillover.get_groups_for_target",
        fake_get_groups,
    )
    monkeypatch.setattr(
        "analysis.spillover.vol_spillover.load_ticker_group",
        fake_load_group,
    )

    result = run_spillover(
        df,
        tickers=["AAA", "BBB", "CCC"],
        threshold=0.10,
        lookback=30,
        horizons=[1],
        events_path=str(tmp_path / "events.parquet"),
        summary_path=str(tmp_path / "summary.parquet"),
    )

    bbb_event_responses = result["responses"][
        (result["responses"]["ticker"] == "BBB")
        & (result["responses"]["t0"] == dates[1])
    ]

    assert set(bbb_event_responses["peer"]) == {"AAA", "CCC"}
    assert not result["summary"][
        (result["summary"]["ticker"] == "BBB")
        & (result["summary"]["peer"].isin(["AAA", "CCC"]))
    ].empty


def test_summarise_adds_statistical_context():
    responses = pd.DataFrame({
        "ticker": ["AAA"] * 6,
        "peer": ["BBB"] * 6,
        "t0": pd.date_range("2024-01-01", periods=6),
        "h": [1] * 6,
        "trigger_pct": [0.10] * 6,
        "peer_pct": [0.04, 0.05, 0.05, 0.06, 0.05, 0.04],
        "sign": [1] * 6,
    })
    baseline = pd.DataFrame({
        "ticker": ["AAA"] * 20,
        "peer": ["BBB"] * 20,
        "h": [1] * 20,
        "peer_pct": np.zeros(20),
    })

    summary = summarise(
        responses,
        threshold=0.02,
        baseline=baseline,
        n_boot=50,
        n_perm=99,
        random_state=1,
    )
    row = summary.iloc[0]

    assert row["hit_rate"] == 1.0
    assert row["sign_concord"] == 1.0
    assert row["median_resp"] > 0.0
    assert row["baseline_median_resp"] == 0.0
    assert row["median_abnormal_resp"] == row["median_resp"]
    assert np.isfinite(row["median_resp_ci_low"])
    assert np.isfinite(row["median_resp_ci_high"])
    assert np.isfinite(row["p_value"])
    assert np.isfinite(row["q_value"])
    assert row["strength"] == "Strong"


def test_weighting_and_regression_90d(monkeypatch):
    dates = pd.date_range('2023-01-01', periods=100)
    base = 100.0
    t_ret = np.sin(np.linspace(0, 10, 100)) * 0.01
    p1_ret = 0.6 * t_ret + 0.001  # high correlation
    p2_ret = 0.2 * t_ret + 0.001 * np.cos(np.linspace(0, 3, 100))

    def build_iv(r):
        return base * np.exp(np.cumsum(r))

    df = pd.DataFrame({
        'date': list(dates) * 3,
        'ticker': ['AAA'] * 100 + ['BBB'] * 100 + ['CCC'] * 100,
        'atm_iv': np.concatenate([
            build_iv(t_ret),
            build_iv(p1_ret),
            build_iv(p2_ret),
        ]),
    })

    def fake_get_groups(ticker, conn=None):
        return ['grp'] if ticker == 'AAA' else []

    def fake_load_group(name, conn=None):
        return {'peer_tickers': ['BBB', 'CCC']} if name == 'grp' else None

    monkeypatch.setattr(
        'analysis.spillover.vol_spillover.get_groups_for_target', fake_get_groups
    )
    monkeypatch.setattr(
        'analysis.spillover.vol_spillover.load_ticker_group', fake_load_group
    )

    weights, betas = compute_weights_and_regression(df, 'AAA', window=90)

    assert np.isclose(weights.sum(), 1.0)
    assert weights['BBB'] > weights['CCC']
    assert np.isclose(betas['BBB'], 0.6, atol=0.05)
    assert np.isclose(betas['CCC'], 0.2, atol=0.05)
