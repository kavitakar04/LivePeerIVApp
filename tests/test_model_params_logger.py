from concurrent.futures import ThreadPoolExecutor

import pandas as pd


def test_load_model_params_returns_empty_for_corrupt_parquet(tmp_path, monkeypatch):
    import analysis.persistence.model_params_logger as model_params_logger

    store = tmp_path / "model_params.parquet"
    store.write_text("not parquet", encoding="utf-8")
    monkeypatch.setattr(model_params_logger, "STORE_PATH", store)

    df = model_params_logger.load_model_params()

    assert df.empty
    assert list(df.columns) == model_params_logger.PARAM_COLUMNS


def test_append_params_quarantines_corrupt_store_before_write(tmp_path, monkeypatch):
    import analysis.persistence.model_params_logger as model_params_logger

    store = tmp_path / "model_params.parquet"
    store.write_text("not parquet", encoding="utf-8")
    monkeypatch.setattr(model_params_logger, "STORE_PATH", store)

    model_params_logger.append_params(
        "2024-01-01",
        "SPY",
        "2024-02-01",
        "svi",
        {"rmse": 0.01},
    )

    df = pd.read_parquet(store)
    assert len(df) == 1
    assert df.loc[0, "ticker"] == "SPY"
    assert list(tmp_path.glob("model_params.parquet.corrupt*"))


def test_concurrent_append_params_preserves_all_rows(tmp_path, monkeypatch):
    import analysis.persistence.model_params_logger as model_params_logger

    store = tmp_path / "model_params.parquet"
    monkeypatch.setattr(model_params_logger, "STORE_PATH", store)

    def write_params(idx: int) -> None:
        model_params_logger.append_params(
            "2024-01-01",
            f"T{idx:02d}",
            "2024-02-01",
            "svi",
            {"rmse": idx / 1000.0, "a": idx},
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(write_params, range(40)))

    df = pd.read_parquet(store)
    assert len(df) == 80
    assert df["ticker"].nunique() == 40
    assert not list(tmp_path.glob("model_params.parquet.corrupt*"))
    assert not list(tmp_path.glob(".model_params.parquet.*.tmp"))


def test_recover_model_params_merges_readable_quarantines_and_archives(tmp_path):
    from scripts.recover_model_params import recover_model_params

    store = tmp_path / "model_params.parquet"
    pd.DataFrame(
        [
            {
                "asof_date": pd.Timestamp("2024-01-01"),
                "ticker": "SPY",
                "expiry": pd.Timestamp("2024-02-01"),
                "tenor_d": 31,
                "model": "svi",
                "param": "a",
                "value": 0.1,
                "fit_meta": "{}",
            }
        ]
    ).to_parquet(store, index=False)
    pd.DataFrame(
        [
            {
                "asof_date": pd.Timestamp("2024-01-01"),
                "ticker": "QQQ",
                "expiry": pd.Timestamp("2024-02-01"),
                "tenor_d": 31,
                "model": "sabr",
                "param": "alpha",
                "value": 0.2,
                "fit_meta": "{}",
            }
        ]
    ).to_parquet(tmp_path / "model_params.parquet.corrupt.1", index=False)
    (tmp_path / "model_params.parquet.corrupt.2").write_text("not parquet", encoding="utf-8")

    summary = recover_model_params(store=store, archive_root=tmp_path / "archive")

    df = pd.read_parquet(store)
    assert summary.readable_quarantine_files == 1
    assert summary.unreadable_quarantine_files == 1
    assert summary.output_rows == 2
    assert set(df["ticker"]) == {"SPY", "QQQ"}
    assert not list(tmp_path.glob("model_params.parquet.corrupt*"))
    assert summary.archive_dir is not None
    assert (summary.archive_dir / "model_params.parquet.corrupt.1").exists()
    assert (summary.archive_dir / "model_params.parquet.corrupt.2").exists()
