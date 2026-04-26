import pandas as pd


def test_load_model_params_returns_empty_for_corrupt_parquet(tmp_path, monkeypatch):
    from analysis import model_params_logger

    store = tmp_path / "model_params.parquet"
    store.write_text("not parquet", encoding="utf-8")
    monkeypatch.setattr(model_params_logger, "STORE_PATH", store)

    df = model_params_logger.load_model_params()

    assert df.empty
    assert list(df.columns) == model_params_logger.PARAM_COLUMNS


def test_append_params_quarantines_corrupt_store_before_write(tmp_path, monkeypatch):
    from analysis import model_params_logger

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
