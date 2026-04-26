def test_pipeline_weight_facade_points_to_weight_service():
    import analysis.analysis_pipeline as pipeline
    import analysis.weight_service as weight_service

    assert pipeline.compute_peer_weights is weight_service.compute_peer_weights


def test_focused_pillar_facades_preserve_public_helpers():
    import analysis.atm_extraction as atm_extraction
    import analysis.pillar_selection as pillar_selection
    import analysis.pillars as pillars

    assert atm_extraction.fit_smile_get_atm is pillars._fit_smile_get_atm
    assert atm_extraction.compute_atm_by_expiry is pillars.compute_atm_by_expiry
    assert atm_extraction.atm_curve_for_ticker_on_date is pillars.atm_curve_for_ticker_on_date
    assert pillar_selection.nearest_pillars is pillars.nearest_pillars


def test_model_fit_service_is_gui_and_pipeline_boundary():
    import analysis.analysis_pipeline as pipeline
    import analysis.model_fit_service as model_fit_service
    import display.gui.gui_plot_manager as gui_plot_manager

    assert pipeline.quality_checked_result is model_fit_service.quality_checked_result
    assert pipeline.fit_model_params is model_fit_service.fit_model_params
    assert gui_plot_manager.fit_valid_model_result is model_fit_service.fit_valid_model_result
    assert gui_plot_manager.fit_valid_model_params is model_fit_service.fit_valid_model_params


def test_model_fit_contract_preserves_legacy_tuple(monkeypatch):
    import numpy as np

    import analysis.model_fit_service as model_fit_service

    class Quality:
        ok = True
        reason = ""
        rmse = 0.012
        min_iv = 0.2
        max_iv = 0.3
        n = 3

    monkeypatch.setattr(model_fit_service, "validate_model_fit", lambda *args, **kwargs: Quality())

    params = {"theta": 1.0}
    result = model_fit_service.quality_checked_contract(
        "SVI",
        params,
        100.0,
        np.array([90.0, 100.0, 110.0]),
        0.25,
        np.array([0.22, 0.2, 0.24]),
    )

    assert isinstance(result, model_fit_service.ModelFitResult)
    assert result.requested_model == "svi"
    assert result.actual_model == "svi"
    assert result.params is params
    assert result.ok is True
    assert result.degraded is False
    assert result.rmse == 0.012
    assert result.n == 3
    assert result.as_legacy_tuple() == (params, result.quality)
    assert model_fit_service.quality_checked_result(
        "svi",
        params,
        100.0,
        np.array([90.0, 100.0, 110.0]),
        0.25,
        np.array([0.22, 0.2, 0.24]),
    ) == (params, result.quality)


def test_data_service_facades_preserve_pipeline_routes():
    import analysis.analysis_pipeline as pipeline
    import analysis.data_availability_service as data_availability_service
    import analysis.rv_heatmap_service as rv_heatmap_service
    import analysis.smile_data_service as smile_data_service
    import analysis.term_data_service as term_data_service
    import display.gui.browser as browser
    import display.gui.gui_plot_manager as gui_plot_manager
    import display.gui.spillover_gui as spillover_gui

    assert smile_data_service.get_smile_slice is pipeline.get_smile_slice
    assert smile_data_service.get_smile_slices_batch is pipeline.get_smile_slices_batch
    assert smile_data_service.prepare_smile_data is pipeline.prepare_smile_data
    assert term_data_service.prepare_term_data is pipeline.prepare_term_data
    assert rv_heatmap_service.prepare_rv_heatmap_data is pipeline.prepare_rv_heatmap_data
    assert data_availability_service.available_dates is pipeline.available_dates
    assert data_availability_service.available_tickers is pipeline.available_tickers
    assert data_availability_service.ingest_and_process is pipeline.ingest_and_process
    assert data_availability_service.get_daily_iv_for_spillover is pipeline.get_daily_iv_for_spillover
    assert data_availability_service.get_daily_hv_for_spillover is pipeline.get_daily_hv_for_spillover

    assert browser.available_dates is data_availability_service.available_dates
    assert browser.available_tickers is data_availability_service.available_tickers
    assert browser.ingest_and_process is data_availability_service.ingest_and_process
    assert gui_plot_manager.available_dates is data_availability_service.available_dates
    assert gui_plot_manager.get_smile_slice is smile_data_service.get_smile_slice
    assert gui_plot_manager.prepare_smile_data is smile_data_service.prepare_smile_data
    assert gui_plot_manager.prepare_term_data is term_data_service.prepare_term_data
    assert gui_plot_manager.prepare_rv_heatmap_data is rv_heatmap_service.prepare_rv_heatmap_data
    assert spillover_gui.get_daily_iv_for_spillover is data_availability_service.get_daily_iv_for_spillover
    assert spillover_gui.get_daily_hv_for_spillover is data_availability_service.get_daily_hv_for_spillover


def test_ingest_facade_accepts_underlying_lookback(monkeypatch):
    import analysis.analysis_pipeline as pipeline

    seen = {}

    def fake_save_for_tickers(tickers, max_expiries=10, r=0.0, q=0.0, underlying_lookback_days=500):
        seen["tickers"] = tickers
        seen["underlying_lookback_days"] = underlying_lookback_days
        return 7

    monkeypatch.setattr(pipeline, "save_for_tickers", fake_save_for_tickers)

    inserted = pipeline.ingest_and_process(["aaa"], underlying_lookback_days=800)

    assert inserted == 7
    assert seen == {"tickers": ["AAA"], "underlying_lookback_days": 800}
