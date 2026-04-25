# IVCorrelation agent guide

## Big picture
- `analysis/analysis_pipeline.py` is the GUI-facing orchestration layer. Prefer extending it for user-visible analytics instead of having GUI code call lower-level modules directly.
- Main flow: downloader/raw quotes → `data/data_pipeline.enrich_quotes()` computes `T`, `moneyness`, Greeks, and `is_atm` → `data/db_utils.insert_quotes()` persists to SQLite `options_quotes` (`sigma`/`S`/`T` become DB columns `iv`/`spot`/`ttm_years`) → analysis modules read pandas objects back from the DB.
- Surface contract lives in `analysis/syntheticETFBuilder.py`: `build_surface_grids()` returns `dict[ticker][pd.Timestamp] -> DataFrame` with moneyness rows and tenor columns; `combine_surfaces()` expects the same shape.
- Synthetic ETF orchestration in `analysis/analysis_synthetic_etf.py` intentionally separates weight computation from surface assembly; preserve that split when adding new weighting methods.
- Weight logic is centralized in `analysis/unified_weights.py`. Use canonical mode strings like `corr_iv_atm`, `corr_surface`, `corr_ul`, `cosine_iv_atm`, `pca_surface`, `equal`, and `oi`; short CLI values like `corr` default to ATM features.
- ATM term-structure extraction lives in `analysis/pillars.py`; default working pillars are short-dated `[7, 14, 30]`, with broader detection helpers for sparse datasets.
- GUI responsiveness depends on background threads in `display/gui/browser.py`; do DB/network work off the Tkinter thread and marshal UI updates back with `after()`.

## Developer workflows
- Setup: `python -m venv venv && source venv/bin/activate && pip install -r requirements.txt`
- Full regression suite: `python -m pytest tests/ -v`
- Fast focused checks: `python -m pytest tests/test_surfaces.py -v` and `python -m pytest tests/test_pipeline_routes.py -v`
- CLI smoke test without display requirements: `python scripts/scripts_synthetic_etf_demo.py --target SPY --peers QQQ IWM --no-show`
- GUI entry point: `python display/gui/browser.py` (requires tkinter and a working matplotlib GUI backend)

## Project-specific conventions
- Many entry points prepend the repo root to `sys.path`; preserve that pattern in new top-level scripts or GUI modules.
- The database path comes from `data/db_utils.DB_PATH` and can be overridden with the `DB_PATH` environment variable.
- `*_ul` weight modes depend on `underlying_prices`; `data.data_pipeline.ensure_underlying_price_data()` may auto-fetch missing history, so do not assume `options_quotes` alone is sufficient.
- Caching is layered: `analysis/analysis_pipeline.py` uses `lru_cache`, plot paths use SQLite-backed artifact caches in `analysis/compute_or_load.py` / `analysis/cache_io.py`, and `PlotManager` keeps an in-memory surface cache keyed by ticker set plus `max_expiries`.
- Tests usually isolate data with in-memory SQLite and monkeypatch the module-local `get_conn()` they exercise (for example `analysis.syntheticETFBuilder.get_conn`) instead of using the real DB.
- Prefer pandas DataFrames/Series as public results. Most high-level APIs return plain pandas structures or nested dicts; `SyntheticETFArtifacts` is the main structured exception.
- When changing weight-mode UI, update both the GUI split fields (`weight_method`, `feature_mode`) and the recombined canonical string consumed by `PlotManager` and `compute_peer_weights()`.

## Useful files
- `analysis/analysis_pipeline.py`, `analysis/unified_weights.py`, `analysis/syntheticETFBuilder.py`, `analysis/analysis_synthetic_etf.py`
- `data/data_pipeline.py`, `data/db_utils.py`
- `display/gui/browser.py`, `display/gui/gui_plot_manager.py`
- `tests/test_surfaces.py`, `tests/test_unified_weight_methods.py`, `tests/test_no_errors.py`