# Data Package Layout

The Python files in this directory are source modules for ingestion,
normalization, persistence, and reference-data handling.

Runtime artifacts should not be added to source control. This includes:

- `*.db`, `*.db-shm`, `*.db-wal`, and `*.db.bak-*`
- generated parquet outputs such as `model_params.parquet` and spillover files
- GUI/runtime settings such as `gui_settings.json`
- cache folders such as `cache/` and `cache_synth_etf/`

Current database paths are still rooted here for compatibility with existing
GUI and analysis code. New runtime outputs should prefer ignored subdirectories
such as `data/runtime/`, `data/artifacts/`, or `data/cache/`.

New code should keep responsibilities separated:

- `db_schema.py` and `db_utils.py`: SQLite schema and low-level persistence
- `data_downloader.py` and `historical_saver.py`: external option-chain ingestion
- `data_pipeline.py`, `quote_quality.py`, and `greeks.py`: normalization,
  validation, and enrichment
- `interest_rates.py`: global and ticker-specific rate reference data
- `ticker_groups.py`: saved peer-group presets
- `underlying_prices.py`: underlying close history
