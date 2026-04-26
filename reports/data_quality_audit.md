# Data Quality Audit

Generated: 2026-04-25T01:11:26.477607+00:00
Project root: `/Users/kavitakar/LivePeerIVApp`

## Storage Map

- Raw options and underlying prices: `data/iv_data.db`
- Calculation cache table: `data/calculations.db::calc_cache`
- Model parameter history: `data/model_params.parquet`
- External rate CSVs: `data/ML_Rates/*.csv`

## Access Map

| Data | Stored In | Main Accessors | Notes |
|---|---|---|---|
| Raw option chains | `data/iv_data.db::options_quotes` | `data.db_utils.get_conn`, `data.db_utils.fetch_quotes`, `analysis.analysis_pipeline.get_smile_slice` | Primary source for smiles, terms, surfaces, weights. `DB_PATH` can override the file. |
| Underlying closes | `data/iv_data.db::underlying_prices` | `analysis.unified_weights.underlying_returns_matrix` | Used by `ul` feature modes and fallback weights. |
| Ticker presets | `data/iv_data.db::ticker_groups` | `data.ticker_groups`, `display.gui.gui_input.InputPanel` | GUI universe presets. |
| Interest rates | `data/iv_data.db::interest_rates`, `ticker_interest_rates` | `data.interest_rates`, GUI rate controls | Used during ingestion/Greek enrichment. |
| Calculation cache | `data/calculations.db::calc_cache` | `analysis.cache_io.compute_or_load`, `analysis.cache_io.WarmupWorker` | Canonical TTL/versioned cache for computed smiles, terms, correlations, surfaces, and warmup artifacts. |
| Model params | `data/model_params.parquet` | `analysis.model_params_logger` | Fit parameter history shown in parameter views. |

## Issues

| Severity | Area | Finding |
|---|---|---|
| medium | market_fields | bid_null: 66,520 rows (68.4%); OI/liquidity-weighted paths may be unreliable |
| medium | market_fields | ask_null: 66,335 rows (68.2%); OI/liquidity-weighted paths may be unreliable |
| medium | market_fields | open_interest_null: 66,727 rows (68.6%); OI/liquidity-weighted paths may be unreliable |
| high | coverage | 12 tickers have only one as-of date; time-series/spillover/correlation history is weak |
| medium | coverage | 26 tickers are stale versus latest as-of 2026-04-25 |
| medium | atm_flags | 13 tickers have <2% rows flagged ATM; do not rely on persisted is_atm alone |

## Raw Options Summary

- Rows: 97,299
- Tickers: 46
- As-of dates: 9
- Date range: `2025-08-12` to `2026-04-25`
- SQLite quick_check: `ok`

### Sanity Counts

| Check | Rows |
|---|---:|
| bad_iv | 0 |
| bad_spot | 0 |
| bad_strike | 0 |
| bad_ttm | 0 |
| bad_moneyness | 0 |
| bad_bid | 0 |
| bad_ask | 0 |
| crossed_market | 0 |
| mid_outside_market | 0 |
| bad_volume | 0 |
| bad_open_interest | 0 |

### Null Counts

| Column | Null rows |
|---|---:|
| iv | 0 |
| spot | 0 |
| ttm | 0 |
| moneyness | 0 |
| bid | 66,520 |
| ask | 66,335 |
| open_interest | 66,727 |
| delta | 0 |

### Ticker Coverage

| Ticker | Rows | Dates | Date Range | Expiries | IV Rows | ATM Share |
|---|---:|---:|---|---:|---:|---:|
| SPY | 9,582 | 6 | 2025-08-12 to 2026-04-25 | 28 | 9,582 | 0.0101 |
| QQQ | 8,197 | 6 | 2025-08-12 to 2026-04-25 | 28 | 8,197 | 0.0116 |
| UNH | 6,170 | 6 | 2025-08-15 to 2026-04-25 | 25 | 6,170 | 0.0196 |
| TQQQ | 5,104 | 6 | 2025-08-12 to 2026-04-25 | 17 | 5,104 | 0.0182 |
| GS | 4,541 | 4 | 2025-08-12 to 2025-08-18 | 15 | 4,541 | 0.0172 |
| IWM | 4,530 | 6 | 2025-08-12 to 2026-04-25 | 26 | 4,530 | 0.0196 |
| XLK | 3,443 | 6 | 2025-08-12 to 2026-04-25 | 20 | 3,443 | 0.0299 |
| NVDA | 3,310 | 3 | 2025-08-13 to 2025-08-15 | 12 | 3,310 | 0.0124 |
| ELV | 3,281 | 6 | 2025-08-15 to 2026-04-25 | 14 | 3,281 | 0.0277 |
| JPM | 2,772 | 4 | 2025-08-12 to 2025-08-18 | 15 | 2,772 | 0.0274 |
| HUM | 2,759 | 5 | 2025-08-15 to 2026-04-25 | 24 | 2,759 | 0.0312 |
| MSTR | 2,601 | 1 | 2025-08-13 to 2025-08-13 | 12 | 2,601 | 0.0088 |
| CVS | 2,522 | 6 | 2025-08-15 to 2026-04-25 | 24 | 2,522 | 0.0476 |
| CI | 2,494 | 6 | 2025-08-15 to 2026-04-25 | 22 | 2,494 | 0.0469 |
| AMD | 2,429 | 3 | 2025-08-13 to 2025-08-15 | 12 | 2,429 | 0.0185 |
| AVGO | 2,255 | 3 | 2025-08-13 to 2025-08-15 | 12 | 2,255 | 0.0195 |
| XLF | 2,143 | 4 | 2025-08-12 to 2025-08-18 | 15 | 2,143 | 0.0359 |
| MS | 2,069 | 4 | 2025-08-12 to 2025-08-18 | 15 | 2,069 | 0.0367 |
| BAC | 1,961 | 4 | 2025-08-12 to 2025-08-18 | 15 | 1,961 | 0.0382 |
| WFC | 1,940 | 4 | 2025-08-12 to 2025-08-18 | 15 | 1,940 | 0.0402 |
| SMH | 1,904 | 3 | 2025-08-13 to 2025-08-15 | 12 | 1,904 | 0.0242 |
| TSM | 1,695 | 3 | 2025-08-13 to 2025-08-15 | 12 | 1,695 | 0.0271 |
| RGTI | 1,411 | 2 | 2025-08-12 to 2026-04-25 | 25 | 1,411 | 0.0354 |
| OSCR | 1,403 | 3 | 2025-08-15 to 2026-04-25 | 22 | 1,403 | 0.0428 |
| IONQ | 1,396 | 2 | 2025-08-12 to 2026-04-25 | 24 | 1,396 | 0.0344 |
| META | 1,349 | 1 | 2025-08-13 to 2025-08-13 | 6 | 1,349 | 0.0089 |
| INTC | 1,300 | 3 | 2025-08-13 to 2025-08-15 | 12 | 1,300 | 0.0354 |
| SMCI | 1,293 | 1 | 2025-08-13 to 2025-08-13 | 12 | 1,293 | 0.017 |
| BMNR | 1,291 | 2 | 2025-08-12 to 2025-08-13 | 12 | 1,291 | 0.0364 |
| QBTS | 1,267 | 2 | 2025-08-12 to 2026-04-25 | 21 | 1,267 | 0.0339 |
| CNC | 1,251 | 3 | 2025-08-15 to 2026-04-25 | 22 | 1,251 | 0.0496 |
| SBET | 1,116 | 2 | 2025-08-12 to 2025-08-13 | 10 | 1,116 | 0.0358 |
| MOH | 1,108 | 3 | 2025-08-15 to 2026-04-25 | 12 | 1,108 | 0.0343 |
| QUBT | 829 | 2 | 2025-08-12 to 2026-04-25 | 19 | 829 | 0.0483 |
| MSFT | 760 | 1 | 2025-08-13 to 2025-08-13 | 6 | 760 | 0.0145 |
| MARA | 625 | 1 | 2025-08-13 to 2025-08-13 | 12 | 625 | 0.0384 |
| AAPL | 510 | 1 | 2025-08-13 to 2025-08-13 | 6 | 510 | 0.0235 |
| BITO | 491 | 1 | 2025-08-13 to 2025-08-13 | 12 | 491 | 0.0468 |
| AMZN | 485 | 1 | 2025-08-13 to 2025-08-13 | 6 | 485 | 0.0268 |
| GOOGL | 430 | 1 | 2025-08-13 to 2025-08-13 | 6 | 430 | 0.0279 |
| CLOV | 338 | 3 | 2025-08-15 to 2026-04-25 | 19 | 338 | 0.1243 |
| WULF | 261 | 1 | 2025-08-13 to 2025-08-13 | 12 | 261 | 0.0843 |
| ALHC | 241 | 3 | 2025-08-15 to 2026-04-25 | 12 | 241 | 0.1577 |
| OPEN | 216 | 1 | 2025-08-13 to 2025-08-13 | 12 | 216 | 0.1065 |
| PGNY | 128 | 3 | 2025-08-15 to 2026-04-25 | 8 | 128 | 0.1797 |
| ARQQ | 98 | 1 | 2025-08-12 to 2025-08-12 | 4 | 98 | 0.0816 |

## Cache Summary

- `data/calculations.db` exists: `True`, quick_check: `ok`
- `iv_data.db::calc_cache` rows: 0
- `calculations.db::calc_cache` rows: 2

## Data Files

| Path | Type | Size | Rows | Null cells |
|---|---|---:|---:|---:|
| `data/ML_Rates/ML_aug06.csv` | .csv | 4,912,307 | 119,580 | 688,691 |
| `data/ML_Rates/ML_aug08.csv` | .csv | 4,921,122 | 119,808 | 690,017 |
| `data/calculations.db` | .db | 69,632 |  |  |
| `data/iv_data.db` | .db | 36,933,632 |  |  |
| `data/model_params.parquet` | .parquet | 17,304 | 1,350 | 0 |
