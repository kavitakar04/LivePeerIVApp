# TASKS.md

## Completed

### TASK-001: Fix SABR implied volatility explosions

**Category:** MODEL BUG  
**Priority:** P0  
**Status:** DONE

**Root Cause**
- SABR Hagan `z / x(z)` is sign-sensitive.
- Guarding `x(z)` with a positive-only denominator can flip the ratio sign and produce exploding or invalid IVs.

**Fix**
- Preserved signed denominator behavior in `volModel/sabrFit.py`.
- Added shared fitted-model validation in `volModel/quality.py` that rejects:
  - NaN/inf predictions
  - non-positive IV predictions
  - absurdly high IV predictions
  - non-finite or excessive RMSE
  - excessive skew/curvature where reported
- Fallback logs now explicitly mark degraded output.

**Files Changed**
- `volModel/sabrFit.py`
- `volModel/quality.py`
- `tests/test_vol_model_repairs.py`

**Validation**
- Direct SABR synthetic-smile regression passed.
- `py_compile` passed for model/quality/test files.
- Earlier environment issue caused some pytest invocations to exit with code `-1`; focused direct regression execution passed.

**Downstream Impact**
- Broken SABR output cannot be accepted by `auto` or plotted/logged as valid.
- Reduces fake relative-value divergences from model explosions.

---

### TASK-003: Fix pillar ATM SVI route

**Category:** MODEL BUG  
**Priority:** P0  
**Status:** DONE

**Root Cause**
- Pillar ATM SVI path previously called an incompatible SVI alias/signature and fell back to `poly2`, hiding the failure.

**Fix**
- `analysis/pillars.py` now routes SVI to `fit_svi_slice(S, K, T, iv)` with the same strike/spot inputs used by the model implementation.
- Failed SVI fits log structured fallback information instead of silently changing model.

**Files Changed**
- `analysis/pillars.py`
- `tests/test_vol_model_repairs.py`

**Validation**
- Regression confirms valid synthetic smile with `model="svi"` returns `model == "svi"` and low RMSE.

**Downstream Impact**
- Term/pillar ATM values no longer silently switch model because of dispatch/signature mismatch.
- Target/peer comparisons can identify the actual fitted model used.

---

### TASK-004: Add model quality gate for auto selection

**Category:** MODEL SELECTION  
**Priority:** P0  
**Status:** DONE

**Root Cause**
- `auto` could accept the first model that returned a dictionary, even if predictions were invalid or RMSE was absurd.

**Fix**
- `analysis/pillars.py` validates each candidate before accepting it.
- Invalid candidates log the reason and `auto` tries the next available model.
- Shared validator lives in `volModel/quality.py`.

**Files Changed**
- `analysis/pillars.py`
- `volModel/quality.py`
- `tests/test_vol_model_repairs.py`

**Validation**
- Regression monkeypatches broken SABR output and confirms `auto` does not return `sabr`.
- Direct focused regressions passed.

**Downstream Impact**
- `auto` selection now rejects invalid model fits before downstream plotting/logging.

---

### TASK-005: Audit weight modes: corr, pca, oi, cosine, equal

**Category:** WEIGHTS  
**Priority:** P0  
**Status:** DONE

**Observed**
- GUI exposes five modes:
  - `corr`
  - `pca`
  - `oi`
  - `cosine`
  - `equal`
- Some calculations are likely broken or unstable.

**Root Cause**
- Weight validation mostly existed, but diagnostics were not detailed enough for relative-value auditability.
- Synthetic test coverage did not fully exercise all GUI-exposed modes through the shared engine path.

**Fix**
- `analysis/unified_weights.py` now logs:
  - mode
  - feature set
  - peers
  - input shape
  - normalized output weights
  - normalization method and scale
  - min/max/mean/sum/L1/L2
  - non-finite and negative counts
  - condition number
  - clipped count
  - fallback and issues
- Failed/invalid weights visibly fall back to equal weights.
- Fallback diagnostics preserve the rejected input shape when available.

**Computation Functions**
- `equal`: `UnifiedWeightComputer.compute_weights()`
- `oi`: `UnifiedWeightComputer._open_interest_weights()`
- `corr`: `corr_weights_from_matrix()`
- `cosine`: `cosine_similarity_weights_from_matrix()`
- `pca`: `pca_regress_weights()`

**Files Changed**
- `analysis/unified_weights.py`
- `tests/test_weight_quality_gate.py`

**Validation**
- `py_compile` passed.
- Direct focused weight quality tests passed for `corr`, `pca`, `oi`, `cosine`, and `equal`.

**Downstream Impact**
- Peer composite weights are now traceable from GUI selection through normalized downstream use.
- Broken weights do not silently distort the consensus surface.

---

### TASK-006: Diagnose peer overlay / term structure mismatch

**Category:** VISUALIZATION / DATA INTEGRITY  
**Priority:** P0  
**Status:** DONE

**Observed**
- Screenshot shows IONQ ATM near 0.765.
- Peer overlays RGTI/QUBT/QBTS near 0.95–1.12.
- Legend shows equal 33.3% weights.
- Peers appear to have different maturity ranges.
- Weighted composite may not actually be plotted.

**Root Cause**
- Target ATM term structure used fit-based, auto-model, vega-weighted extraction.
- Peer term overlays used median, unweighted extraction through `atm_curve_for_ticker_on_date()`.
- Raw peer maturities and aligned weighted composite data were not clearly distinguished.

**Fix**
- Target and peers now use the same term ATM extraction helper:
  - `method="fit"`
  - `model="auto"`
  - `vega_weighted=True`
- ATM output rows now include audit metadata:
  - `spot`
  - `atm_strike`
  - `iv_source`
  - `extraction_status`
- Term prep logs for target and each peer:
  - ticker
  - expiry
  - T
  - spot/reference price
  - selected ATM strike
  - ATM IV
  - IV source column
  - number of valid options
  - raw/interpolated/fallback status
  - assigned weight
  - alignment status
- Weighted peer composite is built only on maturities aligned to the target grid.
- The returned plot payload includes:
  - `alignment_status`
  - `composite_status`
  - `term_warnings`
- Plot annotation now states:
  - individual peer lines are raw expiry points
  - peer composite is aligned to target maturities
  - warnings when comparisons are degraded
- If weights exist but composite overlay is hidden, the UI description says so.
- Warnings are emitted for no maturity overlap and extreme target/peer ATM level differences.

**Files Changed**
- `analysis/pillars.py`
- `analysis/analysis_pipeline.py`
- `display/plotting/term_plot.py`
- `display/gui/gui_plot_manager.py`
- `tests/test_term_plot.py`

**Validation**
- `py_compile` passed for touched files.
- `pytest tests/test_term_plot.py -vv`: 6 passed.
- `pytest tests/test_plot_manager_term_summary.py tests/test_term_weight_mode_replot.py -vv`: 2 passed.
- Adjacent `tests/test_ci_fix.py` passed during broader validation.

**Downstream Impact**
- Target and peer ATM term structures are now comparable.
- Raw peer overlays and aligned weighted composites are explicitly labeled.
- Fake divergences caused by ATM extraction mismatch are reduced.

---

## Active Queue

### TASK-007: Consolidate calculation cache APIs

**Category:** ANALYSIS CLEANUP  
**Priority:** P1
**Status:** PARTIAL - canonical backend selected and active callers migrated

**Observed**
- `analysis/compute_or_load.py` writes pickled artifacts to the main IV DB.
- `analysis/cache_io.py` writes TTL/versioned compressed artifacts to a separate calculations DB.
- The two cache APIs have different schemas and lifecycle rules.
- Update: `analysis.cache_io` is now the canonical TTL/versioned backend.
- Update: `analysis.compute_or_load` is now a compatibility wrapper.
- Update: active GUI/correlation callers import `compute_or_load` from `analysis.cache_io`.
- Update: cache key/version/TTL behavior is documented in `reports/cache_consolidation.md`.

**Expected**
- There should be one persistent calculation-cache API with clear ownership, TTL behavior, and storage location.

**Acceptance Criteria**
- [x] Pick the canonical cache backend.
- [x] Migrate active callers to the canonical API.
- [x] Document cache key/version/TTL behavior.
- [x] Add regression coverage for cache read/write/expiry behavior.
- [ ] Decide whether to migrate or retire legacy raw-marker helpers `save_calc_cache()` / `load_calc_cache()`.
- [ ] Audit scripts/reports that still describe the old split cache state.
- [ ] Add a cache cleanup/migration note for old `iv_data.db::calc_cache` entries if needed.

---

### TASK-008: Make unified weights the only weight engine

**Category:** WEIGHTS / ANALYSIS CLEANUP  
**Priority:** P1
**Status:** PARTIAL - active pipeline route migrated; legacy compatibility APIs remain

**Observed**
- `analysis/unified_weights.py` is the intended shared engine.
- Compatibility weight dispatch still exists in `analysis/beta_builder.py`.
- Update: `analysis.weight_service.compute_peer_weights()` is the focused unified-weight facade.
- Update: `analysis.analysis_pipeline.compute_peer_weights` now points at `analysis.weight_service`.
- Update: `relative_value_atm_report_corrweighted()` no longer calls `beta_builder.peer_weights_from_correlations()`.
- Update: `beta_builder.peer_weights_from_correlations()` and `build_peer_weights()` are marked deprecated compatibility paths.

**Expected**
- All active callers should route through `analysis.unified_weights`.
- Compatibility dispatch should be retired only after callers and tests are migrated.

**Acceptance Criteria**
- [x] Audit active weight call sites.
- [x] Replace active `beta_builder` compatibility dispatch where appropriate.
- [x] Run focused weight-route regression tests.
- [x] Clearly deprecate old compatibility APIs.
- [ ] Reconcile older weight tests that expect hard raises with current unified-weight equal fallback behavior.
- [ ] Migrate legacy direct tests/imports from `analysis.beta_builder` where they are testing active behavior rather than compatibility behavior.
- [ ] Remove compatibility APIs only after tests and external callers no longer depend on them.

---

### TASK-009: Split oversized analysis modules into focused services

**Category:** ANALYSIS CLEANUP  
**Priority:** P2
**Status:** STARTED - focused facades added, large bodies not yet moved

**Observed**
- `analysis/analysis_pipeline.py` contains ingestion, cache access, surface helpers, plot prep, RV workflows, and GUI-facing orchestration.
- `analysis/pillars.py` mixes pillar selection, ATM extraction, and model-fit helpers.
- Update: `analysis.weight_service` now owns the peer-weight facade.
- Update: `analysis.pillar_selection` exposes pillar selection helpers.
- Update: `analysis.atm_extraction` exposes ATM extraction helpers.
- Update: route tests preserve current public imports during migration.

**Expected**
- Analysis modules should have focused ownership boundaries while preserving existing public facades during migration.

**Acceptance Criteria**
- [ ] Split `analysis/analysis_pipeline.py` into focused services or reduce it to a facade.
- [ ] Split `analysis/pillars.py` into pillar selection, ATM extraction, and model-fit helpers.
- [x] Preserve current public imports until callers migrate.
- [x] Add targeted import/route tests.
- [ ] Move implementation bodies behind the new facades once call paths are stable.
- [ ] Add a dedicated model-fit helper module for `_fit_smile_get_atm` internals.
- [ ] Update imports gradually so new code uses focused modules directly.

---

### TASK-010: Finish peer-composite naming cleanup for persisted artifacts

**Category:** NAMING / PERSISTENCE  
**Priority:** P2

**Observed**
- User-facing labels and active modules now use "Peer Composite".
- Some exported artifact fields and folder names still use `synthetic_*`.
- Renaming these can break saved outputs, notebooks, and downstream readers.

**Expected**
- Persisted names should be migrated only after saved outputs and downstream consumers are audited.

**Acceptance Criteria**
- Audit all persisted artifact fields, export folders, and downstream readers.
- Define compatibility/migration rules for old saved outputs.
- Rename remaining `synthetic_*` fields where safe.
- Run GUI-facing plot manager and peer-composite tests before and after migration.

---

### TASK-011: Build a typed GUI settings panel

**Category:** SETTINGS / GUI  
**Priority:** P2

**Observed**
- Defaults now live in `analysis/settings.py`.
- Many previously hardcoded controls are routed through GUI settings.
- There is still no single typed settings object exposed through a dedicated settings tab/modal.

**Expected**
- Users should edit analysis defaults in one settings surface backed by a typed `AnalysisDefaults`-like object.

**Acceptance Criteria**
- Add settings UI sections for data/expiry, ATM/term, weighting, surface, RV, spillover, and cache.
- Persist settings to user preferences.
- Route plot-call overrides from the typed settings object.
- Clearly label or retire legacy near-term pillar defaults.

---

### TASK-012: Add NetworkX graph adapters and metrics

**Category:** NETWORK GRAPH ANALYTICS  
**Priority:** P2

**Observed**
- Spillover summaries and correlation matrices are naturally graph-shaped.
- Existing workflows return DataFrames but do not expose graph metrics.

**Expected**
- Add a graph layer without changing existing analytics outputs.

**Acceptance Criteria**
- Implement `build_spillover_digraph(summary_df, ...)`.
- Implement `build_corr_graph(corr_df, ...)`.
- Implement deterministic `graph_metrics_df(G)` output.
- Add tests for graph construction, edge filtering, metric stability, and signed edge handling.
- Keep NetworkX optional or provide a clear missing-dependency error.

---

### TASK-013: Remediate data quality risks

**Category:** DATA QUALITY  
**Priority:** P1

**Observed**
- Data audit found high null rates for bid/ask/open_interest.
- Several tickers have weak history or stale coverage.
- Some tickers have very low persisted ATM flag share.

**Expected**
- Analytics should warn or degrade gracefully when data quality is insufficient.

**Acceptance Criteria**
- Add data-quality gates for OI/liquidity-weighted paths.
- Warn on stale or one-date ticker histories for time-series/spillover/correlation workflows.
- Avoid relying solely on persisted `is_atm` when ATM flag share is too low.
- Add tests or fixtures for sparse OI, stale ticker coverage, and low ATM-flag coverage.

---

### TASK-014: Clean up duplicate TPS/poly implementations

**Category:** MODEL CLEANUP  
**Priority:** P2

**Observed**
- `volModel/polyFit.py` defines `fit_tps_slice()` and `tps_smile_iv()` twice.
- Later definitions override earlier ones, making behavior easy to misread.

**Expected**
- TPS/poly implementation should have one canonical definition per public function.

**Acceptance Criteria**
- Collapse duplicate `fit_tps_slice()` definitions into one.
- Collapse duplicate `tps_smile_iv()` definitions into one.
- Preserve intentional fallback behavior for params without an interpolator.
- Add/adjust tests for final import and fallback behavior.

---

### TASK-017: Audit repository file utility and remove dead assets

**Category:** REPOSITORY HYGIENE  
**Priority:** P1

**Observed**
- The repository contains legacy modules, reports, debug scripts, generated artifacts, caches, screenshots, and compatibility-era files.
- It is unclear which files are active product code, useful diagnostics, generated output, obsolete experiments, or safe to delete.

**Expected**
- Every file should have a clear purpose, owner/category, and retention decision.
- Files with no current utility should be removed only after dependency and validation checks prove they are unused.

**Acceptance Criteria**
- Inventory all tracked and untracked files by category:
  - runtime source
  - tests
  - GUI assets
  - scripts/tools
  - reports/docs
  - generated artifacts
  - local caches/data
  - compatibility/legacy files
- For each questionable file, determine:
  - direct imports/references
  - CLI or GUI entrypoints
  - whether it is generated or hand-authored
  - whether it is needed for tests, docs, or user workflows
- Produce an audit table with keep/delete/move decisions and rationale.
- Delete only files classified as unused and non-generated-critical.
- Move durable documentation into the right docs/report location instead of deleting it.
- Update `.gitignore` for generated artifacts that should not reappear.
- Run import checks and focused tests after deletion.
- Do not remove user data, databases, or generated reports unless explicitly confirmed safe.

---

### TASK-019: Fix RV Heatmap surface-grid dimension mismatch

**Category:** GUI / RELATIVE VALUE / SURFACE ALIGNMENT  
**Priority:** P0

**Observed**
- Screenshot: `RV Heatmap` for IONQ vs peers with `Weights=pca` and `Features=surface_grid` fails in-plot with raw NumPy error:
  - `all the input array dimensions except for the concatenation axis must match exactly`
  - one surface vector has size `15`, another has size `18`
- The top status text describes an "IV similarity matrix" and right-panel portfolio weights, which does not match the selected `RV Heatmap` plot.
- Model selector is disabled but still displays `sabr`, which is confusing for a heatmap that does not use a smile model.
- The heatmap axes show an empty default 0-1 plot instead of an actionable data-quality/alignment message.

**Expected**
- RV Heatmap should align target and peer surface grids to common moneyness/tenor cells before concatenating or differencing.
- Missing grid cells should be handled deliberately through intersection, reindexing, or explicit "insufficient common grid" fallback.
- The GUI should show RV Heatmap-specific status text and diagnostics, not stale correlation/similarity copy.
- User-facing errors should explain which tickers/dates/grid dimensions failed and what fallback was used.

**Acceptance Criteria**
- Trace `RV Heatmap` path from GUI selection through `prepare_rv_heatmap_data` and surface-grid construction.
- Identify where unequal flattened surface vectors are concatenated.
- Add a shared surface-grid alignment helper that reports:
  - target ticker/date/grid shape
  - each peer ticker/date/grid shape
  - common row/column counts
  - dropped/missing cells
- Use common grid intersection or safe reindexing before vector concatenation/PCA/heatmap residual computation.
- If fewer than the minimum common cells remain, render a clear "insufficient common surface grid" message with ticker details.
- Fix plot description text for `RV Heatmap` so it explains residuals/spreads, not similarity-matrix weights.
- Hide or blank model selection for plots that do not consume model choice, rather than showing a disabled stale model value.
- Add regression test with peer surfaces of shapes 15 and 18 proving the heatmap path aligns or fails gracefully without raw NumPy exceptions.
- Add integration/plot-manager test for `RV Heatmap` with `surface_grid` features and PCA weights.

---

### TASK-020: Reduce repository mess through explicit ownership boundaries

**Category:** REPOSITORY HYGIENE / ARCHITECTURE  
**Priority:** P1  
**Status:** STARTED - first active import boundary migrated
**Update:** shared model-fit/quality boundary added and active GUI/pipeline callers migrated

**Observed**
- GUI code still imports some low-level analysis helpers directly.
- `analysis/analysis_pipeline.py` remains a broad orchestrator with mixed responsibilities.
- `analysis/pillars.py` still owns ATM extraction, pillar logic, and model-fit internals.
- Compatibility facades exist, but active imports have not all moved to those facades.
- Debug files, reports, generated images, local DB artifacts, and compatibility modules make it hard to tell what is product code versus temporary analysis.

**Product Direction**
- Keep product-facing flows organized around relative-value questions:
  - clean GUI dispatch
  - explicit analysis services
  - validated model/weight contracts
  - plotting that only visualizes already-validated payloads
- Avoid broad rewrites. Move one boundary at a time, keeping compatibility aliases until call sites and tests prove migration is safe.

**Ownership Targets**
- `display/gui`: UI state, event handling, user-visible warnings.
- `display/plotting`: render already-prepared payloads; no model selection or data fetching.
- `analysis/*_service.py`: product workflow facades for weights, ATM extraction, peer composite, RV analysis, and fit summaries.
- `analysis/analysis_pipeline.py`: compatibility facade and orchestration only; shrink over time.
- `volModel`: model fitting, prediction, and numerical quality validation only.
- `data`: persistence, ingestion, schema, and raw quote quality.

**Acceptance Criteria**
- [x] Add a public ATM extraction facade name: `analysis.atm_extraction.fit_smile_get_atm`.
- [x] Keep `_fit_smile_get_atm` as a compatibility alias while callers migrate.
- [x] Move active GUI/pipeline imports from `analysis.pillars._fit_smile_get_atm` to `analysis.atm_extraction.fit_smile_get_atm`.
- [x] Create a small shared model-fit service for params, prediction, quality metadata, fallback logging, and diagnostics.
- [x] Move duplicate model-quality serialization helpers out of GUI/pipeline code into `analysis.model_fit_service`.
- [x] Split `analysis/analysis_pipeline.py` into thinner compatibility facades for:
  - smile data
  - term data
  - RV heatmap data
  - ingestion/data availability
- [x] Move ATM extraction implementation bodies out of `analysis.pillars.py` once callers are stable.
- [x] Add a stricter fit-result dataclass contract once call sites no longer depend on plain dictionaries.
- [ ] Move GUI plot-manager fitting/logging side effects into analysis services.
- [ ] Audit root-level generated images and local data artifacts for `.gitignore` or deletion decisions.
- [ ] Produce a keep/delete/move table for reports, debug scripts, generated files, and legacy compatibility modules.

**Execution Order**
- Phase 1: migrate active imports to existing facades.
  Update: Active GUI imports for smile data, term data, RV heatmap data, spillover data availability, model-fit helpers, weight service, ATM extraction, and pillar selection now route through focused service/facade modules.
- Phase 2: extract shared contracts/helpers without changing behavior.  
  Update: `analysis.model_fit_service` now centralizes SVI/SABR/TPS fit, prediction, quality serialization, and checked-fit helpers used by `analysis.analysis_pipeline` and `display.gui.gui_plot_manager`.
- Phase 3: move implementation bodies behind facades.
  Update: ATM extraction implementation bodies now live in `analysis.atm_extraction`; `analysis.pillars` re-exports compatibility names.
- Phase 4: delete compatibility aliases only after import scans and tests show no active dependency.
- Phase 5: remove or ignore generated artifacts after explicit classification.

**Follow-On Phases**
- **Phase 1A: Finish Facade Import Migration** — DONE for active GUI imports.
  - Move active imports to existing facades: `atm_extraction`, `pillar_selection`, `weight_service`, and peer-composite services.
  - Keep compatibility aliases in place.
  - Add route/import tests for each migrated facade.
  - Update: Added `analysis.smile_data_service`, `analysis.term_data_service`, `analysis.rv_heatmap_service`, and `analysis.data_availability_service` as thin compatibility facades over the current pipeline implementations.
  - Update: `display.gui.gui_plot_manager` now imports smile, term, and RV heatmap preparation through the focused facades.
  - Update: `display.gui.spillover_gui` now imports spillover IV/HV availability through `analysis.data_availability_service`.
  - Validation: `py_compile` passed for touched facade/GUI/test modules; focused route, term, weight-replot, and spillover tests passed.
- **Phase 1B: Shared Fit/Quality Contract**
  - Add one shared fit-result representation for params, quality, fallback, RMSE, N, and reason.
  - Remove duplicate quality-to-dict helpers from GUI and pipeline code.
  - Keep current dict payload compatibility until downstream callers migrate.
  - Update: `analysis.model_fit_service.ModelFitResult` now carries requested/actual model, params, quality, ok/degraded state, reason, RMSE, and N.
  - Update: Existing tuple/dict helpers remain compatibility wrappers over the typed contract.
- **Phase 2A: Extract ATM And Pillar Implementations** — DONE for ATM extraction.
  - Move ATM extraction implementation bodies out of `analysis/pillars.py`.
  - Leave `analysis/pillars.py` as a compatibility facade.
  - Migrate active callers to public facade names.
  - Update: `analysis.atm_extraction` now owns `fit_smile_get_atm`, `_fit_smile_get_atm`, `compute_atm_by_expiry`, and `atm_curve_for_ticker_on_date`.
  - Update: `analysis.pillars` imports and re-exports those names for legacy callers.
  - Validation: import scan confirms ATM fit/compute function definitions live in `analysis.atm_extraction`; focused route/model/CI/term tests passed.
- **Phase 2B: Split Pipeline By Product Workflow**
  - Move smile, term, RV heatmap, and market-data availability logic into focused services.
  - Keep `analysis_pipeline.py` as a re-export facade during migration.
- **Phase 3A: Move Computation Out Of GUI Plot Manager**
  - Move fitting, logging, fallback, and weight-resolution decisions into analysis services.
  - Leave `gui_plot_manager.py` responsible for dispatch, UI descriptions, and renderer calls.
  - Update: Parameter Summary is now a separate user-facing tab with one row per expiry showing ATM vol, skew, curvature, and a simple fit-quality label.
  - Update: Model/data diagnostics, fallback details, RMSE, completeness, and weight/status events now live in the Settings / Status diagnostics table.
- **Phase 3B: Make Plotting Render-Only**
  - Ensure plotting modules receive validated payloads.
  - Plot explicit degraded/error payloads instead of doing validation or fallback internally.
- **Phase 4A: Repository File Utility Audit**
  - Classify reports, debug scripts, generated images, local DB/cache artifacts, and compatibility files.
  - Produce keep/delete/move decisions before deleting anything.
  - Update `.gitignore` for generated artifacts that should not reappear.
- **Phase 4B: Compatibility Removal**
  - Remove aliases/shims only after import scans and tests prove no active dependency.
  - Keep external/notebook compatibility notes where removal could break user workflows.

**Follow-On Test Plan**
- Add or update route tests whenever an import moves.
- Add contract tests before replacing loose dict payloads.
- Run focused tests for each migrated workflow.
- Run `py_compile` for touched modules.
- Use `rg` import scans before removing compatibility aliases.

**Follow-On Assumptions**
- Task 20 remains the parent cleanup task.
- Follow-ons should be phased checklist items, not separate task numbers.
- The implementation should stay incremental: one boundary per patch, no broad rewrite.

---

### TASK-021: Normalize ingestion as-of dates outside market hours

**Category:** DATA INGESTION / DATE SEMANTICS  
**Priority:** P1  
**Status:** TODO

**Observed**
- Option-chain ingestion currently stamps downloaded rows with `datetime.now(timezone.utc).date()`.
- The GUI date dropdown reads distinct `options_quotes.asof_date` values and auto-selects the latest stored date.
- Outside regular market hours, especially after the US close or before the next open, vendor option-chain data may still represent the last completed trading session even though UTC/local calendar date has advanced.
- This can make the GUI show a fresh-looking date that is not a true market data session, or split related option/underlying data across inconsistent dates.

**Expected**
- `asof_date` should represent the market data session date, not the wall-clock ingestion date.
- Download/ingest should be deterministic before market open, during market hours, after market close, weekends, and market holidays.
- GUI date selection should surface only valid data-session dates, with clear status text when data is stale or vendor data belongs to the prior session.

**Design Direction**
- Add a single market-session date resolver used by ingestion and GUI availability paths.
- Resolve dates in `America/New_York`, not UTC calendar date.
- Treat regular US equity session hours as the default policy unless a market-calendar dependency is added:
  - before market open: use previous trading session
  - during market hours: use current trading session, but mark intraday/partial if needed
  - after market close: use current trading session
  - weekends/holidays: use previous trading session
- Prefer vendor-provided quote/history timestamps when available; fall back to session resolver only when vendor data lacks a usable timestamp.
- Store enough metadata to distinguish:
  - `asof_date` / market session date
  - `ingested_at` / wall-clock timestamp
  - `vendor_timestamp` when available
  - stale/partial-session status

**Acceptance Criteria**
- Audit current ingestion path from `Download / Ingest` → `ingest_and_process` → raw download → enriched quotes → DB insert → GUI date dropdown.
- Add a shared resolver for market-session date and trading-session status.
- Stop stamping option rows solely with UTC `now().date()`.
- Ensure option quotes, underlying prices, and GUI available dates use consistent session-date semantics.
- On ingest completion, refresh the GUI to the resolved session date and show whether the data is current, prior-session, partial, or stale.
- Add safeguards against creating duplicate misleading dates when downloading outside market hours.
- Add structured logs for:
  - wall-clock ingest time
  - resolved session date
  - vendor timestamp if present
  - session status
  - tickers affected
- Add tests for:
  - before US market open
  - during market hours
  - after market close
  - weekend
  - holiday or supplied market-closed date
  - vendor timestamp overriding wall-clock fallback

**Implementation Notes**
- Candidate files: `data/data_downloader.py`, `data/historical_saver.py`, `data/db_schema.py`, `data/db_utils.py`, `display/gui/browser.py`, and `analysis/data_availability_service.py`.
- Consider an additive DB migration for ingestion metadata rather than changing the existing primary key until compatibility impact is understood.
- Do not delete or rewrite existing dates automatically; first add a repair/audit script that reports suspicious dates and proposed remapping.
- Keep external API compatibility for callers that pass explicit `asof_date`.

## Rules for All Tasks

Every patch must include:
- Minimal root-cause fix.
- Regression test.
- No broad refactor unless necessary.
- No silent fallback.
- Structured logging for failure modes.
- Validation before plotting or logging model results.
