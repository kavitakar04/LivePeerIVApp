# Analysis Folder Legacy Audit

Date: 2026-04-25

## Summary

The `analysis/` package is not mostly dead code. It is mostly transitional code:
large legacy implementation modules are still active, while newer focused service
facades have been added around them. The cleanup should therefore be a migration,
not a broad delete.

Immediate safe cleanup:
- Remove generated `analysis/__pycache__/` artifacts from the working tree if they
  are present locally.
- Keep `analysis/compute_or_load.py` deleted; active callers have moved to
  `analysis/cache_io.py`.
- Treat `analysis_background_tasks.py` as a likely maintenance-only path and audit
  it with `data/db_maintainance.py` before keeping it in the main package.

Highest-value refactor targets:
- `analysis/analysis_pipeline.py`
- `analysis/unified_weights.py`
- `analysis/rv_analysis.py`
- `analysis/beta_builder.py`
- `analysis/pillars.py`
- `analysis/peer_composite_builder.py`

## Import Evidence

Import scans were run against Python and Markdown files with `rg`, excluding
`__pycache__`.

## File Classification

| File | Status | Evidence | Recommendation |
| --- | --- | --- | --- |
| `analysis/__init__.py` | Keep | Package marker only. | Keep minimal. |
| `analysis/analysis_pipeline.py` | Active legacy core | Imported by GUI facades, tests, scripts, `rv_analysis`, `unified_weights`, and demos. | Do not delete. Continue splitting into focused services, then downgrade to re-export facade. |
| `analysis/analysis_background_tasks.py` | Likely legacy/maintenance | Only direct caller found is `data/db_maintainance.py`. Not part of GUI path. | Candidate to move under `data/` or `scripts/maintenance/` after validating background spillover workflow. |
| `analysis/atm_extraction.py` | Active service | Used by GUI plot manager, `analysis_pipeline`, and tests. | Keep. This is a good target boundary. |
| `analysis/beta_builder.py` | Legacy compatibility/shim | Many functions delegate to `unified_weights`; still used by `analysis_pipeline`, tests, and one fallback in `unified_weights`. | Migrate active callers to `unified_weights`/`weight_service`; then reduce to compatibility aliases or remove. |
| `analysis/cache_io.py` | Active service | Used by GUI plot manager, warm cache script, feature health, correlation view, and tests. | Keep as canonical cache module. |
| `analysis/confidence_bands.py` | Active numerical utility | Used by GUI plot manager, plotting modules, and tests. | Keep. |
| `analysis/correlation_utils.py` | Active but mixed legacy | Used by `analysis_pipeline`, `beta_builder`, `weight_view`, `unified_weights`, tests. | Keep short term. Move matrix/view construction toward `correlation_view`; retire duplicate weighting helpers later. |
| `analysis/correlation_view.py` | Active service | Used by GUI plot manager, correlation detail plotting, warm cache, tests. | Keep. This is a good target boundary. |
| `analysis/data_availability_service.py` | Active facade | Used by browser, GUI plot manager, spillover GUI, tests. | Keep as GUI-facing facade while `analysis_pipeline` is split. |
| `analysis/explanations.py` | Active small utility | Used by GUI plot manager. | Keep unless explanation text moves into plotting metadata. |
| `analysis/feature_health.py` | Active new service | Used by browser, GUI plot manager, correlation view, RV analysis, tests. | Keep. |
| `analysis/model_fit_service.py` | Active service | Used by GUI plot manager, peer smile composite, `analysis_pipeline`, tests. | Keep as model-fit contract boundary. |
| `analysis/model_params_logger.py` | Active persistence utility | Used by GUI plot manager, parameter GUI, `analysis_pipeline`, RV analysis, tests. | Keep. Consider moving under `analysis/persistence/` later. |
| `analysis/peer_composite_builder.py` | Active legacy builder | Used by GUI plot manager, `analysis_pipeline`, `unified_weights`, peer composite service, RV analysis, tests. | Do not delete. Split surface-grid construction from peer-composite synthesis. |
| `analysis/peer_composite_service.py` | Semi-active orchestration | Used by peer composite viewer, demo script, tests. Not central browser path. | Keep until deciding whether peer-composite viewer/demo remain product features. |
| `analysis/peer_smile_composite.py` | Active focused service | Used by GUI plot manager and tests. | Keep. |
| `analysis/pillar_selection.py` | Active facade | Used by `analysis_pipeline` and route tests; wraps `pillars`. | Keep, but migrate callers here from `pillars` where possible. |
| `analysis/pillars.py` | Legacy mixed implementation | Still used by correlation utils, peer composite builder/service, weight service, beta builder, unified weights, RV analysis, and tests. | Do not delete. Extract remaining implementation into `atm_extraction.py`/`pillar_selection.py`, then leave facade temporarily. |
| `analysis/rv_analysis.py` | Active product surface, too large | Used by RV Signals tab, `analysis_pipeline`, tests. | Keep. Split into signal generation, model/data quality, event context, and presentation payload services. |
| `analysis/rv_heatmap_service.py` | Active facade | Used by GUI plot manager and route tests. | Keep while `prepare_rv_heatmap_data` remains in `analysis_pipeline`. |
| `analysis/settings.py` | Active canonical defaults | Imported broadly across GUI, analysis, scripts, plotting. | Keep. |
| `analysis/smile_data_service.py` | Active facade | Used by GUI plot manager and route tests. | Keep while `get_smile_slice` and `prepare_smile_data` remain in `analysis_pipeline`. |
| `analysis/spillover/network_graph.py` | Active graph adapter | Used by spillover GUI and tests. | Keep. |
| `analysis/spillover/vol_spillover.py` | Active spillover computation | Used by spillover GUI, warm cache script, tests. | Keep, but consider splitting data loading from signal computation. |
| `analysis/term_data_service.py` | Active facade | Used by GUI plot manager and route tests. | Keep while `prepare_term_data` remains in `analysis_pipeline`. |
| `analysis/term_view.py` | Active plotting computation helper | Used by term plotting and tests. | Keep. |
| `analysis/unified_weights.py` | Active but oversized core | Used by weight service/view, correlation view, feature health, beta builder, peer composite service, tests. | Keep. Split feature construction, weighting algorithms, diagnostics, and fallback policy. |
| `analysis/weight_service.py` | Active facade | Used by `analysis_pipeline` and route tests. | Keep as canonical peer-weight service. |
| `analysis/weight_view.py` | Active GUI adapter | Used by GUI plot manager and tests. | Keep, but keep it thin. |

## Removal Candidates

These are the only currently credible removal/archive candidates:

1. `analysis/__pycache__/`
   - Generated runtime artifacts.
   - Safe to delete locally.

2. `analysis/compute_or_load.py`
   - Already deleted in the working tree.
   - Keep deletion if tests continue to pass; `analysis/cache_io.py` is the canonical replacement.

3. `analysis/analysis_background_tasks.py`
   - Only direct caller found: `data/db_maintainance.py`.
   - Not safe to delete without deciding whether the maintenance command is still used.
   - Better move target: `data/maintenance_spillover.py` or `scripts/maintenance/`.

4. `analysis/beta_builder.py`
   - Not safe to delete today.
   - It is a compatibility/shim module with active callers.
   - Delete only after `analysis_pipeline`, tests, and `unified_weights` no longer import it.

5. `analysis/pillars.py`
   - Not safe to delete today.
   - It remains a central dependency even though newer `atm_extraction.py` and
     `pillar_selection.py` boundaries exist.
   - Convert to facade only after callers move.

## Migration Plan

### Phase 1: Remove Generated And Already-Replaced Artifacts

- Delete local `analysis/__pycache__/`.
- Keep `analysis/compute_or_load.py` removed.
- Run:
  - `rg "compute_or_load" analysis display scripts tests`
  - `venv/bin/python -m py_compile analysis/*.py analysis/spillover/*.py`
  - focused cache tests.

### Phase 2: Finish Facade Migration

- Move active GUI and analysis imports toward:
  - `analysis.services.smile_data_service`
  - `analysis.services.term_data_service`
  - `analysis.services.rv_heatmap_service`
  - `analysis.services.data_availability_service`
  - `analysis.weights.weight_service`
  - `analysis.surfaces.model_fit_service`
  - `analysis.surfaces.atm_extraction`
  - `analysis.surfaces.pillar_selection`
- Keep old paths temporarily with route tests.

### Phase 3: Shrink Legacy Core Modules

- Split `analysis_pipeline.py` by workflow:
  - smile data
  - term data
  - RV heatmap data
  - market/data availability
  - historical/cache utilities
- Split `unified_weights.py` by responsibility:
  - feature matrix construction
  - algorithm implementations
  - diagnostics/quality gate
  - public resolver service
- Split `rv_analysis.py` by product layer:
  - raw signal computation
  - model/data quality
  - event/spillover context
  - dashboard payload formatting

### Phase 4: Retire Compatibility Modules

- Migrate `beta_builder.py` callers to `unified_weights` and `weight_service`.
- Migrate `pillars.py` callers to `atm_extraction` and `pillar_selection`.
- Only remove compatibility files after `rg` import scans and tests show no active dependency.

## Recommended Next Patch

Make one small cleanup patch:

- Delete `analysis/__pycache__/`.
- Confirm `analysis/compute_or_load.py` stays deleted.
- Add or update a task under the existing refactor task to track:
  - `beta_builder.py` retirement
  - `pillars.py` facade migration
  - `analysis_pipeline.py` service extraction
  - `rv_analysis.py` split
  - `unified_weights.py` split

