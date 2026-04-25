# Analysis Package Cleanup Todo

Purpose: keep `analysis/` focused on current, tested analysis services and avoid
legacy compatibility code being mixed into active paths.

## Immediate cleanup

- [x] Remove unused placeholder `analysis/cache_warmup.py`.
- [x] Remove or archive unused legacy `analysis/spillover/spillover_engine.py`.
- [x] Move `analysis/networkx_integration_assessment.md` to `reports/`.
- [x] Remove local `analysis/**/__pycache__` artifacts from the workspace.
- [x] Fix `analysis/cache_io.WarmupWorker` so it calls its cache API with a DB path.

## Consolidation

- [ ] Pick one persistent calculation-cache API:
  - `analysis/compute_or_load.py` currently writes to the main IV DB.
  - `analysis/cache_io.py` currently writes TTL/versioned compressed artifacts.
- [ ] Make `analysis/unified_weights.py` the only weight engine.
- [ ] Retire compatibility weight dispatch in `analysis/beta_builder.py` after callers migrate.
- [x] Consolidate expiry-rank ATM correlation logic currently duplicated in
  `analysis/correlation_utils.py` and `analysis/correlation_view.py`.
- [ ] Split `analysis/analysis_pipeline.py` into focused services or reduce it to a facade.
- [ ] Split `analysis/pillars.py` into pillar selection, ATM extraction, and model-fit helpers.
- [ ] Consolidate `analysis/syntheticETFBuilder.py` and `analysis/analysis_synthetic_etf.py`
  under a clearer snake_case synthetic ETF service.

## Verification before deletion-heavy follow-up

- [ ] Run GUI-facing plot manager tests.
- [ ] Run synthetic ETF tests before changing `analysis_synthetic_etf.py`.
- [ ] Run weight-mode tests before removing `beta_builder` compatibility APIs.
- [ ] Run spillover GUI/tests after removing old spillover engine.
