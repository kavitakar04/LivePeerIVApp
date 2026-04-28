# Weight Modes Diagnosis And Repair Plan

## End-to-end path

GUI selection starts in `display/gui/gui_input.py`:

- `cmb_weight_method`: `corr`, `pca`, `cosine`, `equal`, `oi`
- `cmb_feature_mode`: `iv_atm`, `ul`, `surface`, `surface_grid`
- settings are read by `display/gui/gui_plot_manager.py`
- plot manager calls `analysis.weights.weight_view.resolve_peer_weights`
- resolver calls `analysis.weights.unified_weights.compute_unified_weights`
- `UnifiedWeightComputer.compute_weights` builds features and dispatches by `WeightMethod`
- downstream users include peer-composite surfaces, smiles, term overlays, relative value views, and weight bars

## Mode Audit

### corr

- Responsible functions: `corr_weights_from_matrix`, `UnifiedWeightComputer.compute_weights`
- Expected input: `feature_df` with rows as tickers and columns as comparable features/time observations.
- Actual input: ATM pillar/rank grids, surface grids, or underlying-return matrices depending on GUI feature mode.
- Output: non-negative simplex weights after clipping negative correlations, power transform, shrinkage, validation.
- Risks found: pairwise correlations can become NaN or ill-conditioned with small samples or constant vectors.
- Repair: finite fill, correlation clipping, diagonal reset, shrinkage toward identity, condition-number diagnostics, quality gate.

### pca

- Responsible functions: `pca_regress_weights`, `UnifiedWeightComputer.compute_weights`
- Expected input: peer matrix shaped `n_peers x n_features`, target vector shaped `n_features`.
- Actual input: same feature builders as corr/cosine.
- Output: non-negative simplex weights by default. Signed weights are only possible if `clip_negative=False`.
- Risks found: unregularized SVD can explode or flip under rank deficiency; raw target scaling was inconsistent.
- Repair: row-wise centering, unit-norm scaling, ridge-stabilized SVD solve, near-zero vector rejection, condition diagnostics.

### oi

- Responsible functions: `_open_interest_weights`, `UnifiedWeightComputer.compute_weights`
- Expected input: DB rows with `ticker`, `asof_date`, `open_interest` for all peers.
- Actual input: `options_quotes` aggregated by peer ticker at the selected/latest date.
- Output: non-negative simplex weights.
- Risks found: sparse OI coverage, NaN OI, zero total OI, and extreme concentration could silently distort composites.
- Repair: coverage check, finite/non-negative coercion, zero-sum rejection, max-weight quality gate, equal fallback.

### cosine

- Responsible functions: `cosine_similarity_weights_from_matrix`, `UnifiedWeightComputer.compute_weights`
- Expected input: ticker feature vectors with non-zero norm.
- Actual input: same feature builders as corr/PCA.
- Output: non-negative simplex weights after clipping negative similarities.
- Risks found: near-zero target/peer norms and constant vectors produce undefined cosine similarity.
- Repair: row-wise de-meaning, finite fill, explicit near-zero norm rejection, condition diagnostics, equal fallback.

### equal

- Responsible functions: `UnifiedWeightComputer.compute_weights`, `_fallback_equal`
- Expected input: peer list only.
- Actual input: peer list from GUI.
- Output: uniform non-negative simplex weights.
- Risks found: none beyond empty peer lists.
- Repair: same validation/diagnostics path as computed modes.

## Correctness Criteria

All active GUI modes now pass through a shared quality gate:

- finite values only
- non-negative weights for GUI defaults
- predictable normalization to sum 1
- max absolute weight guard, default `0.98`
- L1 norm guard, default `3.0`
- fallback to equal weights on invalid results

PCA can still support signed weights internally if `clip_negative=False`; GUI defaults keep it non-negative.

## Diagnostics

Each computed or fallback result logs:

- min, max, mean
- sum, L1 norm, L2 norm
- non-finite count
- negative count
- condition number when available
- clipped count
- fallback reason and validation issues

Fallback warnings are attached to the returned `pd.Series.attrs["weight_warning"]` and surfaced in `PlotManager.last_description`.

## Synthetic Regression Coverage

`tests/test_weight_quality_gate.py` covers:

- equal mode simplex behavior
- correlation normalization on deterministic correlated/anti-correlated features
- cosine near-zero norm rejection
- PCA stability under small perturbations
- compute-time fallback on degenerate cosine data
- OI normalization and fallback on extreme concentration

## Deferred Migration Note

Stable internal route IDs still use names such as `synthetic_surface` because changing those IDs would affect saved GUI preferences and plot routing. User-facing labels have moved to "Peer Composite".

Active follow-up tracking lives in `TASKS.MD` under `TASK-010`.
