# Vol Model Wiring TODO

Created from the vol model audit. Scope is `volModel/`, model call sites in `analysis/`, and GUI plotting paths in `display/`.

## 1. Fix SABR correctness

Status: open

Problem:
- `volModel/sabrFit.py` can produce huge negative or nonsensical implied vols on ordinary smiles.
- The Hagan `z / x(z)` calculation clips negative `x(z)` through `_safe()`, which destroys the sign-preserving ratio.
- Current tests only check imports/finite behavior, so this failure is not caught.

Files:
- `volModel/sabrFit.py`
- `tests/test_volmodel_imports.py` or new `tests/test_sabr_fit.py`
- `analysis/confidence_bands.py`
- `display/gui/gui_plot_manager.py`

Tasks:
- Correct the Hagan lognormal SABR formula sign convention and ratio handling.
- Ensure `sabr_smile_iv()` never returns negative IVs for valid positive inputs.
- Add a synthetic-smile regression test with expected low RMSE and positive fitted vols.
- Add a test for both sides of ATM, including `K < S`, `K = S`, and `K > S`.
- Add a guard in GUI fitting paths to mark SABR unavailable or degraded if fit output is nonfinite, negative, or RMSE is extreme.

Acceptance checks:
- Synthetic quadratic smile fit has reasonable RMSE, ideally below `0.02`.
- `VolModel(model="sabr").smile(...)` returns finite positive IVs across a normal moneyness grid.
- SABR confidence bands do not generate negative or explosive bands.

## 2. Fix pillar SVI wiring

Status: open

Problem:
- `analysis/pillars.py` imports `fit_svi_smile`, which aliases `fit_svi_slice(S, K, T, iv)`.
- `_fit_smile_get_atm()` calls it as `_fit_svi_smile(k, iv, T)`, so the signature is wrong.
- It then expands a params dict into `svi_implied_vol()` incorrectly.
- The broad `except` hides this and silently falls through.

Files:
- `analysis/pillars.py`
- `volModel/sviFit.py`
- Tests covering `compute_atm_by_expiry()` and `_fit_smile_get_atm()`

Tasks:
- Replace the pillar SVI path with the correct API:
  - Either call `fit_svi_slice(S, K, T, iv)` and evaluate with `svi_smile_iv(S, K, T, params)`.
  - Or call `fit_svi_slice_from_moneyness(mny, T, iv)` and evaluate with the raw SVI helpers consistently.
- Remove or narrow the silent `except` so signature/API failures are visible in tests.
- Add a test proving `model="svi"` returns `"model": "svi"` for a valid slice instead of falling back.
- Add a test for `model="auto"` priority order once SABR is repaired or explicitly deprioritized.

Acceptance checks:
- `_fit_smile_get_atm(valid_slice, model="svi")` returns an SVI result with finite `atm_vol`, `skew`, `curv`, and `rmse`.
- `compute_atm_by_expiry(..., method="fit", model="svi")` uses SVI and does not silently fall back for valid slices.

## 3. Clean up TPS/poly duplicate definitions

Status: open

Problem:
- `volModel/polyFit.py` defines `fit_tps_slice()` and `tps_smile_iv()` twice.
- The second definitions override the first ones.
- The two fallback behaviors differ, making future fixes easy to misread.

Files:
- `volModel/polyFit.py`
- `tests/test_polyfit.py`
- `tests/test_tps_plotting.py`
- `tests/test_volmodel_poly.py`

Tasks:
- Collapse `fit_tps_slice()` into one definition.
- Collapse `tps_smile_iv()` into one definition.
- Preserve intentional behavior:
  - Convert strikes to log-moneyness safely.
  - Store `S` and `T` in returned params if downstream code expects them.
  - Use polynomial fallback when an interpolator is unavailable or fails.
- Add a test that imports the final functions and verifies fallback behavior explicitly.

Acceptance checks:
- Existing TPS/poly tests pass.
- A params dict without `interpolator` still returns a sensible quadratic or constant fallback by design.
- No duplicate function definitions remain in `volModel/polyFit.py`.

## 4. Normalize model naming and dispatch

Status: open

Problem:
- `VolModel` declares `ModelName = Literal["svi", "sabr", "poly"]`.
- GUI and plotting code use `"tps"` as the third model name.
- `VolModel(model="tps")` works only accidentally because unknown models fall into the polynomial branch.

Files:
- `volModel/volModel.py`
- `display/plotting/smile_plot.py`
- `display/gui/gui_plot_manager.py`
- `analysis/analysis_pipeline.py`
- Tests for GUI/model dispatch

Tasks:
- Decide one canonical name:
  - Prefer `"tps"` for GUI/user-facing model selection.
  - Keep `"poly"` as a backwards-compatible alias if needed.
- Update `ModelName` literals and docstrings.
- Make dispatch explicit:
  - `"svi"` -> SVI
  - `"sabr"` -> SABR
  - `"tps"` / `"poly"` -> polynomial/TPS path
  - unknown model -> raise `ValueError`
- Update `fit_smile_for()` docs and validation.
- Add tests for accepted names and rejected names.

Acceptance checks:
- `VolModel(model="tps")` works intentionally.
- `VolModel(model="poly")` either works as documented alias or is rejected clearly.
- `VolModel(model="bad")` raises a clear error instead of silently using TPS/poly.

## 5. Add model-quality gates before plotting/logging

Status: open

Problem:
- GUI paths fit all models and log their parameters even if the output is unusable.
- A broken SABR fit can be plotted, cached, and persisted in `data/model_params.parquet`.

Files:
- `display/gui/gui_plot_manager.py`
- `analysis/analysis_pipeline.py`
- `analysis/model_params_logger.py`
- Optional helper in `volModel/`

Tasks:
- Add a shared validation helper for fitted params and predicted curves.
- Require finite params, finite predictions, positive IVs, and bounded RMSE before plotting/logging.
- Surface degraded fit state in plot title or info panel.
- Avoid appending invalid model params to `model_params.parquet`.

Acceptance checks:
- Invalid SABR fits are not logged as normal successful fits.
- Smile plot chooses only the selected model's valid params for display.
- Bad model output produces a clear message rather than a misleading line.

## 6. Strengthen test coverage

Status: open

Problem:
- Current tests passed, but missed the SABR correctness failure and pillar SVI wiring bug.

Files:
- `tests/test_volmodel_imports.py`
- `tests/test_volmodel_poly.py`
- New focused tests as needed

Tasks:
- Add synthetic-fit tests for SVI, SABR, and TPS using the same generated smile.
- Add tests for direct model functions and `VolModel` wrapper behavior.
- Add tests for `_fit_smile_get_atm()` for `model="svi"`, `model="sabr"`, `model="auto"`, and fallback.
- Add a regression test asserting no model returns negative IVs on valid positive inputs.

Acceptance checks:
- The synthetic SABR failure is reproducible before the fix and passes after the fix.
- Pillar SVI wiring failure is reproducible before the fix and passes after the fix.
- Tests validate quality, not only importability.
