# VALIDATION.md

## Implied Volatility Validation

A fitted smile or term structure is valid only if:

- all predicted IVs are finite
- all predicted IVs are positive
- predicted IVs are within reasonable bounds
- RMSE is finite
- RMSE is below configured threshold
- enough observations were used
- model metadata correctly identifies the fitted model

Invalid outputs must not be plotted, logged as successful, or selected by `auto`.

## Weight Validation

Weights are valid only if:

- all values are finite
- values have expected sign
- magnitudes are bounded
- normalization is explicit
- dimensionality matches peers/features
- fallback behavior is visible

If validation fails, fall back to `equal` and log the failure.

## Term Structure / Peer Overlay Validation

Before plotting peer overlays:

- target and peers must use the same ATM extraction logic
- maturities must either be aligned or explicitly labeled as raw
- selected IV source column must be logged
- stale or missing expiries must be excluded or marked
- weighted composite must be plotted if weights are selected

## GUI Consistency Validation

For every GUI option:

- option exists in backend registry
- dispatch path is explicit
- output metadata identifies actual model/mode used
- disabled options are visibly disabled for a reason
- fallback is shown in logs or UI status text