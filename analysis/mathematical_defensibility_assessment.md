# Mathematical Defensibility Assessment

This document assesses whether the major calculations currently implemented in the repository are mathematically defensible (i.e., coherent with standard quantitative/statistical practice), and where caveats apply.

## Scope

Assessment covers the concrete formulas and transformations used in:

- `analysis/unified_weights.py`
- `analysis/beta_builder.py`
- `analysis/correlation_utils.py`

## Verdict legend

- **Defensible**: standard method with reasonable implementation details.
- **Conditionally defensible**: method is valid, but assumptions/edge-case handling can materially change interpretation.
- **Not defensible as stated**: formula or implementation has a clear mathematical mismatch.

---


## Quick answer for the requested modes

### cosine
- **Mathematically defensible?** **Yes, conditionally.**
- **Cleanness of implementation:** **Reasonably clean** (clear normalization pipeline, explicit guards for zero-sum failure).
- **Main caveat:** if negative similarities are clipped, interpretation becomes long-only similarity mixing rather than full signed matching.

### pca
- **Mathematically defensible?** **Yes, conditionally.**
- **Cleanness of implementation:** **Moderate** (standard SVD-based regression and PC1 extraction, but non-negativity is imposed after solve by clipping).
- **Main caveat:** post-hoc clipping is heuristic and not the same as solving constrained NNLS directly.

### cointegration
- **Mathematically defensible?** **Not assessable in current code path because cointegration weighting is not implemented as a supported method in the unified weighting engine.**
- **Cleanness of implementation:** **N/A for current engine**.
- **What would be defensible:** unit-root prechecks, robust coint test selection (Engle–Granger/Johansen by context), multiple-testing control across many peers, and stable hedge-ratio estimation with rolling validation.

---

## What calculations are currently possible (navigable modes)

If a mode is selectable in the GUI or parsable by `compute_unified_weights`, users should be told exactly what math is actually available.

### Weight methods currently supported by unified engine

- `corr`
- `pca`
- `cosine`
- `equal`
- `oi`

### Feature sets currently supported by unified engine

- `iv_atm`
- `iv_atm_ranks`
- `surface`
- `surface_grid` (alias of `surface`)
- `ul`

### GUI-exposed selections (navigable in the app)

- Weight method dropdown: `corr`, `pca`, `cosine`, `equal`, `oi`
- Feature mode dropdown: `iv_atm`, `ul`, `surface`, `surface_grid`

This means the user-facing composite modes are effectively:

- `corr_iv_atm`, `corr_ul`, `corr_surface`, `corr_surface_grid`
- `pca_iv_atm`, `pca_ul`, `pca_surface`, `pca_surface_grid`
- `cosine_iv_atm`, `cosine_ul`, `cosine_surface`, `cosine_surface_grid`
- `equal_iv_atm`, `equal_ul`, `equal_surface`, `equal_surface_grid`
- `oi` (method-only mode; feature token not required)

### Important exclusions (to avoid user confusion)

- `cointegration` is **not** a currently supported `WeightMethod` in the unified engine, and is **not** exposed in GUI method selections.
- Legacy helper routines in other modules may mention other weighting styles (e.g., distance/similarity in local correlation utilities), but those are not the same as a first-class unified mode.

---

## 1) Data transforms and normalization

### 1.1 Log returns
Formula used: `r_t = log(P_t / P_{t-1})`.

- **Verdict**: **Defensible**.
- **Why**: This is a canonical return transform in quantitative finance and is time-additive.

### 1.2 Median imputation by feature/column
Missing values are filled with per-column medians before similarity/PCA operations.

- **Verdict**: **Conditionally defensible**.
- **Why**: Median imputation is robust and common, but can attenuate dispersion and bias pairwise similarity when missingness is not random.

### 1.3 Z-scoring by feature/column
Standardization uses sample std (`ddof=1`) and substitutes `sd=1` for non-finite/non-positive std.

- **Verdict**: **Defensible**.
- **Why**: Standardization is appropriate for combining heterogeneous features; guarding zero variance avoids numerical failure.

---

## 2) Weighting methods

### 2.1 Correlation weights
Pipeline: pairwise correlation with target -> optional clip negative values to zero -> optional power transform -> L1 normalization.

- **Verdict**: **Conditionally defensible**.
- **Why**:
  - Using correlation as affinity is standard.
  - Clipping negatives is mathematically coherent only if the objective is *long-only similarity pooling* (not hedge replication).
  - Power transform is a valid monotone concentration control.
- **Caveat**: If negative comovement is economically informative, clipping discards signal.

### 2.2 Cosine similarity weights
Pipeline: cosine(target, peer) -> optional clip negatives -> optional power transform -> normalize to sum 1.

- **Verdict**: **Conditionally defensible**.
- **Why**: Cosine similarity is a valid directional similarity measure; post-processing into simplex weights is coherent for convex combinations.
- **Caveat**: Cosine depends on centering/scaling choices; without explicit demeaning in this stage, level effects can influence similarity unless upstream standardization handled it.

### 2.3 Equal weights
Each peer gets `1/N`.

- **Verdict**: **Defensible**.
- **Why**: This is mathematically trivial and a common baseline/fallback.

### 2.4 Open-interest weights
`w_i = OI_i / sum_j OI_j` on a chosen as-of date.

- **Verdict**: **Defensible**.
- **Why**: Non-negative, interpretable proportional weighting.
- **Caveat**: It measures market size/liquidity, not necessarily explanatory similarity.

### 2.5 PCA regression weights
Implementation uses SVD-based low-rank solve of `min_w ||X^T w - y||`; optional non-negativity by post-hoc clipping then renormalization.

- **Verdict**: **Conditionally defensible**.
- **Why**: Truncated-SVD regression is a standard regularized linear approximation strategy.
- **Caveat**: Post-hoc clipping is not equivalent to a true non-negative least squares constrained optimum; it is heuristic but often practical.

### 2.6 PCA market-mode weights (PC1)
Builds row-space covariance-like matrix `R = Z Z^T / (p-1)`, ridge on diagonal, first eigenvector, sign fix, clip negative loadings, normalize.

- **Verdict**: **Conditionally defensible**.
- **Why**: PC1 for common-factor exposure is standard.
- **Caveat**: Clipping negative loadings changes the pure PC solution to a long-only approximation; defensible only under long-only construction goals.

---

## 3) Correlation and beta calculations

### 3.1 Pearson-style correlation matrix from standardized ATM rows
Computation effectively forms `(Z Z^T)/(k-1)` after row-wise standardization and adds a small ridge term.

- **Verdict**: **Defensible**.
- **Why**: This is the usual sample correlation construction with numerical stabilization.
- **Caveat**: Adding ridge changes diagonal from exact 1.0; acceptable for stability but should be documented as regularized correlation.

### 3.2 Underlying-return correlations vs benchmark
Computes correlations between benchmark return series and peers.

- **Verdict**: **Defensible**.
- **Why**: Standard benchmark-relative dependence metric.

### 3.3 Beta as covariance/variance ratio
`beta = Cov(x, b) / Var(b)` with minimum sample check and variance positivity guard.

- **Verdict**: **Defensible**.
- **Why**: This is the canonical OLS slope in simple regression of `x` on benchmark `b`.

---

## 4) Pillar selection and filtering logic

### 4.1 Coverage-based pillar/ticker filtering
Drops pillars with too few tickers and tickers with too few pillars.

- **Verdict**: **Conditionally defensible**.
- **Why**: Ensures minimum data support for stable estimates.
- **Caveat**: Introduces selection dependence; resulting correlations are conditional on survivorship in the filtered panel.

### 4.2 Optimized pillar selection from candidate set
Chooses available pillars favoring original set then adding extras up to max count.

- **Verdict**: **Conditionally defensible**.
- **Why**: Practical bias-variance tradeoff for sparse option panels.
- **Caveat**: Cross-date comparability weakens when feature definitions change with selected pillars.

---

## 5) Overall assessment

At a system level, the implemented calculations are **mathematically defensible overall**, with most caveats stemming from **modeling choices** (long-only clipping, imputation, adaptive feature sets) rather than arithmetic or linear-algebra errors.

### Highest-impact caveats to keep explicit in user-facing outputs

1. **Long-only transformations** (negative clipping + renormalization) change interpretation from “best linear representation” to “non-negative mixture approximation.”
2. **Post-hoc non-negativity in PCA regression** is heuristic vs exact constrained optimization.
3. **Adaptive pillar selection** can reduce temporal comparability of derived correlations/weights.

