# AGENTS.md

You are working on an implied volatility research and visualization system.

## Core Objective

The ultimate purpose of this project is to identify **relative value trading opportunities** from the options market.

The premise is that an options surface encodes forward-looking information about how the market is pricing uncertainty across strikes, expiries, and names.

For related stocks:
- volatility structures should often move together
- shared moves may reflect common catalysts
- divergences may indicate relative value opportunities

The goal is not just correlation.

The goal is:

> Given a peer group, identify where one name’s implied volatility surface is priced differently, and determine whether that difference is meaningful and potentially tradeable.

This can manifest as:
- term structure differences (ATM vs T)
- skew differences
- localized event volatility bumps
- full surface-level dislocations

All system behavior should support answering:

> Where is the options market pricing one name differently from its peers?

---

## Implications for System Design

1. **Comparability is critical**
   - Target and peers must be computed using identical logic.
   - Any mismatch (ATM definition, expiries, interpolation) invalidates conclusions.

2. **Numerical correctness is non-negotiable**
   - A broken model (e.g. SABR explosion) produces false signals.
   - False signals = fake trading opportunities.

3. **No silent fallbacks**
   - If SVI fails and falls back to poly, that changes the surface.
   - This must be visible, or the comparison is meaningless.

4. **Weights must be meaningful**
   - Weight modes (corr, pca, etc.) affect peer aggregation.
   - Broken weights distort the “consensus surface”.

5. **Visualization must reflect actual computation**
   - If peers are plotted raw vs aligned vs weighted, this must be explicit.
   - Misleading plots directly lead to incorrect trading intuition.

---

## Global Principles

1. Diagnose before patching.
2. Prefer small, targeted fixes.
3. Never hide failed calculations behind silent fallbacks.
4. Add regression tests for every bug fixed.
5. Validate numerical outputs before plotting, logging, or using downstream.
6. Keep model registry, GUI options, and dispatch logic consistent.
7. Preserve research interpretability:
   - every plotted number must be explainable
   - every comparison must be valid

---

## Required Workflow

For each task:

1. Read the task in `TASKS.md`.

2. Produce a short plan:
   - files to inspect
   - suspected root cause
   - proposed patch
   - tests to add

3. Trace the full data path:
   GUI → dispatch → computation → validation → plotting

4. Implement the smallest safe patch.

5. Add targeted regression tests.

6. Run tests.

7. Report:
   - files changed
   - root cause
   - fix
   - validation results
   - impact on downstream comparisons

---

## Numerical Quality Rules

Any model fit or weight calculation must reject:

- NaN values
- infinite values
- negative implied vols
- absurdly large implied vols
- exploding RMSE
- unstable matrix operations
- invalid shapes or empty inputs

If violated:
- reject result
- log failure
- attempt fallback (if allowed)
- mark output as degraded

---

## Relative Value Integrity Rules (CRITICAL)

Because this system is used for **cross-name comparison**, the following must always hold:

### 1. ATM Consistency
- Same ATM definition for target and peers
- Same strike selection rule
- Same interpolation logic (if any)

### 2. Expiry Alignment
- Either:
  - align peers to target maturities
  - OR explicitly mark them as raw
- Never silently mix aligned and raw data

### 3. Units Consistency
- All IVs must be in the same unit (decimal, not %)
- No mixed transformations

### 4. Model Consistency
- If comparing model-based fits:
  - same model must be used
  - or differences must be explicitly labeled

### 5. Weight Integrity
- If weights are selected:
  - they must actually be applied
  - normalization must be correct
- If weights fail:
  - fallback must be visible

---

## Logging Rules

When anything non-standard happens, log:

- requested model / weight mode
- actual model used
- fallback model/mode (if any)
- exception or failure reason
- validation metrics (RMSE, bounds, etc.)
- whether result is degraded

---

## Visualization Rules

Plots must never mislead.

Before plotting:

- validate all inputs
- confirm consistency across target and peers

Plot must clearly indicate:

- raw vs fitted data
- weighted vs individual peer series
- aligned vs non-aligned maturities
- confidence intervals / uncertainty

If comparison is invalid:
- show warning instead of silently plotting

---

## Do Not

- Do not silently coerce broken outputs into valid-looking plots.
- Do not allow different computation paths for target vs peers.
- Do not display a model name that was not actually used.
- Do not accept model output just because it didn’t crash.
- Do not prioritize aesthetics over correctness.

---

## Success Criterion

The system is correct when:

- two names with identical surfaces look identical
- real divergences are preserved
- fake divergences (from bugs, scaling, misalignment) are eliminated

Only then can the output be trusted for identifying:

> genuine relative value opportunities in the options market