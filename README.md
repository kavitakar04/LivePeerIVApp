# LivePeerIVApp

LivePeerIVApp is a research platform for analyzing implied volatility across
related assets and identifying relative-value dislocations in options markets.

The system is organized around three core capabilities:

- **IV Browser**: explore, model, and validate volatility structure.
- **Spillover Estimation**: measure cross-asset volatility relationships.
- **Signal Layer**: surface relative-value opportunities with supporting
  diagnostics.

These components share a common pipeline for ingesting option chains, fitting
volatility models, aligning features across tickers, and comparing each target
against a peer-implied baseline.

This repository is research software. It is not investment advice, an execution
system, or a production trading platform.

## What The System Does

LivePeerIVApp converts raw option chains into comparable volatility
representations and uses those representations to study relationships across
tickers.

At a high level:

```text
raw option chains
  -> quote filtering and enrichment
  -> smile / surface modeling
  -> aligned feature construction
  -> peer relationships and spillover structure
  -> relative-value diagnostics
```

The purpose is not simply to compute correlations. The purpose is to understand
where one name's implied-volatility surface is priced differently from related
names, and whether that difference looks meaningful after model, data-quality,
and relationship checks.

## Core Components

### 1. IV Browser

The IV Browser is the primary research interface. It is used to inspect the
volatility structure of a target and its peers before interpreting any signal.

It supports:

- volatility smiles by expiry
- ATM term structures
- target versus peer overlays
- peer-composite term and surface views
- SVI, SABR, and TPS model fits
- model-quality and confidence-band context
- native surface and aligned surface-grid views
- feature representations such as ATM, surface, surface grid, and returns

The browser is meant to make the construction of each comparison visible. If two
tickers are being compared, the user should be able to see whether the
comparison is raw, fitted, weighted, aligned, or degraded.

### 2. Spillover Estimation

The spillover layer estimates how volatility movements propagate across assets.
It provides context for whether a target move appears isolated, peer-driven, or
part of a broader cluster.

It tracks:

- lagged relationships between tickers
- response magnitude
- response direction
- hit rate and sign concordance
- leading and responding assets
- network-style spillover summaries

This matters because a surface difference is not automatically a relative-value
opportunity. It may reflect delayed co-movement, shared catalysts, or a broader
volatility regime. Spillover context helps separate a true dislocation from a
relationship that is still propagating through the peer group.

### 3. Signal Layer

The signal layer combines volatility differences with relationship and quality
context.

It evaluates:

- ATM level differences
- term-structure deviations
- skew and curvature spreads
- localized event-volatility bumps
- full-surface target-minus-peer-composite residuals
- data quality and model-fit health
- spillover support
- contract-level auditability

Signals are intended to be inspected, not blindly acted on. Each signal should
be traceable to a feature representation, peer weighting method, supporting
contracts, and diagnostics.

## Key Concepts

### Target and Peers

Most analysis starts with a target ticker and a peer set.

```text
Target: SPXL
Peers:  UPRO, TQQQ, TNA
```

The system asks how the target differs from the surface implied by the peer
group.

### Feature Representations

Different features define different notions of similarity:

- `iv_atm`: ATM implied-volatility pillars.
- `iv_atm_ranks`: ATM IV by expiry rank.
- `surface`: native expiry-rank x moneyness-bin surface features.
- `surface_grid`: aligned tenor x moneyness grids sampled from fitted surfaces.
- `ul`: underlying price returns.

The selected feature representation determines what the peer weights and
similarity matrix are measuring.

### Peer Weighting

Peer composites can be constructed using:

- `corr`: correlation-based weights.
- `pca`: PCA/regression-style reconstruction weights.
- `cosine`: shape-based similarity weights.
- `oi`: open-interest-based weights.
- `equal`: uniform weights.

Each method produces a different expected surface for the target. All active GUI
weight modes pass through validation checks for finite values, normalization,
concentration, and fallback behavior.

### Surface Alignment

Option chains are irregular across tickers. Expiries, strikes, liquidity, and
quote coverage differ from name to name.

To make cross-name surface comparisons meaningful, the system can:

- fit each expiry smile
- sample a common moneyness grid
- interpolate onto a common tenor axis
- flatten that aligned grid into a feature vector

This avoids treating missing raw quote buckets as genuine surface information.

### Diagnostics

The system is intentionally conservative. It attempts to surface:

- missing or sparse quote coverage
- invalid or non-finite IV values
- model-fit rejection or degradation
- high RMSE or unstable fitted surfaces
- sparse feature matrices
- unstable correlation, PCA, or cosine inputs
- invalid weight vectors
- equal-weight fallback reasons
- weak spillover or comparability support

For this domain, a clean-looking but invalid plot is worse than no plot.

## Typical Research Workflow

A common research loop is:

1. **Select a target and peers**
   - Define the comparison universe.

2. **Inspect the IV Browser**
   - Review smiles, ATM term structure, model fits, and raw/fitted behavior.

3. **Construct a peer composite**
   - Choose a feature representation and weighting method.

4. **Compare target vs peer baseline**
   - Inspect term, smile, and surface differences.

5. **Check spillover relationships**
   - Determine whether the move is isolated, clustered, or still propagating.

6. **Review candidate signals**
   - Evaluate signal magnitude, context, data quality, and auditability.

The workflow moves between structure, relationships, and deviations. A signal is
only interesting if the underlying comparison is valid.

## Example Analysis: SPXL vs Leveraged ETF Peers

One natural peer group is:

```text
Target: SPXL
Peers:  UPRO, TQQQ, TNA
```

A typical analysis might proceed as follows:

1. Load or ingest option chains for all four tickers.
2. Use the IV Browser to inspect SPXL's smile and ATM term structure.
3. Choose `surface_grid` as the feature representation so each ticker is mapped
   onto the same tenor x moneyness grid.
4. Choose `pca` or `corr` weights to build a peer-composite surface.
5. Plot `Peer Composite Surface` to compare:
   - SPXL surface
   - weighted peer-composite surface
   - SPXL minus peer-composite spread
6. Open the spillover view to check whether volatility moves are broad across
   the group or concentrated in SPXL.
7. Review the signal layer to see whether any ATM, skew, curvature, term, or
   surface dislocation survives the quality and relationship checks.

The goal is not to say "SPXL is different" just because a plot shows a spread.
The goal is to determine whether SPXL is different after controlling for peer
structure, model quality, feature alignment, and spillover behavior.

## Running The Application

### Requirements

- Python 3.11 or newer
- A working Tkinter installation for the GUI
- A virtual environment is recommended

### Setup

```bash
git clone <repo-url>
cd LivePeerIVApp

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

On Windows:

```powershell
venv\Scripts\activate
```

### Start the GUI

```bash
make gui
```

Alternative entry points:

```bash
scripts/gui
python -m display.gui.app.browser
```

## GUI Workflow

The main controls are:

- `Target`: ticker under analysis.
- `Peers`: comma-separated peer tickers.
- `Date`: as-of date.
- `Plot`: smile, term structure, relative weight matrix, peer-composite surface,
  RV views, and related displays.
- `Model`: SVI, SABR, or TPS where applicable.
- `Weights`: correlation, PCA, cosine, equal, or OI.
- `Features`: ATM, underlying, native surface, or aligned surface grid.
- `Overlay synth`: show the weighted peer-composite overlay.
- `Show individual peers`: show peer curves/surfaces alongside target.

Useful first views:

- `Smile`
- `Term (ATM vs T)`
- `Relative Weight Matrix`
- `Peer Composite Surface`
- RV signals / anomaly views
- Spillover views

## Command-Line Examples

### Ingest data

```python
from analysis.services.data_availability_service import ingest_and_process

tickers = ["SPY", "QQQ", "IWM"]
ingest_and_process(tickers, max_expiries=6)
```

### Compute peer weights

```python
from analysis.weights.unified_weights import compute_unified_weights

weights = compute_unified_weights(
    target="SPY",
    peers=["QQQ", "IWM"],
    mode="pca_surface_grid",
    asof="2026-04-27",
)

print(weights)
print(weights.attrs.get("weight_diagnostics"))
```

### Build aligned surface grids

```python
from analysis.surfaces.peer_composite_builder import build_surface_grids

surfaces = build_surface_grids(
    tickers=["SPY", "QQQ", "IWM"],
    tenors=(7, 14, 21, 30, 60, 90),
    mny_bins=((0.80, 0.90), (0.90, 1.00), (1.00, 1.10), (1.10, 1.20)),
    surface_source="fit",
    model="svi",
    max_expiries=6,
)
```

### Run a peer-composite demo

```bash
python scripts/peer_composite_demo.py \
  --target SPY \
  --peers QQQ IWM \
  --weight-mode pca \
  --export-dir out/peer_spy \
  --no-show
```

### Audit local data quality

```bash
python scripts/audit_data_quality.py
```

## Architecture

The package is organized by responsibility:

```text
analysis/
  config/        shared defaults and configuration
  persistence/   cache and model-parameter logging
  surfaces/      ATM extraction, smile fits, confidence bands, surface grids
  services/      GUI-facing data preparation services
  weights/       feature matrices, weighting methods, diagnostics
  views/         view models, explanations, feature health, graph helpers
  rv/            relative-value signal generation
  spillover/     spillover analytics and network graph adapters
  jobs/          background and maintenance jobs

data/
  ingestion, SQLite schema, quote quality, Greeks, ticker groups, rates

display/
  gui/           Tkinter application shell, controls, controllers, panels
  plotting/      Matplotlib chart renderers and plotting utilities

volModel/
  SVI, SABR, TPS/poly fitting implementations

reports/
  audits, diagnosis notes, and project framing

tests/
  regression and integration coverage
```

High-level GUI path:

```text
display.gui.controls.gui_input
  -> display.gui.controllers.gui_plot_manager
  -> analysis.services / analysis.weights
  -> analysis.surfaces / analysis.spillover / analysis.rv
  -> display.plotting
```

Peer-weight path:

```text
GUI weight + feature selection
  -> analysis.weights.weight_view.resolve_peer_weights
  -> analysis.weights.unified_weights.compute_unified_weights
  -> feature matrix construction
  -> method-specific weight computation
  -> validation and diagnostics
  -> normalized peer weights
```

Surface-grid path:

```text
raw option quotes
  -> quote filtering
  -> per-expiry smile fit
  -> sampled K/S grid
  -> tenor interpolation
  -> ticker x feature matrix
  -> similarity / weights / peer composite
```

New code should prefer the canonical package paths documented in
`analysis/README.md` and `display/README.md`.

## Data And Runtime Artifacts

The project uses local SQLite and parquet artifacts:

- `data/iv_data.db`: option quotes and related market data.
- `data/calculations.db`: cached computations.
- `data/model_params.parquet`: fitted model parameter snapshots.
- `data/spill_events.parquet`: spillover event records.
- `data/spill_summary.parquet`: summarized spillover relationships.
- `data/gui_settings.json`: local GUI preferences.

These files are runtime state, not source code.

## Testing

Run the full test suite:

```bash
make test
```

Run selected tests:

```bash
python -m pytest tests/test_unified_weight_methods.py -q
python -m pytest tests/test_surfaces.py -q
python -m pytest tests/test_term_plot.py -q
python -m pytest tests/test_rv_analysis.py -q
```

The tests cover:

- option quote ingestion and schema behavior
- surface construction and cache routing
- model-fit quality paths
- weight-mode validation and fallback behavior
- GUI setting propagation
- smile, term, and surface plot behavior
- spillover and graph adapters
- relative-value analysis helpers

## Design Principles

- **Aligned comparisons first**: target and peers must use compatible feature
  definitions.
- **Model outputs must be validated**: failed or degraded fits should not look
  like clean data.
- **Fallbacks should be visible**: equal-weight or model fallback behavior must
  be surfaced.
- **Configuration determines the dataset**: model, grid, feature mode, and
  weight method are part of the analysis definition.
- **Signals are diagnostic outputs**: they are meant to guide inspection, not
  replace judgment.

## Scope

This project is intended for:

- volatility research
- cross-asset comparison
- peer-composite modeling
- relative-value exploration
- diagnostic visualization

It is not:

- an execution system
- a production trading platform
- a substitute for independent market, liquidity, or risk analysis

## Useful Reports

- `reports/data_engineering_project_framing.md`
  - Broader framing of the system as a volatility data platform.

- `reports/weight_modes_diagnosis.md`
  - Weight-mode validation, fallback, and repair notes.

- `reports/data_quality_audit.md`
  - Local data-quality audit output.

- `reports/networkx_integration_assessment.md`
  - Notes on graph-based relationship modeling.

- `reports/settings_unification_audit.md`
  - Notes on configuration consistency and settings migration.

## Summary

LivePeerIVApp provides a unified environment for:

- exploring implied-volatility structure
- estimating relationships between related assets
- constructing peer-composite baselines
- identifying and auditing relative-value dislocations

The system is built to keep comparisons explicit: what data was used, how it was
modeled, how peers were weighted, and what diagnostics support or weaken the
result.

