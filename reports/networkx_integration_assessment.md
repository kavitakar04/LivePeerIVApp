# NetworkX Integration Assessment for LivePeerIVApp

## Executive summary

NetworkX is a strong fit for this repository because the existing spillover and correlation workflows already produce graph-like structures (ticker nodes, weighted relationships, directed responses). The highest-value integration is to add a **graph layer** on top of existing DataFrame outputs rather than replacing current analytics. This keeps risk low and lets current GUI/analysis consumers continue to work unchanged.

## Where NetworkX maps naturally to the current code

### 1) Spillover relationships (`analysis/spillover/vol_spillover.py`)

Current outputs in `run_spillover` include event-level and summary-level peer responses with directional information (`ticker` -> `peer`) and horizon-specific effects. That is directly representable as a directed weighted graph:

- Node: ticker symbol
- Edge: trigger ticker -> responding peer
- Edge attributes:
  - `median_elasticity`
  - `hit_rate`
  - `sign_concord`
  - `h` (horizon)
  - `n` (sample count)

This enables centrality and path analysis without modifying the existing event detector.

### 2) Correlation matrices (`analysis/correlation_utils.py`)

`compute_atm_corr` already returns a correlation matrix suitable for conversion to a weighted undirected graph:

- Keep edges above a configurable threshold (for sparsity)
- Use absolute correlation for topology and signed correlation as an attribute
- Compute community structure and bridge nodes (cross-cluster connectors)

This can improve peer selection beyond fixed `top_k` or static group lists.

### 3) GUI visualization (`display/gui/spillover_gui.py`)

The spillover GUI currently uses table views and line plots. A graph panel can be added as an optional view:

- Node size = out-strength or centrality
- Edge width/color = spillover strength/sign
- Filter by horizon and minimum confidence (`n`, `hit_rate`)

This is additive and can be feature-flagged so existing users are unaffected.

## Recommended integration strategy (low-risk, incremental)

### Phase 1 — Graph adapter utilities (no behavior change)

Add a new module, e.g. `analysis/spillover/network_graph.py`, with pure adapters:

1. `build_spillover_digraph(summary_df, horizon=None, min_n=10, min_hit_rate=0.0)`
2. `build_corr_graph(corr_df, min_abs_corr=0.3)`
3. `compute_graph_metrics(G)` returning DataFrame-friendly metrics

Design principle: all functions accept/return pandas-friendly structures so callers are not forced into NetworkX types unless needed.

### Phase 2 — Use graph metrics as optional ranking signals

Add optional weighting/ranking inputs in synthetic peer workflows:

- Blend existing statistical weight with graph influence score
- Example blend: `final_score = alpha * corr_weight + (1 - alpha) * centrality_norm`
- Keep default `alpha=1.0` to preserve current behavior

This creates measurable value without breaking reproducibility.

### Phase 3 — Visualization and diagnostics

Add graph view to spillover GUI:

- Render from precomputed coordinates (`spring_layout` cached by ticker universe)
- Add controls for horizon/threshold/sign
- Add export to GraphML/CSV edge list for research workflows

## Concrete analysis use-cases unlocked

1. **Systemically important tickers**
   - Identify names that propagate shocks widely (high out-degree/out-strength, PageRank).
2. **Shock vulnerability map**
   - Identify names that receive shocks from many influential peers (in-strength/eigenvector).
3. **Community detection**
   - Discover dynamic clusters that can validate or refine `ticker_groups`.
4. **Contagion path tracing**
   - Explain multi-hop spillovers (A -> B -> C) during stressed windows.
5. **Regime comparison**
   - Compare graph topology by date range (dense vs fragmented markets).

## Data-model and API recommendations

- Keep graph building deterministic by sorting tickers and setting layout seeds.
- Store edge attributes in plain numeric types (avoid object columns).
- Normalize edge weights consistently across horizons before cross-horizon comparisons.
- Avoid overloading correlation and causality: keep separate graph constructors (`corr`, `spillover`).

Suggested function signatures:

```python
def build_spillover_digraph(
    summary: pd.DataFrame,
    horizon: int | None = None,
    weight_col: str = "median_elasticity",
    min_n: int = 10,
    min_hit_rate: float = 0.0,
    include_negative: bool = True,
) -> nx.DiGraph: ...
```

```python
def graph_metrics_df(G: nx.Graph) -> pd.DataFrame: ...
```

## Dependency and performance considerations

- Add `networkx` as an optional dependency first (or soft import with clear error message).
- For typical ticker universes in this project (tens to low hundreds), NetworkX performance is sufficient.
- If this expands to thousands of nodes with repeated recalculation, consider:
  - Caching graphs by `(asof, mode, horizon, thresholds)`
  - Offloading heavy centrality computations
  - Future migration to `igraph`/`graph-tool` only if profiling shows bottlenecks

## Testing approach

Add targeted tests (no GUI dependency required):

1. Graph construction fidelity from known summary fixtures
2. Edge filtering by thresholds (`min_n`, `min_hit_rate`, abs weight)
3. Metric stability (deterministic ordering and finite values)
4. Signed edge handling (positive/negative spillovers)

Potential test files:

- `tests/test_spillover_network_graph.py`
- `tests/test_corr_network_graph.py`

## Risks and mitigations

- **Risk:** Users interpret correlation graph edges as causal.
  - **Mitigation:** Explicit labels and separate constructors for correlation vs spillover graphs.
- **Risk:** Graph metrics overfit sparse/noisy samples.
  - **Mitigation:** hard filters on `n`, optional bootstrap/rolling stability checks.
- **Risk:** Visual clutter in GUI.
  - **Mitigation:** threshold sliders, top-N edge display, cluster-based coloring.

## Suggested first deliverable (1 sprint)

A pragmatic first step is:

1. Implement graph adapters for spillover summary and correlation matrices.
2. Add metric table output (`centrality`, `degree`, `betweenness`) consumable by existing UI tables.
3. Add CLI/demo script that generates and prints top influencers.

This gives immediate analytic value with minimal refactor and a clear path to deeper graph-native features later.
