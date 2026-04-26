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

## Integration Tracking

The implementation work from this assessment was migrated to `TASKS.MD` as
`TASK-012`.

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

## Market knowledge graph layer

The first implementation is additive in `analysis/market_graph.py`.  It builds
a typed `networkx.MultiDiGraph` from existing analysis artifacts rather than
introducing a separate data model.

Current evidence layers:

- Correlation matrix edges: `correlated_with`
- Surface-similarity matrix edges: `similar_surface_to`
- Spillover summary edges: `spills_over_to`
- Peer-composite weight edges: `explains_composite_for`
- Optional theme edges: `shares_theme_with`
- Model quality attributes on ticker nodes

Current analysis outputs:

- `rank_peer_candidates(G)`: ranks peers using surface similarity, absolute
  correlation, spillover strength, composite weight, and model quality.
- `graph_confidence_features(G)`: summarizes peer-count, mean/min graph score,
  evidence density, and whether surface/spillover support is present.
- `explain_rv_signal_with_graph(G)`: emits compact explanation bullets that can
  be surfaced in RV diagnostics.

This is intended to improve analysis quality in three ways:

1. Peer sets can be scored by relationship evidence instead of only by the
   manually entered list.
2. RV confidence can distinguish broad support from a single fragile evidence
   layer.
3. The GUI can explain which peers are driving a signal and why those peers are
   valid comparables.

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

## Risks and mitigations

- **Risk:** Users interpret correlation graph edges as causal.
  - **Mitigation:** Explicit labels and separate constructors for correlation vs spillover graphs.
- **Risk:** Graph metrics overfit sparse/noisy samples.
  - **Mitigation:** hard filters on `n`, optional bootstrap/rolling stability checks.
- **Risk:** Visual clutter in GUI.
  - **Mitigation:** threshold sliders, top-N edge display, cluster-based coloring.
