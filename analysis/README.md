# Analysis Package Layout

The analysis package is organized by responsibility so cross-name relative-value
comparisons remain traceable from GUI request to computed output.

## Public Module Layout

`analysis.analysis_pipeline` remains the high-level GUI-oriented facade. Focused
implementation modules live in the subpackages below; import those canonical
paths directly.

## Implementation Packages

- `config/`: shared defaults and configuration constants.
- `persistence/`: durable caches and model-parameter logging.
- `surfaces/`: ATM extraction, pillars, smile/surface fits, confidence bands,
  and peer-composite surface primitives.
- `services/`: GUI-facing data preparation for smiles, term structure,
  availability, RV heatmaps, and peer-composite workflows.
- `weights/`: correlation, beta, unified weighting, and weight view logic.
- `views/`: view-model preparation, graph/explanation helpers, and UI summaries.
- `rv/`: relative-value signal generation and dashboard logic.
- `jobs/`: background analysis and maintenance jobs.
- `spillover/`: spillover analytics and network graph adapters.

## Migration Rule

New code should import implementation modules directly, for example:

```python
from analysis.services.smile_data_service import prepare_smile_data
from analysis.weights.unified_weights import compute_unified_weights
from analysis.surfaces.peer_composite_builder import build_surface_grids
```

Root-level compatibility routes have been removed. Use the canonical package
paths above.
