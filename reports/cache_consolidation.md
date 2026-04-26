# Cache Consolidation

`analysis.cache_io` is the canonical persistent calculation-cache backend.

- Storage: `data/calculations.db` by default.
- Key: SHA-256 over artifact kind, artifact version, and stable JSON payload.
- Value: zlib-compressed pickle artifact.
- TTL: `analysis.settings.DEFAULT_CALC_CACHE_TTL_SEC`.
- Versioning: per-kind versions live in `analysis.cache_io.ARTIFACT_VERSION`.

The legacy `analysis.compute_or_load` compatibility module has been removed.
New code should import `compute_or_load` from `analysis.cache_io` directly.
