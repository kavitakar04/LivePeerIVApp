# Cache Consolidation

`analysis.cache_io` is the canonical persistent calculation-cache backend.

- Storage: `data/calculations.db` by default.
- Key: SHA-256 over artifact kind, artifact version, and stable JSON payload.
- Value: zlib-compressed pickle artifact.
- TTL: `analysis.settings.DEFAULT_CALC_CACHE_TTL_SEC`.
- Versioning: per-kind versions live in `analysis.cache_io.ARTIFACT_VERSION`.

`analysis.compute_or_load` is now only a compatibility import path that delegates
to `analysis.cache_io.compute_or_load`. New code should import from
`analysis.cache_io` directly.
