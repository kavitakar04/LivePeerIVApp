# Settings Unification Audit

This audit tracks analysis defaults that were previously hardcoded across the
codebase and should be user-tunable from a GUI settings panel.

## Central source added

Defaults now live in `analysis/settings.py`.

The module exposes:

- Surface grid: `DEFAULT_SURFACE_TENORS`, `DEFAULT_MONEYNESS_BINS`
- Pillars: `DEFAULT_PILLAR_DAYS`, `DEFAULT_NEAR_TERM_PILLAR_DAYS`,
  `DEFAULT_EXTENDED_PILLAR_DAYS`, `DEFAULT_PILLAR_TOLERANCE_DAYS`
- ATM extraction: `DEFAULT_ATM_BAND`
- Weighting ATM extraction: `DEFAULT_WEIGHT_ATM_BAND`,
  `DEFAULT_WEIGHT_ATM_TOLERANCE_DAYS`
- Expiry limiting: `DEFAULT_MAX_EXPIRIES`
- Relative value: `DEFAULT_RV_LOOKBACK_DAYS`
- Spillover: `DEFAULT_SPILLOVER_EVENT_THRESHOLD`,
  `DEFAULT_SPILLOVER_LOOKBACK_DAYS`,
  `DEFAULT_SPILLOVER_REGRESSION_WINDOW_DAYS`,
  `DEFAULT_SPILLOVER_HORIZONS`
- GUI defaults: model, CI, weight method, feature mode, weight power,
  negative-weight clipping, overlays
- Cache: `DEFAULT_CALC_CACHE_TTL_SEC`

## Hardcoded values found

| Setting | Previous scattered values | Meaning | GUI setting candidate |
| --- | --- | --- | --- |
| Pillar days | `7,30,60,90`, `7,30,60,90,180,365`, `7,14,30` | Fixed target expiries for ATM/pillar analytics | Yes |
| Pillar tolerance | previously `7.0` days; now `14.0` days | Max distance from target pillar | Yes |
| Surface tenors | `7,30,60,90,180,365` | Tenor buckets for surface features | Yes |
| Moneyness bins | previously `0.80-0.90`, `0.95-1.05`, `1.10-1.25`; now contiguous `0.80-0.90`, `0.90-1.10`, `1.10-1.25` | Surface moneyness buckets | Advanced |
| ATM band | `0.05` | Near-ATM moneyness band | Yes |
| Weight ATM band | `0.08` | Wider band used by weight feature matrices | Advanced |
| Weight ATM tolerance | `10.0` days | Pillar match tolerance for weight matrices | Advanced |
| Max expiries | `6` | Limit shortest expiries used by plots/features | Yes |
| RV lookback | `60` | Rolling window for spread z/pct rank | Yes |
| Spillover threshold | `0.10` | Event trigger threshold | Yes, spillover tab |
| Spillover horizons | `1,3,5`; background `0,1,3,5,10` | Response horizons | Yes, spillover tab |
| Spillover regression window | `90` | Historical window for regression weights | Yes, spillover tab |
| Cache TTL | `900` seconds | TTL for `analysis.persistence.cache_io` artifacts | Advanced |
| Weight power | `1.0` | Weight sharpening exponent | Yes |
| Clip negative weights | `True` | Non-negative portfolio weights | Yes |

## What was unified now

- Added `analysis/settings.py` and `AnalysisDefaults`.
- Routed GUI input defaults through `analysis.config.settings`.
- Routed GUI plot-manager fallbacks through `analysis.config.settings`.
- Routed synthetic ETF defaults through `analysis.config.settings`.
- Routed unified-weight defaults through `analysis.config.settings`.
- Routed correlation-view defaults through `analysis.config.settings`.
- Routed spillover defaults through `analysis.config.settings`.
- Routed analysis background spillover defaults through `analysis.config.settings`.
- Routed cache TTL through `analysis.config.settings`.
- Replaced remaining obvious pillar/tenor/moneyness literals in beta/correlation helpers.
- Expanded the default ATM-adjacent moneyness bin to remove silent gaps in
  `0.90-0.95` and `1.05-1.10`.
- Increased general pillar tolerance from 7 to 14 days.
- Added visible GUI controls for max expiries, pillar days, pillar tolerance,
  ATM band, and moneyness bins.

## Important design note

There are two legitimate pillar defaults today:

- `DEFAULT_PILLAR_DAYS = (7, 30, 60, 90, 180, 365)` for GUI/synthetic/RV workflows.
- `DEFAULT_NEAR_TERM_PILLAR_DAYS = (7, 14, 30)` for legacy near-term ATM helper behavior.

The settings panel should expose the GUI/default workflow pillars first. The
near-term default should either be retired or explicitly labeled as a legacy
ATM-extraction default after caller migration.

## Settings Panel Follow-Up

The settings-panel work from this audit was migrated to `TASKS.MD` as
`TASK-011`.
