"""Context-aware explanatory text for the vol browser's gray description bar.

Each entry answers three questions for a trader:
  1. What is measured?
  2. How to read high/low values?
  3. What is the trading interpretation?

Keys are (feature_mode, plot_type) tuples.  get_explanation() handles fallbacks
and optional weight-method / overlay append.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Feature-mode × plot-type combinations
# ---------------------------------------------------------------------------

EXPLANATIONS: dict[tuple[str, str], str] = {
    # ── Smile ────────────────────────────────────────────────────────────────
    ("iv_atm", "smile"): (
        "Implied volatility smile fitted to observed option quotes.  "
        "x-axis is moneyness (K/S); the curve is the model fit; dots are market quotes.  "
        "A steep left skew means the market pays up for downside protection — typical in equity/index names.  "
        "A flat or right-skewed smile signals the market fears an upside gap (e.g. single-stock M&A risk).  "
        "RMSE above ~0.010 indicates a stressed or illiquid surface."
    ),
    ("surface_grid", "smile"): (
        "Implied volatility smile with surface-grid synthetic overlay.  "
        "The dashed line is the peer composite interpolated onto the fixed moneyness grid.  "
        "A gap where the target sits above the synthetic means richness relative to peers."
    ),
    # ── Term structure ────────────────────────────────────────────────────────
    ("iv_atm", "term"): (
        "ATM implied volatility plotted across expiries.  "
        "Upward slope (contango) is the default regime — longer-dated vol costs more.  "
        "An inverted (backwardated) curve signals near-term stress: "
        "the market expects higher volatility now than later.  "
        "Watch the front-end for event risk (earnings, Fed) and the back-end for structural vol-of-vol regime signals."
    ),
    ("surface_grid", "term"): (
        "ATM term structure with surface-grid weighted synthetic composite.  "
        "The dashed line blends peer term structures using full-surface weights.  "
        "Divergence at a specific expiry often points to a single-name event premium."
    ),
    # ── Correlation / weight matrix ──────────────────────────────────────────
    ("iv_atm", "corr_matrix"): (
        "Pairwise correlation of ATM implied volatilities ranked by expiry slot.  "
        "Bright cells (near +1) mean those tickers share the same vol dynamics — they tend to move together.  "
        "Dark or blue cells mean independent or counter-moving vol regimes.  "
        "Use this to identify which peers are genuine vol proxies and which are diversifiers."
    ),
    ("iv_atm_ranks", "corr_matrix"): (
        "Pairwise correlation of ATM vol ranked by expiry order (not calendar date).  "
        "Rank alignment makes tenors comparable across tickers with different expiry calendars.  "
        "High correlation means similar vol shapes; low means divergent curves.  "
        "Prefer this when comparing names with unequal expiry density."
    ),
    ("surface", "corr_matrix"): (
        "Pairwise correlation of the full IV surface (all moneyness × expiry cells).  "
        "Captures skew co-movement, not just ATM level changes.  "
        "High correlation means similar skew dynamics — e.g. both tickers price in downside together.  "
        "Use when the skew spread matters more than outright vol level."
    ),
    ("surface_grid", "corr_matrix"): (
        "Pairwise correlation of the fixed-grid IV surface (standard moneyness bins × fixed tenors).  "
        "Eliminates variation from different expiry calendars by interpolating onto a shared grid.  "
        "Most stable basis for synthetic ETF construction — use when dates or tenors differ across tickers."
    ),
    ("oi", "corr_matrix"): (
        "Open-interest weighted correlation — each expiry's contribution is scaled by its OI.  "
        "Heavily-traded maturities drive the signal; low-OI expiries are down-weighted.  "
        "Compare with ATM-rank correlation to spot distortions caused by OI concentration in near-term expiries."
    ),
    ("ul", "corr_matrix"): (
        "Pairwise correlation of underlying equity returns, not implied vol.  "
        "High values mean the two stocks move together in price.  "
        "Note: high equity correlation does not guarantee high IV correlation — skew dynamics can diverge.  "
        "Use as a sanity check: large divergence from IV-based weights is a signal worth investigating."
    ),
    ("volume", "corr_matrix"): (
        "Volume-weighted correlation across the option chain.  "
        "Heavily-traded strikes and expiries carry more weight in the similarity computation.  "
        "Useful when OI may lag actual trading activity (e.g. freshly listed options)."
    ),
    # ── Synthetic surface ─────────────────────────────────────────────────────
    ("iv_atm", "synthetic_surface"): (
        "Three-panel IV surface view: target (left), weighted peer composite (middle), spread (right).  "
        "Red spread cells mean the target is priced richer (higher IV) than peers at that surface point.  "
        "Blue cells mean the target is cheaper.  "
        "A systematic red left-tail band signals the target's downside is priced more aggressively than peers.  "
        "An isolated red spot at a specific expiry often reflects an event premium (earnings, macro)."
    ),
    ("surface_grid", "synthetic_surface"): (
        "Three-panel surface comparison on the fixed moneyness × tenor grid.  "
        "The fixed grid makes spread values comparable across rebalancing dates.  "
        "Persistent red regions suggest structural richness; persistent blue suggests a hedging inefficiency.  "
        "Use this view to track richness/cheapness through time."
    ),
    ("iv_atm_ranks", "synthetic_surface"): (
        "Synthetic surface built from rank-correlated ATM vol peers.  "
        "The composite reflects the median term-structure shape of the highest-correlated peers.  "
        "Spread shows where the target deviates from this consensus shape."
    ),
}

# ---------------------------------------------------------------------------
# Weight method descriptions (HOW weights are computed)
# ---------------------------------------------------------------------------

WEIGHT_MODE_EXPLANATIONS: dict[str, str] = {
    "corr": (
        "Correlation weights: each peer's weight is proportional to its Pearson correlation with the target feature.  "
        "Simple and interpretable — peers that move with the target get more weight.  "
        "Sensitive to outlier dates; consider ridge or PCA for more stability."
    ),
    "ols": (
        "OLS regression weights: solved via ordinary least squares to minimize in-sample reconstruction error.  "
        "Can overfit with many correlated peers — weights may be large or negative.  "
        "Use ridge regularization when peers are highly collinear."
    ),
    "ridge": (
        "Ridge-regularized weights: OLS with an L2 penalty that shrinks large or negative coefficients.  "
        "More stable than plain OLS when peers are correlated; higher lambda pushes toward equal weights.  "
        "Preferred in production replication when robustness matters more than in-sample fit."
    ),
    "pca": (
        "PCA-based weights: the target feature vector is projected onto the principal components of the peer matrix.  "
        "Captures dominant shared variation; robust to collinearity among peers.  "
        "Useful when many peers carry redundant information — PCA compresses them into independent factors."
    ),
    "cosine": (
        "Cosine similarity weights: each peer is sized by the cosine between its feature vector and the target's.  "
        "Scale-invariant — measures shape similarity rather than level similarity.  "
        "Useful when peers have different IV levels but similar surface shapes (e.g. comparing ADRs to U.S. names)."
    ),
    "equal": (
        "Equal weights: all peers receive weight 1/N.  "
        "A model-free benchmark requiring no fitting.  "
        "Use to check whether data-driven weights genuinely improve on equal weighting before committing to a model."
    ),
    "oi": (
        "Open interest weights: peers weighted by relative open interest (liquidity proxy).  "
        "Model-free — no vol correlation computation; allocates to the most actively traded names.  "
        "Best when liquidity is the binding constraint on replication quality."
    ),
}

# ---------------------------------------------------------------------------
# Overlay state descriptions
# ---------------------------------------------------------------------------

OVERLAY_EXPLANATIONS: dict[str, str] = {
    "overlay_synth": (
        "Dashed line: weighted peer composite at the same expiry.  "
        "A gap where the target sits above the synthetic signals richness — potential mean-reversion.  "
        "A gap below signals cheapness relative to what peers imply the vol should be."
    ),
    "overlay_peers": (
        "Semi-transparent curves: individual peer smiles at the nearest matching expiry.  "
        "Compare shapes — a steeper left skew on a peer means more downside protection priced in.  "
        "ATM level differences reflect different vol regimes or staggered earnings timing."
    ),
    "overlay_both": (
        "Both individual peer smiles and the weighted synthetic composite are shown.  "
        "The composite averages across peers; individual curves show where each name diverges.  "
        "Useful for identifying which peer drives the composite at each moneyness point."
    ),
    "overlay_none": (
        "No overlays — only the target's smile is shown.  "
        "Clean view for analyzing the target's vol surface in isolation."
    ),
}

# ---------------------------------------------------------------------------
# Public helper
# ---------------------------------------------------------------------------

_FEATURE_ALIASES: dict[str, str] = {
    "iv_atm_ranks": "iv_atm_ranks",
    "iv_atm": "iv_atm",
    "surface": "surface",
    "surface_grid": "surface_grid",
    "surface_vector": "surface_grid",  # treat as grid
    "oi": "oi",
    "ul": "ul",
    "underlying": "ul",
    "underlying_returns": "ul",
    "volume": "volume",
    "vol": "volume",
}


def get_explanation(
    plot_type: str,
    *,
    feature_mode: str = "iv_atm",
    weight_method: str = "corr",
    overlay_synth: bool = False,
    overlay_peers: bool = False,
    include_weight_method: bool = True,
    include_overlay: bool = True,
) -> str:
    """Return the full static explanation string for the given context.

    Callers should append their own dynamic stats (ATM vol, skew, RMSE, etc.)
    after this string — this function returns only the educational/structural part.
    """
    feature = _FEATURE_ALIASES.get(str(feature_mode).lower(), "iv_atm")
    pid = str(plot_type).lower()

    # Primary lookup with fallbacks
    base = EXPLANATIONS.get((feature, pid)) or EXPLANATIONS.get(("iv_atm", pid)) or ""

    parts: list[str] = [base] if base else []

    # Append weight method explanation for plots where method matters
    if include_weight_method and pid in ("corr_matrix", "synthetic_surface"):
        wm_text = WEIGHT_MODE_EXPLANATIONS.get(str(weight_method).lower(), "")
        if wm_text:
            # One sentence: first sentence of the method explanation
            first = wm_text.split(".")[0].strip() + "."
            parts.append(f"Method — {first}")

    # Append overlay explanation for smile/term plots
    if include_overlay and pid in ("smile", "term"):
        if overlay_synth and overlay_peers:
            ov = OVERLAY_EXPLANATIONS.get("overlay_both", "")
        elif overlay_synth:
            ov = OVERLAY_EXPLANATIONS.get("overlay_synth", "")
        elif overlay_peers:
            ov = OVERLAY_EXPLANATIONS.get("overlay_peers", "")
        else:
            ov = ""
        if ov:
            parts.append(ov)

    return "  ".join(p for p in parts if p)
