import numpy as np
import pandas as pd
from typing import Iterable, List, Tuple
from dataclasses import dataclass

from analysis.surfaces.pillar_selection import build_atm_matrix, detect_available_pillars, EXTENDED_PILLARS_DAYS
from analysis.config.settings import (
    DEFAULT_ATM_BAND,
    DEFAULT_MAX_EXPIRIES,
    DEFAULT_PILLAR_DAYS,
    DEFAULT_PILLAR_TOLERANCE_DAYS,
    DEFAULT_WEIGHT_POWER,
)


@dataclass
class PillarConfig:
    """Configuration for pillar selection and optimization."""

    # Base pillars to use
    base_pillars: List[int] = None

    # Pillar selection mode
    use_restricted_pillars: bool = True
    optimize_pillars: bool = False

    # Optimization parameters
    max_pillars: int = 10
    min_tickers_per_pillar: int = 3

    # Tolerance for pillar matching
    tol_days: float = DEFAULT_PILLAR_TOLERANCE_DAYS

    def __post_init__(self):
        if self.base_pillars is None:
            self.base_pillars = list(DEFAULT_PILLAR_DAYS[:4])

    @classmethod
    def restricted(cls, pillars: List[int] = None) -> "PillarConfig":
        """Create a restricted pillar configuration."""
        return cls(
            base_pillars=pillars or list(DEFAULT_PILLAR_DAYS[:4]),
            use_restricted_pillars=True,
            optimize_pillars=False,
        )

    @classmethod
    def optimized(cls, base_pillars: List[int] = None, max_pillars: int = 8) -> "PillarConfig":
        """Create an optimized pillar configuration."""
        return cls(
            base_pillars=base_pillars or list(DEFAULT_PILLAR_DAYS[:4]),
            use_restricted_pillars=False,
            optimize_pillars=True,
            max_pillars=max_pillars,
        )

    @classmethod
    def extended(cls, base_pillars: List[int] = None, max_pillars: int = 10) -> "PillarConfig":
        """Create an extended pillar configuration without optimization."""
        return cls(
            base_pillars=base_pillars or list(DEFAULT_PILLAR_DAYS[:4]),
            use_restricted_pillars=False,
            optimize_pillars=False,
            max_pillars=max_pillars,
        )


def compute_atm_corr(
    get_smile_slice,
    tickers: Iterable[str],
    asof: str,
    pillars_days: Iterable[int],
    atm_band: float = DEFAULT_ATM_BAND,
    tol_days: float = DEFAULT_PILLAR_TOLERANCE_DAYS,
    min_pillars: int = 2,
    demean_rows: bool = False,
    corr_method: str = "pearson",
    min_tickers_per_pillar: int = 3,
    min_pillars_per_ticker: int = 2,
    ridge: float = 1e-6,
    use_restricted_pillars: bool = True,
    optimize_pillars: bool = False,
    max_pillars: int = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return (ATM matrix, correlation matrix) for one as-of date.

    Parameters
    ----------
    use_restricted_pillars : bool, default True
        If True, use only the provided pillars_days.
        If False, optimize pillar selection based on data availability.
    optimize_pillars : bool, default False
        If True and use_restricted_pillars=False, dynamically select the best
        available pillars to maximize data coverage.
    max_pillars : int, default 10
        Maximum number of pillars to use when optimizing.
    """

    # Pillar optimization logic
    if not use_restricted_pillars and optimize_pillars:
        # Dynamically detect and optimize pillars based on data availability
        candidate_pillars = list(EXTENDED_PILLARS_DAYS) + list(pillars_days)
        candidate_pillars = sorted(set(candidate_pillars))  # Remove duplicates and sort

        # Find pillars with sufficient data coverage
        available_pillars = detect_available_pillars(
            get_smile_slice=get_smile_slice,
            tickers=tickers,
            asof=asof,
            candidate_pillars=candidate_pillars,
            min_tickers_per_pillar=min_tickers_per_pillar,
            tol_days=tol_days,
        )

        if available_pillars:
            # Limit to max_pillars, preferring the original pillars if they're available
            original_pillars = [p for p in pillars_days if p in available_pillars]
            additional_pillars = [p for p in available_pillars if p not in pillars_days]

            # Start with original pillars, then add additional ones up to max_pillars
            optimized_pillars = original_pillars[:max_pillars]
            remaining_slots = max_pillars - len(optimized_pillars)
            if remaining_slots > 0:
                optimized_pillars.extend(additional_pillars[:remaining_slots])

            pillars_days = sorted(optimized_pillars)
            print(f"Optimized pillars for {asof}: {pillars_days} (from {len(candidate_pillars)} candidates)")
        else:
            print(f"Warning: No available pillars found for {asof}, using original: {list(pillars_days)}")
    elif not use_restricted_pillars:
        # Use extended pillars without optimization
        extended_pillars = list(EXTENDED_PILLARS_DAYS) + list(pillars_days)
        pillars_days = sorted(set(extended_pillars))[:max_pillars]
        print(f"Using extended pillars for {asof}: {pillars_days}")

    # Continue with existing logic
    atm_df, corr_df = build_atm_matrix(
        get_smile_slice=get_smile_slice,
        tickers=tickers,
        asof=asof,
        pillars_days=pillars_days,
        atm_band=atm_band,
        tol_days=tol_days,
        min_pillars=min_pillars,
        corr_method=corr_method,
        demean_rows=demean_rows,
    )
    # Drop sparse pillars/tickers (ETF-style filtering)
    if not atm_df.empty:
        # keep pillars with at least min_tickers_per_pillar non-NaN entries
        col_coverage = atm_df.count(axis=0)
        good_pillars = col_coverage[col_coverage >= min_tickers_per_pillar].index
        atm_df = atm_df[good_pillars] if len(good_pillars) >= 2 else atm_df
        # keep tickers with at least min_pillars_per_ticker pillars
        row_coverage = atm_df.count(axis=1)
        good_tickers = row_coverage[row_coverage >= min_pillars_per_ticker].index
        atm_df = atm_df.loc[good_tickers] if len(good_tickers) >= 2 else atm_df
        # recompute correlation with ridge regularisation
        if not atm_df.empty and atm_df.shape[0] >= 2 and atm_df.shape[1] >= 2:
            atm_clean = atm_df.dropna()
            if atm_clean.shape[0] >= 2 and atm_clean.shape[1] >= 2:
                atm_std = (atm_clean - atm_clean.mean(axis=1).values.reshape(-1, 1)) / (
                    atm_clean.std(axis=1).values.reshape(-1, 1) + 1e-8
                )
                corr_matrix = (atm_std @ atm_std.T) / max(atm_std.shape[1] - 1, 1)
                corr_matrix += ridge * np.eye(corr_matrix.shape[0])
                corr_df = pd.DataFrame(corr_matrix, index=atm_clean.index, columns=atm_clean.index)
    return atm_df, corr_df


def compute_atm_corr_optimized(
    get_smile_slice,
    tickers: Iterable[str],
    asof: str,
    base_pillars_days: Iterable[int] = DEFAULT_PILLAR_DAYS[:4],
    **kwargs,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[int]]:
    """
    Compute ATM correlations with automatic pillar optimization.

    Returns
    -------
    atm_df : pd.DataFrame
        ATM matrix (tickers x pillars)
    corr_df : pd.DataFrame
        Correlation matrix (tickers x tickers)
    used_pillars : List[int]
        List of pillars actually used after optimization
    """
    # Set optimization defaults
    kwargs.setdefault("use_restricted_pillars", False)
    kwargs.setdefault("optimize_pillars", True)
    kwargs.setdefault("max_pillars", 8)

    # Store original pillars for comparison
    list(base_pillars_days)

    atm_df, corr_df = compute_atm_corr(
        get_smile_slice=get_smile_slice, tickers=tickers, asof=asof, pillars_days=base_pillars_days, **kwargs
    )

    # Extract used pillars from the ATM dataframe columns
    used_pillars = [int(col) for col in atm_df.columns if str(col).isdigit()]

    return atm_df, corr_df, used_pillars


def compute_atm_corr_restricted(
    get_smile_slice,
    tickers: Iterable[str],
    asof: str,
    pillars_days: Iterable[int],
    **kwargs,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute ATM correlations using only the specified pillars (restricted mode).

    This is equivalent to the original behavior - only use the exact pillars provided.
    """
    # Force restricted pillar usage
    kwargs["use_restricted_pillars"] = True
    kwargs["optimize_pillars"] = False

    return compute_atm_corr(
        get_smile_slice=get_smile_slice, tickers=tickers, asof=asof, pillars_days=pillars_days, **kwargs
    )


def compute_atm_corr_with_config(
    get_smile_slice,
    tickers: Iterable[str],
    asof: str,
    config: PillarConfig,
    **kwargs,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[int]]:
    """
    Compute ATM correlations using a PillarConfig.

    Returns
    -------
    atm_df : pd.DataFrame
        ATM matrix (tickers x pillars)
    corr_df : pd.DataFrame
        Correlation matrix (tickers x tickers)
    used_pillars : List[int]
        List of pillars actually used
    """
    # Apply config settings to kwargs
    kwargs.update(
        {
            "use_restricted_pillars": config.use_restricted_pillars,
            "optimize_pillars": config.optimize_pillars,
            "max_pillars": config.max_pillars,
            "min_tickers_per_pillar": config.min_tickers_per_pillar,
            "tol_days": config.tol_days,
        }
    )

    atm_df, corr_df = compute_atm_corr(
        get_smile_slice=get_smile_slice, tickers=tickers, asof=asof, pillars_days=config.base_pillars, **kwargs
    )

    # Extract used pillars from the ATM dataframe columns
    used_pillars = [int(col) for col in atm_df.columns if str(col).isdigit()]

    return atm_df, corr_df, used_pillars


def corr_weights(
    corr_df: pd.DataFrame,
    target: str,
    peers: List[str],
    clip_negative: bool = True,
    power: float = DEFAULT_WEIGHT_POWER,
) -> pd.Series:
    """Convert correlations with target into normalised positive weights on peers."""
    target = target.upper()
    peers = [p.upper() for p in peers]

    # Debug: Check if target exists in correlation matrix
    if target not in corr_df.columns:
        raise ValueError(f"Target {target} not found in correlation matrix columns: {list(corr_df.columns)}")

    # Extract correlations with target
    s = corr_df.reindex(index=peers, columns=[target]).iloc[:, 0].apply(pd.to_numeric, errors="coerce")

    # Debug: Check raw correlations
    print(f"Raw correlations for {target} vs {peers}:")
    print(f"  Values: {s.to_dict()}")
    print(f"  NaN count: {s.isna().sum()}")
    print(f"  Valid count: {s.notna().sum()}")

    if clip_negative:
        s_before_clip = s.copy()
        s = s.clip(lower=0.0)
        print(f"  After clipping negatives: {s.to_dict()}")
        print(f"  Clipped {(s_before_clip < 0).sum()} negative values")

    if power is not None and float(power) != 1.0:
        s_before_power = s.copy()
        s = s.pow(float(power))
        print(f"  After power={power}: {s.to_dict()}")
        print(f"  Power changed {(s_before_power != s).sum()} values")

    total = float(s.sum())
    print(f"  Total sum: {total}")
    print(f"  Sum is finite: {np.isfinite(total)}")
    print(f"  Sum > 0: {total > 0}")

    if not np.isfinite(total) or total <= 0:
        print("ERROR: Correlation weights computation failed!")
        print(f"  Correlation matrix shape: {corr_df.shape}")
        print(f"  Available tickers: {list(corr_df.index)}")
        print(f"  Requested peers: {peers}")
        print(f"  Final weights before normalization: {s.to_dict()}")
        raise ValueError("Correlation weights sum to zero or NaN")

    return (s / total).fillna(0.0)


def flexible_weights(
    data_df: pd.DataFrame,
    target: str,
    peers: List[str],
    weight_mode: str = "auto",
    clip_negative: bool = True,
    power: float = DEFAULT_WEIGHT_POWER,
    min_weight_threshold: float = 0.01,
) -> pd.Series:
    """
    Flexible weight computation that adapts to data quality and weight mode.

    This function is much more lax with weight modes and determines importance
    based on weights rather than strict correlation requirements.

    Parameters
    ----------
    data_df : pd.DataFrame
        Feature matrix (tickers x features) or correlation matrix
    target : str
        Target ticker
    peers : List[str]
        Peer tickers
    weight_mode : str, default "auto"
        Weight computation mode: "auto", "corr", "equal", "distance", "similarity"
    clip_negative : bool, default True
        Whether to clip negative weights to zero
    power : float, default 1.0
        Power to apply to weights before normalization
    min_weight_threshold : float, default 0.01
        Minimum weight threshold for inclusion (helps filter noise)

    Returns
    -------
    pd.Series
        Normalized weights for peers
    """
    target = target.upper()
    peers = [p.upper() for p in peers]

    if data_df is None or data_df.empty or not peers:
        raise ValueError("invalid data for weight computation")

    if weight_mode == "equal":
        weights = pd.Series(1.0, index=peers, dtype=float)

    elif weight_mode in ("corr", "correlation", "auto"):
        if data_df.shape[1] < 2 and weight_mode == "auto":
            raise ValueError("not enough features for correlation weights")
        if target in data_df.index and target in data_df.columns:
            corr_series = data_df.loc[peers, target]
        elif target in data_df.index:
            feature_corr = data_df.T.corr()
            if target not in feature_corr.columns:
                raise ValueError("target not present for correlation computation")
            corr_series = feature_corr.loc[peers, target]
        else:
            raise ValueError("target not found in data")
        weights = corr_series.fillna(0.0)

    elif weight_mode in ("distance", "similarity"):
        if target not in data_df.index:
            raise ValueError("target not found in data")
        target_features = data_df.loc[target].fillna(0.0)
        peer_features = data_df.loc[data_df.index.intersection(peers)].fillna(0.0)
        if peer_features.empty:
            raise ValueError("peer feature data unavailable")
        if weight_mode == "distance":
            distances = np.sqrt(((peer_features - target_features) ** 2).sum(axis=1))
            weights = 1.0 / (1.0 + distances)
        else:
            target_norm = np.linalg.norm(target_features)
            if target_norm <= 0:
                raise ValueError("target feature norm is zero")
            similarities = peer_features.dot(target_features) / (np.linalg.norm(peer_features, axis=1) * target_norm)
            weights = pd.Series(similarities, index=peer_features.index)
        weights = weights.reindex(peers).fillna(0.0)

    else:
        raise ValueError(f"unknown weight_mode {weight_mode}")

    weights = weights.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    if clip_negative:
        weights = weights.clip(lower=0.0)
    if power is not None and float(power) != 1.0:
        weights = weights.pow(float(power))
    if min_weight_threshold > 0:
        weights = weights.where(weights >= min_weight_threshold, 0.0)

    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("weight computation produced zero sum")
    weights = weights / total

    weights = weights.reindex(peers).fillna(0.0)

    return weights


def compute_atm_corr_pillar_free(
    get_smile_slice,
    tickers: Iterable[str],
    asof: str,
    max_expiries: int = DEFAULT_MAX_EXPIRIES,
    atm_band: float = DEFAULT_ATM_BAND,
    min_tickers: int = 2,
    corr_method: str = "pearson",
    min_periods: int = 2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute ATM correlations without using fixed pillars.

    Instead of using fixed pillar days, this function:
    1. Extracts ATM volatility for each available expiry per ticker
    2. Aligns tickers by expiry rank (1st, 2nd, 3rd shortest expiry, etc.)
    3. Computes correlations across expiry ranks

    This approach uses all available data and doesn't suffer from pillar
    alignment issues that can create too many NaNs.

    Parameters
    ----------
    get_smile_slice : callable
        Function to get option data for a ticker on a date
    tickers : Iterable[str]
        List of tickers to analyze
    asof : str
        Analysis date
    max_expiries : int, default 6
        Maximum number of expiry ranks to consider
    atm_band : float, default 0.05
        Moneyness band around 1.0 to define ATM
    min_tickers : int, default 2
        Minimum tickers needed for correlation
    corr_method : str, default "pearson"
        Correlation method
    min_periods : int, default 2
        Minimum overlapping observations for correlation

    Returns
    -------
    atm_df : pd.DataFrame
        ATM matrix (tickers x expiry_ranks)
    corr_df : pd.DataFrame
        Correlation matrix (tickers x tickers)
    """
    from analysis.views.correlation_view import corr_by_expiry_rank

    return corr_by_expiry_rank(
        get_slice=get_smile_slice,
        tickers=tickers,
        asof=asof,
        max_expiries=max_expiries,
        atm_band=atm_band,
        min_tickers=min_tickers,
        corr_method=corr_method,
        min_periods=min_periods,
    )


def adaptive_correlation_computation(
    get_smile_slice,
    tickers: Iterable[str],
    asof: str,
    pillars_days: Iterable[int],
    weight_mode: str = "auto",
    **kwargs,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Compute correlations without fallback strategies."""
    atm_df, corr_df = compute_atm_corr(
        get_smile_slice=get_smile_slice,
        tickers=tickers,
        asof=asof,
        pillars_days=pillars_days,
        **kwargs,
    )

    metadata = {
        "strategy_used": "primary",
        "pillars_attempted": list(pillars_days),
        "pillars_used": list(atm_df.columns) if not atm_df.empty else [],
        "data_quality": "unknown",
        "fallback_applied": False,
    }

    return atm_df, corr_df, metadata
