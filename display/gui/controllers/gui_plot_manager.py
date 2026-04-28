# display/gui/gui_plot_manager.py
from __future__ import annotations
import sys
from pathlib import Path
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Ensure project root on path so local packages resolve first ---
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Plot helpers
from display.plotting.charts.correlation_detail_plot import (
    plot_correlation_details,
)
from display.plotting.charts.smile_plot import fit_and_plot_smile
from display.plotting.charts.term_plot import (
    plot_atm_term_structure,
    plot_term_structure_comparison,
)
from display.plotting.charts.rv_plots import (
    plot_surface_residual_heatmap,
)

# Surfaces & synthetic construction
from analysis.surfaces.peer_composite_builder import build_surface_grids, combine_surfaces

# Data/analysis utilities
from analysis.services.rv_heatmap_service import prepare_rv_heatmap_data
from analysis.services.smile_data_service import get_smile_slice, prepare_smile_data
from analysis.services.term_data_service import prepare_term_data
from analysis.persistence.cache_io import compute_or_load
from display.gui.controls.gui_input import plot_id


from analysis.persistence.cache_io import WarmupWorker
from analysis.views.correlation_view import corr_by_expiry_rank, prepare_correlation_view
from analysis.config.settings import (
    DEFAULT_ATM_BAND,
    DEFAULT_CLIP_NEGATIVE_WEIGHTS,
    DEFAULT_FEATURE_MODE,
    DEFAULT_MAX_EXPIRIES,
    DEFAULT_MONEYNESS_BINS,
    DEFAULT_SMILE_GRID_POINTS,
    DEFAULT_SMILE_MONEYNESS_RANGE,
    DEFAULT_SURFACE_TENORS,
    DEFAULT_WEIGHT_METHOD,
    DEFAULT_WEIGHT_POWER,
)
from analysis.weights.weight_view import resolve_peer_weights
from analysis.views.explanations import get_explanation
from analysis.services.data_availability_service import available_dates  # noqa: F401


from analysis.persistence.model_params_logger import append_params
from analysis.surfaces.atm_extraction import fit_smile_get_atm
from analysis.surfaces.model_fit_service import fit_valid_model_params, fit_valid_model_result
from analysis.surfaces.peer_smile_composite import build_peer_smile_composite
from analysis.surfaces.confidence_bands import (
    svi_confidence_bands,
    sabr_confidence_bands,
    tps_confidence_bands,
)

# ---------------------------------------------------------------------------
# Small local helpers
# ---------------------------------------------------------------------------
LOGGER = logging.getLogger(__name__)
SMILE_MNY_RANGE = DEFAULT_SMILE_MONEYNESS_RANGE
SMILE_RMSE_WARN = 0.05


def _smile_mny_range_from_settings(settings: dict | None) -> tuple[float, float]:
    raw = (settings or {}).get("smile_moneyness_range", DEFAULT_SMILE_MONEYNESS_RANGE)
    try:
        lo, hi = raw
        lo = float(lo)
        hi = float(hi)
        if 0.0 < lo < hi and hi < 10.0:
            return (lo, hi)
    except Exception:
        pass
    return DEFAULT_SMILE_MONEYNESS_RANGE


def _smile_grid_from_settings(settings: dict | None) -> tuple[float, float, int]:
    lo, hi = _smile_mny_range_from_settings(settings)
    return (lo, hi, DEFAULT_SMILE_GRID_POINTS)


def _filter_smile_quotes(
    S: float,
    K: np.ndarray,
    IV: np.ndarray,
    cp: np.ndarray | None = None,
    mny_range: tuple[float, float] = SMILE_MNY_RANGE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, int]:
    K = np.asarray(K, dtype=float)
    IV = np.asarray(IV, dtype=float)
    cp_arr = np.asarray(cp) if cp is not None and len(cp) == len(K) else None
    mny = K / float(S)
    finite = np.isfinite(K) & np.isfinite(IV) & np.isfinite(mny)
    lo, hi = mny_range
    in_band = finite & (mny >= float(lo)) & (mny <= float(hi))
    if int(in_band.sum()) >= 3:
        cp_filtered = cp_arr[in_band] if cp_arr is not None else None
        return K[in_band], IV[in_band], cp_filtered, int(finite.sum() - in_band.sum())
    cp_filtered = cp_arr[finite] if cp_arr is not None else None
    return K[finite], IV[finite], cp_filtered, 0


def _plot_xy_mask(x, y, label: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, str | None]:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if x_arr.shape != y_arr.shape:
        return (
            x_arr,
            y_arr,
            np.zeros(0, dtype=bool),
            f"{label} produced mismatched shapes: x={x_arr.shape}, y={y_arr.shape}",
        )
    if x_arr.size == 0:
        return x_arr, y_arr, np.zeros(0, dtype=bool), f"{label} produced no points"
    valid = np.isfinite(x_arr) & np.isfinite(y_arr)
    if int(valid.sum()) < 2:
        return x_arr, y_arr, valid, f"{label} has fewer than 2 finite points"
    return x_arr, y_arr, valid, None


def _fit_quality_text(rmse: float, quality: dict | None = None) -> str:
    if quality and not quality.get("ok", True):
        return "rejected"
    if not np.isfinite(rmse):
        return "n/a"
    return f"{rmse:.4f}"


def _dedupe_legend(ax: plt.Axes):
    handles, labels = ax.get_legend_handles_labels()
    seen: dict[str, plt.Artist] = {}
    for handle, label in zip(handles, labels):
        if not label or label.startswith("_") or label in seen:
            continue
        seen[label] = handle
    if seen:
        handles_out = list(seen.values())
        labels_out = list(seen.keys())
        if len(labels_out) >= 5:
            if len(ax.figure.axes) == 1:
                ax.figure.subplots_adjust(right=0.78)
            return ax.legend(
                handles_out,
                labels_out,
                loc="upper left",
                bbox_to_anchor=(1.01, 1.0),
                borderaxespad=0.0,
                fontsize=8,
                framealpha=0.92,
            )
        if len(ax.figure.axes) == 1:
            ax.figure.subplots_adjust(right=0.94)
        return ax.legend(handles_out, labels_out, loc="upper right", fontsize=8, framealpha=0.92)
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()
    return None


def _weights_summary(weights, limit: int = 3) -> str:
    if weights is None:
        return ""
    try:
        series = pd.Series(weights, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
        if series.empty:
            return ""
        series = series.reindex(series.abs().sort_values(ascending=False).index)
        parts = [f"{idx} {val:.0%}" for idx, val in series.head(limit).items()]
        return ", ".join(parts)
    except Exception:
        return ""


def _surface_grid_label(df: pd.DataFrame | None) -> str:
    if df is None:
        return "0x0"
    try:
        return f"{int(df.shape[0])} mny x {int(df.shape[1])} tenor"
    except Exception:
        return "unknown grid"


def _surface_axis_range(values) -> str:
    vals = []
    for value in values:
        try:
            text = str(value).strip()
            if "-" in text:
                for part in text.split("-", 1):
                    vals.append(float(part.strip()))
            else:
                vals.append(float(text.split(":")[0]))
        except Exception:
            continue
    if not vals:
        return "n/a"
    return f"{min(vals):g}-{max(vals):g}"


def _surface_weight_lines(weights: pd.Series, peers: list[str], limit: int = 5) -> tuple[str, str]:
    try:
        series = pd.Series(weights, dtype=float).reindex(peers).dropna()
    except Exception:
        return "Weights: n/a", "Peers used: " + ", ".join(peers)
    if series.empty:
        return "Weights: n/a", "Peers used: " + ", ".join(peers)
    series = series.reindex(series.abs().sort_values(ascending=False).index)
    shown = [f"{idx} {val:.0%}" for idx, val in series.head(limit).items()]
    more = len(series) - len(shown)
    suffix = f", +{more} more" if more > 0 else ""
    return "Weights: " + ", ".join(shown) + suffix, "Peers used: " + ", ".join(series.index.astype(str))


def _weight_info(target: str, asof: str, weight_mode: str, weights) -> dict | None:
    if weights is None:
        return None
    try:
        series = pd.Series(weights, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
        if series.empty:
            return None
        warning = getattr(weights, "attrs", {}).get("weight_warning", "")
        return {
            "target": target,
            "asof": asof,
            "mode": weight_mode,
            "weights": {str(k): float(v) for k, v in series.items()},
            "warning": warning,
            "feature_diagnostics": getattr(weights, "attrs", {}).get("feature_diagnostics", {}),
            "feature_health": getattr(weights, "attrs", {}).get("feature_health", {}),
            "weight_diagnostics": getattr(weights, "attrs", {}).get("weight_diagnostics", {}),
        }
    except Exception:
        return None


def _status_event(category: str, status: str, message: str, **kwargs) -> dict:
    event = {"category": category, "status": status, "message": message}
    event.update({k: v for k, v in kwargs.items() if v is not None})
    return event


def _cols_to_days(cols) -> np.ndarray:
    out = []
    for c in cols:
        try:
            out.append(int(float(c)))
        except Exception:
            s = "".join(ch for ch in str(c) if ch.isdigit() or ch in ".-")
            out.append(int(float(s)) if s else 0)
    return np.array(out, dtype=int)


def _nearest_tenor_idx(tenor_days, target_days: float) -> int:
    arr = np.array(list(tenor_days), dtype=float)
    return int(np.argmin(np.abs(arr - float(target_days))))


def _mny_from_index_labels(idx) -> np.ndarray:
    out = []
    for label in idx:
        s = str(label)
        if "-" in s:
            try:
                a, b = s.split("-", 1)
                out.append((float(a) + float(b)) / 2.0)
                continue
            except Exception:
                pass
        try:
            out.append(float(s))
        except Exception:
            num = "".join(ch for ch in s if ch.isdigit() or ch == ".")
            out.append(float(num) if num else np.nan)
    return np.array(out, dtype=float)


def _asof_candidates(asof) -> list:
    """Return candidate keys used by surface dictionaries for a requested as-of value."""
    vals = []
    if asof is None:
        return vals
    vals.append(asof)
    try:
        ts = pd.Timestamp(asof)
        vals.extend([ts, ts.normalize()])
        if ts.tzinfo is not None:
            vals.append(ts.tz_localize(None))
            vals.append(ts.tz_localize(None).normalize())
    except Exception:
        pass

    out = []
    seen = set()
    for v in vals:
        key = (type(v), str(v))
        if key in seen:
            continue
        seen.add(key)
        out.append(v)
    return out


def _value_for_asof(mapping: dict, asof):
    """Retrieve an as-of keyed value from dicts that may use str or Timestamp keys."""
    if not mapping:
        return None
    for k in _asof_candidates(asof):
        if k in mapping:
            return mapping[k]
    return None


# ---------------------------------------------------------------------------
# Plot Manager
# ---------------------------------------------------------------------------
class PlotManager:
    def __init__(self):
        # get_smile_slice is rebound per-plot to respect max_expiries
        self.get_smile_slice = get_smile_slice

        # caches
        self.last_corr_df: pd.DataFrame | None = None
        self.last_atm_df: pd.DataFrame | None = None
        self.last_corr_meta: dict = {}

        # ui state
        self._current_max_expiries = None
        self.canvas = None
        self._cid_click = None
        self._current_plot_type = None

        # smile click-through state (for fast re-render without re-query)
        self._smile_ctx = None  # dict storing arrays + current index + overlay bits

        # preserve last settings for weight computation context
        self.last_settings: dict = {}
        # store latest fit parameters for UI parameter tab
        self.last_fit_info: dict | None = None
        # plain-English description of the current plot (read by the browser)
        self.last_description: str = ""
        self.last_weight_warning: str | None = None

        # cache for surface grids: key is (tickers tuple, max_expiries)
        self._surface_cache: dict[tuple[tuple[str, ...], int], dict] = {}

        # background cache warmer
        self._warm = WarmupWorker("data/calculations.db")

    # -------------------- canvas wiring --------------------
    def attach_canvas(self, canvas):
        self.canvas = canvas
        if self._cid_click is not None:
            try:
                self.canvas.mpl_disconnect(self._cid_click)
            except Exception:
                pass
        self._cid_click = self.canvas.mpl_connect("button_press_event", self._on_click)

    def _clear_correlation_colorbar(self, ax: plt.Axes):
        """Remove any existing correlation colorbar to prevent it from
        appearing on non-correlation plots. Keeps axes geometry consistent
        with tests that assert position stability.
        """
        try:
            if hasattr(ax.figure, "_correlation_colorbar"):
                cbar = getattr(ax.figure, "_correlation_colorbar")
                try:
                    cbar.remove()
                except Exception:
                    pass
                delattr(ax.figure, "_correlation_colorbar")
                if hasattr(ax.figure, "_orig_position"):
                    ax.set_position(ax.figure._orig_position)
                if hasattr(ax.figure, "_orig_subplotpars"):
                    l, r, b, t = ax.figure._orig_subplotpars
                    ax.figure.subplots_adjust(left=l, right=r, bottom=b, top=t)
            for attr in ("_corr_weight_ax", "_corr_coverage_ax", "_corr_colorbar_ax"):
                if hasattr(ax.figure, attr):
                    try:
                        getattr(ax.figure, attr).remove()
                    except Exception:
                        pass
                    try:
                        delattr(ax.figure, attr)
                    except Exception:
                        pass
        except Exception:
            pass

    def _clear_child_axes(self, ax: plt.Axes):
        """Remove inset/helper axes attached to the main plotting axes."""
        try:
            for other in list(ax.figure.axes):
                if other is not ax:
                    try:
                        other.remove()
                    except Exception:
                        pass
            if hasattr(ax.figure, "_surface_aux_axes"):
                for aux in list(getattr(ax.figure, "_surface_aux_axes")):
                    try:
                        aux.remove()
                    except Exception:
                        pass
                delattr(ax.figure, "_surface_aux_axes")
            for child in list(getattr(ax, "child_axes", [])):
                child.remove()
            for attr in (
                "_correlation_colorbar",
                "_corr_weight_ax",
                "_corr_coverage_ax",
                "_corr_colorbar_ax",
                "_rv_heatmap_colorbar",
                "_rv_heatmap_colorbar_ax",
            ):
                if hasattr(ax.figure, attr):
                    try:
                        getattr(ax.figure, attr).remove()
                    except Exception:
                        pass
                    try:
                        delattr(ax.figure, attr)
                    except Exception:
                        pass
        except Exception:
            pass

    def invalidate_surface_cache(self):
        """Clear any cached surface grids."""
        self._surface_cache.clear()

    def _get_surface_grids(self, tickers, max_expiries, mny_bins=None, tenors=None, model="svi", surface_source="fit"):
        """Return surface grids for ``tickers`` using cache if available."""
        bins_key = tuple(tuple(float(x) for x in b) for b in (mny_bins or DEFAULT_MONEYNESS_BINS))
        tenor_key = tuple(int(x) for x in (tenors or DEFAULT_SURFACE_TENORS))
        model_key = str(model or "svi").lower()
        source_key = str(surface_source or "fit").lower()
        key = (tuple(sorted(set(tickers))), int(max_expiries), bins_key, tenor_key, model_key, source_key)
        if key not in self._surface_cache:
            payload = {
                "tickers": list(key[0]),
                "max_expiries": key[1],
                "mny_bins": bins_key,
                "tenors": tenor_key,
                "model": model_key,
                "surface_source": source_key,
            }

            def _builder():
                return build_surface_grids(
                    tickers=list(key[0]),
                    tenors=tenor_key,
                    mny_bins=bins_key,
                    use_atm_only=False,
                    max_expiries=key[1],
                    surface_source=source_key,
                    model=model_key,
                )

            try:
                grids = compute_or_load("surface_grids", payload, _builder)
            except Exception:
                grids = _builder()
            self._surface_cache[key] = grids if grids is not None else {}
        return self._surface_cache.get(key, {})

    # -------------------- main entry --------------------
    def plot(self, ax: plt.Axes, settings: dict):
        plot_type = settings["plot_type"]
        target = settings["target"]
        # Normalize to plain YYYY-MM-DD so DB queries, surface dict lookups,
        # and cache keys all use the same canonical form.
        asof = pd.to_datetime(settings["asof"]).strftime("%Y-%m-%d")
        model = settings["model"]
        T_days = settings["T_days"]
        ci = settings["ci"]
        x_units = settings["x_units"]
        atm_band = settings.get("atm_band", DEFAULT_ATM_BAND)
        mny_bins = settings.get("mny_bins", DEFAULT_MONEYNESS_BINS)
        weight_method = settings.get("weight_method", DEFAULT_WEIGHT_METHOD)
        feature_mode = settings.get("feature_mode", DEFAULT_FEATURE_MODE)
        # backward compatibility: allow legacy weight_mode field
        legacy = settings.get("weight_mode")
        if legacy and not settings.get("weight_method"):
            if legacy == "oi":
                weight_method, feature_mode = "oi", DEFAULT_FEATURE_MODE
            elif "_" in legacy:
                weight_method, feature_mode = legacy.split("_", 1)
            else:
                weight_method, feature_mode = legacy, DEFAULT_FEATURE_MODE
        weight_mode = "oi" if weight_method == "oi" else f"{weight_method}_{feature_mode}"
        # TODO: Fix setting- marked as always true.
        overlay_synth = settings.get("overlay_synth", True)
        overlay_peers = settings.get("overlay_peers", False)
        show_term_fit = settings.get("show_term_fit", False)
        peers = settings["peers"]
        pillars = settings["pillars"]
        max_expiries = settings.get("max_expiries", DEFAULT_MAX_EXPIRIES)
        smile_grid = _smile_grid_from_settings(settings)

        # invalidate surface cache if tickers or max_expiries changed
        prev = getattr(self, "last_settings", {}) or {}
        prev_tickers = set([prev.get("target", "")] + prev.get("peers", []))
        curr_tickers = set([target] + peers)
        if (
            prev_tickers != curr_tickers
            or prev.get("max_expiries") != max_expiries
            or prev.get("mny_bins") != mny_bins
            or prev.get("pillars") != settings.get("pillars")
        ):
            self.invalidate_surface_cache()

        # reset last-fit info before plotting
        self.last_fit_info = None

        # remember for other helpers
        self._current_plot_type = plot_type
        self._current_max_expiries = max_expiries
        self.last_settings = settings

        # stable routing ID — all dispatch below uses this, not the raw label
        pid = plot_id(plot_type)

        # create a bounded get_smile_slice with current max_expiries
        def bounded_get_smile_slice(ticker, asof_date=None, T_target_years=None, call_put=None, nearest_by="T"):
            return get_smile_slice(
                ticker, asof_date, T_target_years, call_put, nearest_by, max_expiries=self._current_max_expiries
            )

        self.get_smile_slice = bounded_get_smile_slice

        self._clear_child_axes(ax)
        ax.clear()

        # --- Smile: click-through (preload all expiries for date) ---
        if pid == "smile":
            self._clear_correlation_colorbar(ax)

            weights = None
            if peers:
                weights = self._weights_from_ui_or_matrix(
                    target,
                    peers,
                    weight_mode,
                    asof=asof,
                    pillars=self.last_corr_meta.get("pillars") if self.last_corr_meta else None,
                )

            payload = {
                "target": target,
                "asof": pd.to_datetime(asof).floor("min").isoformat(),
                "T_days": float(T_days),
                "model": model,
                "ci": ci,
                "overlay_synth": overlay_synth,
                "peers": sorted(peers),
                "weights": weights.to_dict() if weights is not None else None,
                "overlay_peers": overlay_peers,
                "max_expiries": max_expiries,
                "smile_moneyness_range": list(smile_grid[:2]),
                "v": 3,  # bump to invalidate caches missing peer slices for composite smile
            }

            def _builder():
                return prepare_smile_data(
                    target=target,
                    asof=asof,
                    T_days=T_days,
                    model=model,
                    ci=ci,
                    overlay_synth=overlay_synth,
                    peers=peers,
                    weights=weights.to_dict() if weights is not None else None,
                    overlay_peers=overlay_peers,
                    max_expiries=max_expiries,
                )

            if hasattr(self, "_warm"):
                try:
                    self._warm.enqueue("smile", payload, _builder)
                except Exception:
                    pass

            data = compute_or_load("smile", payload, _builder)

            if not data:
                ax.set_title("No data")
                self.last_fit_info = {
                    "ticker": target,
                    "asof": asof,
                    "fit_by_expiry": {},
                    "weight_info": _weight_info(target, asof, weight_mode, weights),
                    "status_events": [
                        _status_event("data", "error", "No smile data available for the selected ticker/date.")
                    ],
                }
                return

            fit_map = data.get("fit_by_expiry", {})
            status_events = [
                _status_event(
                    "data",
                    "info",
                    f"Loaded {len(data['Ts'])} expiries and "
                    f"{int(np.isfinite(data['sigma_arr']).sum())} finite IV quotes.",
                    n=int(np.isfinite(data["sigma_arr"]).sum()),
                )
            ]
            self._smile_ctx = {
                "ax": ax,
                "T_arr": data["T_arr"],
                "K_arr": data["K_arr"],
                "sigma_arr": data["sigma_arr"],
                "S_arr": data["S_arr"],
                "cp_arr": data.get("cp_arr"),
                "Ts": data["Ts"],
                "idx": data["idx0"],
                "settings": settings,
                "weights": weights,
                "tgt_surface": data.get("tgt_surface"),
                "syn_surface": data.get("syn_surface"),
                "peer_slices": data.get("peer_slices", {}),
                "expiry_arr": data.get("expiry_arr"),
                "fit_by_expiry": fit_map,
                "status_events": status_events,
            }
            self.last_fit_info = {
                "ticker": target,
                "asof": asof,
                "fit_by_expiry": fit_map,
                "weight_info": _weight_info(target, asof, weight_mode, weights),
                "status_events": status_events,
            }
            self._render_smile_at_index()
            return

        # --- Term: needs all expiries for this day ---
        elif pid == "term":
            self._clear_correlation_colorbar(ax)

            weights = None
            if peers:
                weights = self._weights_from_ui_or_matrix(
                    target,
                    peers,
                    weight_mode,
                    asof=asof,
                    pillars=self.last_corr_meta.get("pillars") if self.last_corr_meta else None,
                )

            payload = {
                "target": target,
                "asof": pd.to_datetime(asof).floor("min").isoformat(),
                "ci": ci,
                "overlay_synth": overlay_synth,
                "peers": sorted(peers),
                "weights": weights.to_dict() if weights is not None else None,
                "atm_band": atm_band,
                "max_expiries": max_expiries,
                "feature_mode": feature_mode,
                "weight_mode": weight_mode,
            }

            def _builder():
                return prepare_term_data(
                    target=target,
                    asof=asof,
                    ci=ci,
                    overlay_synth=overlay_synth,
                    peers=peers,
                    weights=weights.to_dict() if weights is not None else None,
                    atm_band=atm_band,
                    max_expiries=max_expiries,
                    feature_mode=feature_mode,
                )

            if hasattr(self, "_warm"):
                try:
                    self._warm.enqueue("term", payload, _builder)
                except Exception:
                    pass

            data = compute_or_load("term", payload, _builder)

            atm_curve = data.get("atm_curve") if data else None
            if atm_curve is None or atm_curve.empty:
                ax.set_title("No data")
                self.last_fit_info = {
                    "ticker": target,
                    "asof": asof,
                    "fit_by_expiry": {},
                    "weight_info": _weight_info(target, asof, weight_mode, weights),
                    "status_events": [
                        _status_event("data", "error", "No term-structure data available for the selected ticker/date.")
                    ],
                }
                return

            # Prepare data for parameter summary tab
            try:
                fit_map: dict = {}
                for _, row in atm_curve.iterrows():
                    T_val = float(row.get("T", np.nan))
                    if not np.isfinite(T_val):
                        continue
                    entry: dict = {}
                    exp = row.get("expiry")
                    if pd.notna(exp):
                        entry["expiry"] = str(exp)
                    sens = {k: row[k] for k in ("atm_vol", "skew", "curv") if k in row and pd.notna(row[k])}
                    if sens:
                        entry["sens"] = sens
                    if entry:
                        fit_map[T_val] = entry
                self.last_fit_info = {
                    "ticker": target,
                    "asof": asof,
                    "fit_by_expiry": fit_map,
                    "weight_info": _weight_info(target, asof, weight_mode, weights),
                    "status_events": self._term_status_events(target, asof, data, weights),
                }
            except Exception:
                self.last_fit_info = None

            self._plot_term(
                ax,
                data,
                target,
                asof,
                x_units,
                ci,
                overlay_peers=overlay_peers,
                overlay_synth=overlay_synth,
                show_term_fit=show_term_fit,
            )
            return

        # --- Relative Weight Matrix ---
        elif pid == "corr_matrix":
            self._plot_corr_matrix(ax, target, peers, asof, pillars, weight_mode, atm_band)
            return

        # --- Peer Composite Surface ---
        elif pid == "synthetic_surface":
            self._clear_correlation_colorbar(ax)
            self._plot_synth_surface(ax, target, peers, asof, T_days, weight_mode)
            return

        # --- RV Heatmap ---
        elif plot_type.startswith("RV Heatmap"):
            self._clear_correlation_colorbar(ax)
            self._plot_rv_heatmap(ax, target, peers, asof, weight_mode, max_expiries)
            return

        else:
            ax.text(0.5, 0.5, f"Unknown plot: {plot_type}", ha="center", va="center")

    # -------------------- event handlers --------------------
    def _on_click(self, event):
        if not self.is_smile_active() or event.inaxes is None:
            return
        if event.inaxes is not self._smile_ctx["ax"]:
            return

        if event.button == 1:
            self.next_expiry()
        elif event.button in (3, 2):
            self.prev_expiry()

    # -------------------- weights --------------------
    def _weights_from_ui_or_matrix(self, target: str, peers: list[str], weight_mode: str, asof=None, pillars=None):
        """Resolve peer weights via analysis layer."""
        weights = resolve_peer_weights(
            target,
            peers,
            weight_mode,
            asof=asof,
            pillars=pillars,
            settings=getattr(self, "last_settings", {}),
            last_corr_df=self.last_corr_df,
            last_corr_meta=self.last_corr_meta,
        )
        warning = getattr(weights, "attrs", {}).get("weight_warning")
        try:
            from analysis.views.feature_health import build_feature_construction_result

            settings = getattr(self, "last_settings", {}) or {}
            result = build_feature_construction_result(
                target=target,
                peers=peers,
                asof=asof,
                weight_mode=weight_mode,
                atm_band=settings.get("atm_band"),
                max_expiries=settings.get("max_expiries"),
                mny_bins=settings.get("mny_bins"),
                tenors=settings.get("surface_tenors") or settings.get("pillars"),
                surface_source=settings.get("surface_source", "fit"),
                surface_model=settings.get("model", "svi"),
            )
            weights.attrs["feature_health"] = result.feature_health
        except Exception:
            pass
        if warning:
            self.last_weight_warning = warning
            self.last_description = warning
        else:
            self.last_weight_warning = None
        return weights

    # -------------------- specific plotters --------------------
    def _plot_smile(self, ax, df, target, asof, model, T_days, ci, overlay_synth, peers, weight_mode):
        dfe = df.copy()
        S = float(dfe["S"].median())
        K = dfe["K"].to_numpy(float)
        IV = dfe["sigma"].to_numpy(float)
        cp = dfe["call_put"].to_numpy() if "call_put" in dfe.columns else None
        smile_grid = _smile_grid_from_settings(getattr(self, "last_settings", {}))
        smile_mny_range = (smile_grid[0], smile_grid[1])
        K_plot, IV_plot, cp_plot, excluded_quotes = _filter_smile_quotes(S, K, IV, cp, smile_mny_range)
        T_used = float(dfe["T"].median())

        m_grid = np.linspace(smile_grid[0], smile_grid[1], smile_grid[2])
        K_grid = m_grid * S
        svi_params, svi_quality = fit_valid_model_result("svi", S, K_plot, T_used, IV_plot)
        sabr_params, sabr_quality = fit_valid_model_result("sabr", S, K_plot, T_used, IV_plot)
        tps_params, tps_quality = fit_valid_model_result("tps", S, K_plot, T_used, IV_plot)
        quality_map = {"svi": svi_quality, "sabr": sabr_quality, "tps": tps_quality}
        fit_params = {"svi": svi_params, "sabr": sabr_params, "tps": tps_params}.get(model, {})
        bands = None
        if fit_params and ci and ci > 0:
            if model == "svi":
                bands = svi_confidence_bands(S, K_plot, T_used, IV_plot, K_grid, level=float(ci))
            elif model == "sabr":
                bands = sabr_confidence_bands(S, K_plot, T_used, IV_plot, K_grid, level=float(ci))
            else:
                bands = tps_confidence_bands(S, K_plot, T_used, IV_plot, K_grid, level=float(ci))

        if fit_params:
            info = fit_and_plot_smile(
                ax,
                S=S,
                K=K_plot,
                T=T_used,
                iv=IV_plot,
                model=model,
                params=fit_params,
                bands=bands,
                moneyness_grid=smile_grid,
                show_points=True,
                call_put=cp_plot,
                enable_toggles=False,
            )
            rmse_text = _fit_quality_text(info.get("rmse", np.nan), quality_map.get(model))
            title = f"{target}  {asof}  T≈{T_used:.3f}y  RMSE={rmse_text}"
        else:
            ax.scatter(K_plot / S, IV_plot, s=20, alpha=0.85, label="Observed")
            ax.axvline(1.0, color="grey", lw=1, ls="--", alpha=0.85, label="_nolegend_")
            ax.set_xlabel("Moneyness K/S")
            ax.set_ylabel("Implied Vol")
            title = f"{target}  {asof}  T≈{T_used:.3f}y  {model.upper()} rejected"
        status_events = [
            _status_event(
                "data",
                "info",
                f"Current smile has {int(np.isfinite(IV).sum())} finite IV quotes; "
                f"{len(IV_plot)} are used after smile filtering.",
                n=int(np.isfinite(IV_plot).sum()),
            )
        ]
        if excluded_quotes:
            status_events.append(
                _status_event(
                    "data_filter",
                    "warning",
                    f"{excluded_quotes} quotes outside "
                    f"{smile_mny_range[0]:g}-{smile_mny_range[1]:g} K/S were excluded from the smile fit/display.",
                )
            )
        selected_quality = quality_map.get(model) or {}
        if selected_quality.get("ok") is False:
            status_events.append(
                _status_event(
                    "model_fit",
                    "rejected",
                    f"{model.upper()} rejected: {selected_quality.get('reason', 'failed quality gate')}",
                )
            )
        elif fit_params:
            try:
                rmse_val = float(fit_params.get("rmse", np.nan))
            except Exception:
                rmse_val = np.nan
            if np.isfinite(rmse_val) and rmse_val > SMILE_RMSE_WARN:
                status_events.append(
                    _status_event(
                        "model_fit",
                        "warning",
                        f"{model.upper()} RMSE {rmse_val:.4f} exceeds warning threshold {SMILE_RMSE_WARN:.2f}.",
                        rmse=rmse_val,
                    )
                )

        # compute and log parameters for both SVI, SABR and sensitivities
        try:
            expiry_dt = None
            if "expiry" in dfe.columns and not dfe["expiry"].empty:
                expiry_dt = dfe["expiry"].iloc[0]

            dfe2 = dfe.copy()
            dfe2["moneyness"] = dfe2["K"].astype(float) / float(S)
            sens = fit_smile_get_atm(dfe2, model="auto")
            sens_params = {k: sens[k] for k in ("atm_vol", "skew", "curv") if k in sens}

            for model_key, params in (("svi", svi_params), ("sabr", sabr_params), ("tps", tps_params)):
                if params:
                    append_params(
                        asof_date=asof,
                        ticker=target,
                        expiry=str(expiry_dt) if expiry_dt is not None else None,
                        model=model_key,
                        params=params,
                        meta={"rmse": params.get("rmse")},
                    )
            append_params(
                asof_date=asof,
                ticker=target,
                expiry=str(expiry_dt) if expiry_dt is not None else None,
                model="sens",
                params=sens_params,
            )

            fit_map = {
                float(T_used): {
                    "expiry": str(expiry_dt.date()) if expiry_dt is not None else None,
                    "svi": svi_params,
                    "sabr": sabr_params,
                    "tps": tps_params,
                    "sens": sens_params,
                    "quality": quality_map,
                    "fallback": {"svi": "none", "sabr": "none", "tps": "none"},
                }
            }
            self.last_fit_info = {
                "ticker": target,
                "asof": asof,
                "fit_by_expiry": fit_map,
                "status_events": status_events,
            }
        except Exception:
            self.last_fit_info = None

        if overlay_synth and peers:
            try:
                w = self._weights_from_ui_or_matrix(
                    target,
                    peers,
                    weight_mode,
                    asof=asof,
                    pillars=self.last_corr_meta.get("pillars") if self.last_corr_meta else None,
                )
                peer_slices = {}
                for peer in peers:
                    df_peer = get_smile_slice(peer, asof, T_target_years=None, max_expiries=self._current_max_expiries)
                    if df_peer is not None and not df_peer.empty:
                        peer_slices[peer.upper()] = df_peer
                composite = build_peer_smile_composite(
                    peer_slices,
                    w,
                    model=model,
                    target_T=T_used,
                    moneyness_grid=smile_grid,
                )
                x_mny = composite.get("moneyness", np.array([]))
                y_syn = composite.get("iv", np.array([]))
                x_mny, y_syn, valid, invalid_reason = _plot_xy_mask(x_mny, y_syn, "Peer composite smile")
                if np.sum(valid) >= 2:
                    mode_lbl = weight_mode.split("_")[0] if weight_mode else ""
                    if mode_lbl == "corr":
                        mode_lbl = "relative weight matrix"
                    syn_label = f"Peer composite smile ({mode_lbl})" if mode_lbl else "Peer composite smile"
                    ax.plot(x_mny[valid], y_syn[valid], "--", linewidth=1.6, alpha=0.95, label=syn_label)
                elif invalid_reason:
                    LOGGER.warning(
                        "Peer-composite smile overlay skipped: %s skipped=%s",
                        invalid_reason,
                        composite.get("skipped", {}),
                    )
            except Exception as exc:
                LOGGER.warning("Failed to build peer-composite smile on common grid: %s", exc)

        _dedupe_legend(ax)
        ax.set_title(title)

    def _plot_synth_surface(self, ax, target, peers, asof, T_days, weight_mode):
        peers = [p for p in peers if p]
        if not peers:
            self.last_description = (
                "Peer Composite Surface needs at least one peer. It compares the target IV surface "
                "against a weighted peer basket on the common moneyness/tenor grid."
            )
            ax.text(0.5, 0.5, "Provide peers to build peer-composite surface", ha="center", va="center")
            return

        w = self._weights_from_ui_or_matrix(
            target,
            peers,
            weight_mode,
            asof=asof,
            pillars=self.last_corr_meta.get("pillars") if self.last_corr_meta else None,
        )

        try:
            tickers = list({target, *peers})
            settings = getattr(self, "last_settings", {}) or {}
            try:
                surfaces = self._get_surface_grids(
                    tickers,
                    self._current_max_expiries,
                    settings.get("mny_bins"),
                    settings.get("pillars"),
                    settings.get("model", "svi"),
                    settings.get("surface_source", "fit"),
                )
            except TypeError:
                surfaces = self._get_surface_grids(
                    tickers,
                    self._current_max_expiries,
                    settings.get("mny_bins"),
                    settings.get("pillars"),
                )

            tgt_grid = _value_for_asof(surfaces.get(target, {}), asof)
            if tgt_grid is None:
                self.last_description = (
                    f"No target surface was available for {target} on {asof}. "
                    "The peer composite cannot be compared without the target grid."
                )
                ax.text(0.5, 0.5, f"No {target} surface for {asof}", ha="center", va="center")
                ax.set_title(f"Peer Composite Surface - {target} vs peers")
                return

            peer_surfaces = {t: surfaces[t] for t in peers if t in surfaces}
            if not peer_surfaces:
                self.last_description = (
                    f"No peer surfaces were available for {asof}. "
                    "The plot requires at least one peer grid to build a weighted composite."
                )
                ax.text(0.5, 0.5, f"No peer surfaces for {asof}", ha="center", va="center")
                ax.set_title(f"Peer Composite Surface - {target} vs peers")
                return
            synth_by_date = combine_surfaces(peer_surfaces, w.to_dict())
            syn_grid = _value_for_asof(synth_by_date, asof)
            if syn_grid is None:
                used = ", ".join(peer_surfaces.keys())
                self.last_description = (
                    f"Peer surfaces exist for {used}, but no composite grid matched {asof}. "
                    "Check peer date coverage before reading the surface spread."
                )
                ax.text(0.5, 0.5, f"No peer-composite surface for {asof}", ha="center", va="center")
                ax.set_title(f"Peer Composite Surface - {target} vs peers")
                return

            tgt_grid = tgt_grid.copy()
            syn_grid = syn_grid.copy()
            common_rows = tgt_grid.index.intersection(syn_grid.index)
            common_cols = tgt_grid.columns.intersection(syn_grid.columns)
            if len(common_rows) < 2 or len(common_cols) < 2:
                self.last_description = (
                    f"Insufficient common grid for {target} vs peers on {asof}: "
                    f"target={_surface_grid_label(tgt_grid)}, composite={_surface_grid_label(syn_grid)}, "
                    f"common={len(common_rows)} moneyness x {len(common_cols)} tenor."
                )
                ax.text(
                    0.5,
                    0.5,
                    "Insufficient common surface grid\n" f"common={len(common_rows)} mny x {len(common_cols)} tenor",
                    ha="center",
                    va="center",
                )
                ax.set_title(f"Peer Composite Surface - {target} vs peers")
                return

            tgt = tgt_grid.loc[common_rows, common_cols].astype(float)
            syn = syn_grid.loc[common_rows, common_cols].astype(float)
            spread = tgt - syn

            ax.clear()
            ax.axis("off")
            mode_lbl = weight_mode.split("_")[0] if weight_mode else ""
            if mode_lbl == "corr":
                mode_lbl = "relative weight matrix"
            ax.set_title(f"{target} — IV Surface vs Peer Composite | {asof}", pad=12)

            weights_line, _ = _surface_weight_lines(w, list(peer_surfaces.keys()))
            grid_line = (
                f"Common grid: {len(common_rows)} moneyness x {len(common_cols)} tenors "
                f"(moneyness {_surface_axis_range(common_rows)}, tenor {_surface_axis_range(common_cols)} days)"
            )
            context_text = (
                f"Composite = weighted peer IV surface ({mode_lbl}); values are decimal IV.\n"
                f"{weights_line}\n"
                f"{grid_line}"
            )
            interpretation_text = (
                "Spread panel = target - peer composite. "
                "Red: target IV richer than peers. Blue: target IV cheaper than peers."
            )

            self.last_description = (
                f"Peer Composite Surface for {target} on {asof}. "
                f"Left panel is {target}'s IV surface; middle panel is the weighted peer composite "
                f"built from {', '.join(peer_surfaces.keys())}; right panel is target minus composite. "
                f"{weights_line}. {grid_line}. "
                f"Comparison is on intersected moneyness/tenor cells only, so plotted values are directly comparable. "
                f"Positive spread means {target} IV is richer than peers at that cell; negative spread means cheaper."
            )

            panels = [
                (target, tgt, "viridis"),
                ("Peer Composite", syn, "viridis"),
                ("Target - Peer Composite", spread, "coolwarm"),
            ]
            finite_iv = np.concatenate(
                [
                    tgt.to_numpy(float)[np.isfinite(tgt.to_numpy(float))],
                    syn.to_numpy(float)[np.isfinite(syn.to_numpy(float))],
                ]
            )
            vmin = float(np.nanmin(finite_iv)) if finite_iv.size else None
            vmax = float(np.nanmax(finite_iv)) if finite_iv.size else None
            spread_abs = (
                float(np.nanmax(np.abs(spread.to_numpy(float)))) if np.isfinite(spread.to_numpy(float)).any() else 0.01
            )

            ax.text(
                0.02,
                0.96,
                context_text,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                linespacing=1.25,
                bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "0.75", "alpha": 0.92},
            )
            ax.text(
                0.5,
                0.035,
                interpretation_text,
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=8,
                color="0.2",
            )
            ax.text(
                0.5,
                0.11,
                "Tenor (days)",
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=8,
                color="0.2",
            )

            panel_positions = [
                [0.06, 0.25, 0.24, 0.45],
                [0.37, 0.25, 0.24, 0.45],
                [0.69, 0.25, 0.24, 0.45],
            ]
            iv_im = None
            spread_im = None
            for i, (label, grid, cmap) in enumerate(panels):
                child = ax.inset_axes(panel_positions[i])
                if not hasattr(ax.figure, "_surface_aux_axes"):
                    ax.figure._surface_aux_axes = []
                ax.figure._surface_aux_axes.append(child)
                arr = grid.to_numpy(float)
                kwargs = {"aspect": "equal", "origin": "lower", "cmap": cmap}
                if label != "Target - Peer Composite":
                    kwargs.update(vmin=vmin, vmax=vmax)
                else:
                    kwargs.update(vmin=-spread_abs, vmax=spread_abs)
                im = child.imshow(arr, **kwargs)
                if label != "Target - Peer Composite":
                    iv_im = im
                else:
                    spread_im = im
                child.set_title(label, fontsize=10)
                if i == 0:
                    child.set_ylabel("Moneyness", fontsize=8)
                child.set_xticks(range(len(common_cols)))
                child.set_xticklabels([str(c) for c in common_cols], rotation=0, ha="center", fontsize=7)
                child.set_yticks(range(len(common_rows)))
                if i == 0:
                    child.set_yticklabels([str(r) for r in common_rows], fontsize=7)
                else:
                    child.set_yticklabels([])
                child.tick_params(axis="both", length=2, pad=2)

            if iv_im is not None:
                cax = ax.inset_axes([0.625, 0.29, 0.012, 0.34])
                cbar = ax.figure.colorbar(iv_im, cax=cax)
                cbar.set_label("IV", fontsize=8)
                cbar.ax.tick_params(labelsize=7, pad=1)
                ax.figure._surface_aux_axes.append(cax)
            if spread_im is not None:
                cax = ax.inset_axes([0.945, 0.29, 0.012, 0.34])
                cbar = ax.figure.colorbar(spread_im, cax=cax)
                cbar.set_label("Spread", fontsize=8)
                cbar.ax.tick_params(labelsize=7, pad=1)
                ax.figure._surface_aux_axes.append(cax)
        except Exception as exc:
            LOGGER.warning("Peer-composite surface plotting failed: %s", exc)
            self.last_description = (
                f"Peer-composite surface plotting failed for {target} on {asof}: {exc}. "
                "No relative-value conclusion should be drawn from this plot."
            )
            ax.text(0.5, 0.5, "Peer-composite surface plotting failed", ha="center", va="center")
            ax.set_title(f"Peer Composite Surface - {target} vs peers")

    # -------------------- RV plotters --------------------
    def _plot_rv_heatmap(self, ax, target, peers, asof, weight_mode, max_expiries):
        """Plot the surface residual heatmap (target − synthetic, z-scored per cell)."""
        peers = [p for p in peers if p]
        if not peers:
            ax.text(0.5, 0.5, "Provide peers to build RV heatmap", ha="center", va="center")
            return
        try:
            data = prepare_rv_heatmap_data(
                target=target,
                peers=peers,
                asof=asof,
                weight_mode=weight_mode,
                max_expiries=max_expiries,
            )
            residual = data.get("latest_residual")
            stability = data.get("weight_stability")
            mode_lbl = weight_mode.split("_")[0] if weight_mode else ""
            if mode_lbl == "corr":
                mode_lbl = "relative weight matrix"
            title = f"{target} surface residual vs synthetic | {asof} | {mode_lbl}"
            plot_surface_residual_heatmap(ax, residual, title=title)

            # Annotate weight stability as text below the title
            if stability is not None and not stability.empty and "rolling_corr" in stability.columns:
                unstable = stability[~stability["stable"].astype(bool)]
                if not unstable.empty:
                    note = "⚠ Low peer stability: " + ", ".join(unstable.index.tolist())
                    ax.set_xlabel(note, fontsize=7, color="tab:orange")
        except Exception as exc:
            ax.text(0.5, 0.5, f"RV heatmap failed:\n{exc}", ha="center", va="center", wrap=True)
            ax.set_title(f"RV Heatmap - {target}")

    # -------------------- smile click-through renderer --------------------
    def _render_smile_at_index(self):
        if not self._smile_ctx:
            return
        ax = self._smile_ctx["ax"]
        T_arr = self._smile_ctx["T_arr"]
        K_arr = self._smile_ctx["K_arr"]
        sigma_arr = self._smile_ctx["sigma_arr"]
        S_arr = self._smile_ctx["S_arr"]
        Ts = self._smile_ctx["Ts"]
        i = int(np.clip(self._smile_ctx["idx"], 0, len(Ts) - 1))
        self._smile_ctx["idx"] = i

        settings = self._smile_ctx["settings"]
        target = settings["target"]
        asof = settings["asof"]
        model = settings["model"]
        ci = settings["ci"]
        wm = settings.get("weight_mode")
        mode_lbl = wm.split("_")[0] if wm else ""
        if mode_lbl == "corr":
            mode_lbl = "relative weight matrix"

        T_sel = float(Ts[i])
        # pick nearest slice to T_sel
        j = int(np.nanargmin(np.abs(T_arr - T_sel)))
        T0 = float(T_arr[j])
        mask = np.isclose(T_arr, T0)
        if not np.any(mask):
            tol = 1e-6
            mask = (T_arr >= T0 - tol) & (T_arr <= T0 + tol)
        if not np.any(mask):
            ax.clear()
            self._clear_child_axes(ax)
            self._clear_correlation_colorbar(ax)
            ax.set_title("No data")
            if self.canvas is not None:
                self.canvas.draw_idle()
            return

        ax.clear()
        self._clear_child_axes(ax)
        self._clear_correlation_colorbar(ax)
        S = float(np.nanmedian(S_arr[mask]))
        K = K_arr[mask]
        IV = sigma_arr[mask]
        cp_arr = self._smile_ctx.get("cp_arr")
        cp = cp_arr[mask] if cp_arr is not None else None
        smile_grid = _smile_grid_from_settings(settings)
        smile_mny_range = (smile_grid[0], smile_grid[1])
        K_plot, IV_plot, cp_plot, excluded_quotes = _filter_smile_quotes(S, K, IV, cp, smile_mny_range)

        fit_map = self._smile_ctx.get("fit_by_expiry", {})
        pre = fit_map.get(T0)
        pre_params = pre.get(model) if isinstance(pre, dict) else None
        quality_meta = (pre.get("quality", {}).get(model) if isinstance(pre, dict) else None) or {}
        if not pre_params:
            pre_params, quality_meta = fit_valid_model_result(model, S, K_plot, T0, IV_plot)
            if isinstance(pre, dict):
                pre.setdefault("quality", {})[model] = quality_meta
                pre.setdefault("fallback", {})[model] = "none"
                pre[model] = pre_params
        bands = None
        if pre_params and ci and ci > 0:
            m_grid = np.linspace(smile_grid[0], smile_grid[1], smile_grid[2])
            K_grid = m_grid * S
            if model == "svi":
                bands = svi_confidence_bands(S, K_plot, T0, IV_plot, K_grid, level=float(ci))
            elif model == "sabr":
                bands = sabr_confidence_bands(S, K_plot, T0, IV_plot, K_grid, level=float(ci))
            else:
                bands = tps_confidence_bands(S, K_plot, T0, IV_plot, K_grid, level=float(ci))
        rmse_text = "rejected"
        if pre_params:
            info = fit_and_plot_smile(
                ax,
                S=S,
                K=K_plot,
                T=T0,
                iv=IV_plot,
                model=model,
                params=pre_params,
                bands=bands,
                moneyness_grid=smile_grid,
                show_points=True,
                call_put=cp_plot,
                label=f"{target} {model.upper()}",
                enable_toggles=False,
            )
            rmse = info.get("rmse", np.nan)
            rmse_text = _fit_quality_text(rmse, quality_meta)
        else:
            ax.scatter(K_plot / S, IV_plot, s=20, alpha=0.85, label="Observed")
            ax.axvline(1.0, color="grey", lw=1, ls="--", alpha=0.85, label="_nolegend_")
            ax.set_xlabel("Moneyness K/S")
            ax.set_ylabel("Implied Vol")
            ax.set_title(f"{target}  {asof}  T≈{T0:.3f}y  {model.upper()} rejected")

        status_events = list(self._smile_ctx.get("status_events") or [])
        current_dte = int(round(T0 * 365.25))
        status_events.append(
            _status_event(
                "data",
                "info",
                f"Current expiry has {int(np.isfinite(IV).sum())} finite IV quotes; "
                f"{len(IV_plot)} are used after smile filtering.",
                dte=current_dte,
                n=int(np.isfinite(IV_plot).sum()),
            )
        )
        if excluded_quotes:
            status_events.append(
                _status_event(
                    "data_filter",
                    "warning",
                    f"{excluded_quotes} quotes outside "
                    f"{smile_mny_range[0]:g}-{smile_mny_range[1]:g} K/S were excluded from the smile fit/display.",
                    dte=current_dte,
                )
            )
        if quality_meta and quality_meta.get("ok") is False:
            status_events.append(
                _status_event(
                    "model_fit",
                    "rejected",
                    f"{model.upper()} rejected: {quality_meta.get('reason', 'failed quality gate')}",
                    dte=current_dte,
                    rmse=quality_meta.get("rmse"),
                    n=quality_meta.get("n"),
                )
            )
        elif pre_params:
            try:
                rmse_val = float(pre_params.get("rmse", np.nan))
            except Exception:
                rmse_val = np.nan
            if np.isfinite(rmse_val) and rmse_val > SMILE_RMSE_WARN:
                status_events.append(
                    _status_event(
                        "model_fit",
                        "warning",
                        f"{model.upper()} RMSE {rmse_val:.4f} exceeds warning threshold {SMILE_RMSE_WARN:.2f}.",
                        dte=current_dte,
                        rmse=rmse_val,
                        n=pre_params.get("n"),
                    )
                )
        self.last_fit_info = {
            "ticker": target,
            "asof": asof,
            "fit_by_expiry": fit_map,
            "weight_info": _weight_info(target, asof, settings.get("weight_mode", ""), self._smile_ctx.get("weights")),
            "status_events": status_events,
        }

        # overlay: peer-composite smile at this T
        weights = self._smile_ctx.get("weights")
        _weights_summary(weights)
        if settings.get("overlay_synth"):
            try:
                peer_slices_for_composite = self._smile_ctx.get("peer_slices") or {}
                if not peer_slices_for_composite:
                    for peer in settings.get("peers") or []:
                        df_peer = get_smile_slice(
                            peer, asof, T_target_years=None, max_expiries=self._current_max_expiries
                        )
                        if df_peer is not None and not df_peer.empty:
                            peer_slices_for_composite[peer.upper()] = df_peer
                    self._smile_ctx["peer_slices"] = peer_slices_for_composite
                composite = build_peer_smile_composite(
                    peer_slices_for_composite,
                    weights,
                    model=model,
                    target_T=T0,
                    moneyness_grid=smile_grid,
                )
                x_mny = composite.get("moneyness", np.array([]))
                y_syn = composite.get("iv", np.array([]))
                x_mny, y_syn, valid, invalid_reason = _plot_xy_mask(x_mny, y_syn, "Peer composite smile")
                if np.sum(valid) >= 2:
                    syn_label = f"Peer composite smile ({mode_lbl})" if mode_lbl else "Peer composite smile"
                    ax.plot(
                        x_mny[valid],
                        y_syn[valid],
                        linestyle="--",
                        linewidth=1.5,
                        alpha=0.9,
                        label=syn_label,
                    )
                elif invalid_reason:
                    skipped = composite.get("skipped", {})
                    LOGGER.warning("Peer-composite smile overlay skipped: %s skipped=%s", invalid_reason, skipped)
                    status_events.append(
                        _status_event(
                            "peer_composite",
                            "warning",
                            f"Peer composite smile not plotted: {invalid_reason}.",
                            dte=current_dte,
                        )
                    )
            except Exception as exc:
                LOGGER.warning("Failed to build peer-composite smile on common grid: %s", exc)
                status_events.append(
                    _status_event(
                        "peer_composite",
                        "warning",
                        f"Peer composite smile not plotted: {exc}",
                        dte=current_dte,
                    )
                )

        peer_slices = self._smile_ctx.get("peer_slices") or {}
        if settings.get("overlay_peers") and peer_slices:
            for p, d in peer_slices.items():
                T_p = d["T_arr"]
                K_p = d["K_arr"]
                sigma_p = d["sigma_arr"]
                S_p = d["S_arr"]
                if T_p.size == 0:
                    continue
                jp = int(np.nanargmin(np.abs(T_p - T0)))
                T0p = float(T_p[jp])
                maskp = np.isclose(T_p, T0p)
                if not np.any(maskp):
                    tol = 1e-6
                    maskp = (T_p >= T0p - tol) & (T_p <= T0p + tol)
                if not np.any(maskp):
                    continue
                Sp = float(np.nanmedian(S_p[maskp]))
                Kp = K_p[maskp]
                IVp = sigma_p[maskp]
                Kp_plot, IVp_plot, _, _peer_excluded = _filter_smile_quotes(Sp, Kp, IVp, mny_range=smile_mny_range)
                p_params = fit_valid_model_params(model, Sp, Kp_plot, T0p, IVp_plot)
                if not p_params:
                    continue
                fit_and_plot_smile(
                    ax,
                    S=Sp,
                    K=Kp_plot,
                    T=T0p,
                    iv=IVp_plot,
                    model=model,
                    params=p_params,
                    moneyness_grid=smile_grid,
                    show_points=False,
                    label=p,
                    line_kwargs={"alpha": 0.7},
                    enable_toggles=False,
                )

        _dedupe_legend(ax)
        days = int(round(T0 * 365.25))

        # Resolve expiry date and ATM stats for title
        _sens: dict = {}
        _expiry_str = ""
        try:
            _entry = fit_map.get(T0) or {}
            if not _entry:
                for _tk, _tv in fit_map.items():
                    if abs(_tk - T0) < 1e-4:
                        _entry = _tv or {}
                        break
            _sens = _entry.get("sens") or {}
            _raw_exp = _entry.get("expiry") or ""
            if _raw_exp:
                # Keep only date portion (strip time if present)
                _expiry_str = str(_raw_exp).split("T")[0].split(" ")[0]
        except Exception:
            pass

        _atm_part = f"  ATM {float(_sens['atm_vol']):.1%}" if _sens.get("atm_vol") is not None else ""
        _skew_part = f"  Skew {float(_sens['skew']):+.3f}" if _sens.get("skew") is not None else ""

        _title_exp = f"expires {_expiry_str}" if _expiry_str else f"{days}d to expiry"
        _total_exp = len(Ts)
        _exp_num = i + 1

        ax.set_title(
            f"{target} · {_title_exp} ({days}d) [{_exp_num}/{_total_exp}] · {model.upper()} RMSE {rmse_text}"
            f"{_atm_part}{_skew_part}"
        )

        # Update plain-English description for the browser bar
        _feature_mode = settings.get("feature_mode", "iv_atm")
        _weight_method = settings.get("weight_method", "corr")
        _base_expl = get_explanation(
            "smile",
            feature_mode=_feature_mode,
            weight_method=_weight_method,
            overlay_synth=bool(settings.get("overlay_synth")),
            overlay_peers=bool(settings.get("overlay_peers")),
        )
        self.last_description = (
            f"{target} · expiry {_expiry_str or f'{days}d'} [{_exp_num}/{_total_exp}] | {asof}"
            + (f"  ATM {float(_sens['atm_vol']):.1%}" if _sens.get("atm_vol") is not None else "")
            + (f"  Skew {float(_sens['skew']):+.3f}" if _sens.get("skew") is not None else "")
            + f"  RMSE {rmse_text}"
            + f"\n{_base_expl}"
        )

        # Ensure canvas and figure are valid before drawing
        if self.canvas is not None and ax.figure is not None:
            self.canvas.draw_idle()

    # -------------------- smile state helpers --------------------
    def is_smile_active(self) -> bool:
        return (
            self._current_plot_type is not None
            and plot_id(self._current_plot_type) == "smile"
            and self._smile_ctx is not None
        )

    def next_expiry(self):
        if not self.is_smile_active():
            return
        Ts = self._smile_ctx["Ts"]
        self._smile_ctx["idx"] = min(self._smile_ctx["idx"] + 1, len(Ts) - 1)
        self._render_smile_at_index()

    def prev_expiry(self):
        if not self.is_smile_active():
            return
        self._smile_ctx["Ts"]
        self._smile_ctx["idx"] = max(self._smile_ctx["idx"] - 1, 0)
        self._render_smile_at_index()

    # -------------------- term structure --------------------

    def _plot_term(
        self,
        ax,
        data,
        target,
        asof,
        x_units,
        ci,
        *,
        overlay_peers: bool = False,
        overlay_synth: bool = True,
        show_term_fit: bool = False,
    ):
        """Plot precomputed ATM term structure and optional synthetic overlay."""
        atm_curve = data.get("atm_curve")
        n_exp = len(atm_curve)
        title = f"ATM Term Structure: {target} vs Peer Composite"
        peer_curves = data.get("peer_curves") or {}
        weights = data.get("weights")
        synth_curve = data.get("synth_curve") if overlay_synth else None
        term_warnings = data.get("term_warnings") or []
        alignment_status = data.get("alignment_status", "")
        data.get("composite_status", "")
        if weights is not None and len(weights) and not overlay_synth:
            term_warnings = list(term_warnings) + [
                "Weighted peer composite is computed but hidden because peer composite overlay is off."
            ]

        # Build plain-English description
        try:
            _vols = atm_curve["atm_vol"].dropna() if "atm_vol" in atm_curve.columns else pd.Series(dtype=float)
            if len(_vols) >= 2:
                _near, _far = float(_vols.iloc[0]), float(_vols.iloc[-1])
                if _far > _near * 1.02:
                    _shape = "upward-sloping (contango) — near-term vol is lower than longer-dated vol"
                elif _near > _far * 1.02:
                    _shape = (
                        "downward-sloping (backwardation) — near-term stress is elevated relative to longer maturities"
                    )
                else:
                    _shape = "roughly flat"
                _desc_stats = f"  Near-term ATM: {_near:.1%} | Far-term ATM: {_far:.1%}."
            else:
                _shape = "unknown"
                _desc_stats = ""
        except Exception:
            _shape = "unknown"
            _desc_stats = ""

        _last = getattr(self, "last_settings", {}) or {}
        _term_expl = get_explanation(
            "term",
            feature_mode=_last.get("feature_mode", "iv_atm"),
            weight_method=_last.get("weight_method", "corr"),
            overlay_synth=overlay_synth,
            overlay_peers=overlay_peers,
        )
        self.last_description = (
            f"{target} ATM term structure | {asof} | {n_exp} aligned expiries | {_shape}{_desc_stats}"
            + (f"  Individual peers shown: {', '.join(peer_curves.keys())}." if peer_curves and overlay_peers else "")
            + f"\n{_term_expl}"
        )

        if peer_curves or (synth_curve is not None and not synth_curve.empty):
            plot_term_structure_comparison(
                ax,
                atm_curve,
                peer_curves=peer_curves if overlay_peers else {},
                synth_curve=synth_curve,
                weights=weights,
                x_units=x_units,
                fit=show_term_fit,
                show_ci=bool(ci and ci > 0 and {"atm_lo", "atm_hi"}.issubset(atm_curve.columns)),
                title=title,
                warning=" ".join(term_warnings) if term_warnings else None,
                alignment_status=alignment_status,
            )
            return

        plot_atm_term_structure(
            ax,
            atm_curve,
            x_units=x_units,
            fit=show_term_fit,
            show_ci=bool(ci and ci > 0 and {"atm_lo", "atm_hi"}.issubset(atm_curve.columns)),
        )
        ax.set_title(title)

    def _term_status_events(self, target: str, asof: str, data: dict, weights) -> list[dict]:
        atm_curve = data.get("atm_curve")
        n_exp = int(len(atm_curve)) if atm_curve is not None else 0
        events = [_status_event("data", "info", f"Term structure contains {n_exp} target expiries.", n=n_exp)]
        alignment_status = data.get("alignment_status", "")
        composite_status = data.get("composite_status", "")
        if alignment_status:
            events.append(_status_event("expiry_alignment", "info", f"Peer expiry alignment: {alignment_status}."))
        if composite_status and composite_status not in {"aligned_weighted", "not_requested"}:
            events.append(_status_event("peer_composite", "warning", f"Peer composite status: {composite_status}."))
        elif composite_status:
            events.append(_status_event("peer_composite", "info", f"Peer composite status: {composite_status}."))
        for warning in data.get("term_warnings") or []:
            events.append(_status_event("warning", "warning", str(warning)))
        weight_warning = getattr(weights, "attrs", {}).get("weight_warning", "") if weights is not None else ""
        if weight_warning:
            events.append(_status_event("weights", "warning", weight_warning, fallback="equal"))
        return events

    @staticmethod
    def _format_weight_summary(weights) -> str:
        try:
            s = pd.Series(weights, dtype=float).dropna()
            if s.empty:
                return ""
            s = s.reindex(s.abs().sort_values(ascending=False).index)
            return ", ".join(f"{idx} {val:.0%}" for idx, val in s.head(5).items())
        except Exception:
            return ""

    # -------------------- correlation matrix --------------------
    def _plot_corr_matrix(
        self,
        ax,
        target,
        peers,
        asof,
        pillars,
        weight_mode,
        atm_band,
    ):
        tickers = [t for t in [target] + peers if t]
        if not tickers:
            ax.set_title("No tickers")
            return

        matrix_weight_mode = weight_mode
        if isinstance(weight_mode, str) and weight_mode.endswith("_iv_atm"):
            # The matrix is an expiry-rank term-structure diagnostic.  Using
            # fixed ATM pillars here makes the annotations look like only a few
            # expiries were considered even when the view loaded many expiries.
            matrix_weight_mode = weight_mode[: -len("_iv_atm")] + "_iv_atm_ranks"

        settings = getattr(self, "last_settings", {})
        weight_power = settings.get("weight_power", DEFAULT_WEIGHT_POWER)
        clip_negative = settings.get("clip_negative", DEFAULT_CLIP_NEGATIVE_WEIGHTS)
        mny_bins = settings.get("mny_bins")
        surface_tenors = settings.get("surface_tenors") or settings.get("pillars")
        surface_source = settings.get("surface_source", "fit")
        surface_model = settings.get("model", "svi")

        max_exp = self._current_max_expiries or DEFAULT_MAX_EXPIRIES

        payload = {
            "tickers": sorted(tickers),
            "asof": pd.to_datetime(asof).floor("min").isoformat(),
            "atm_band": atm_band,
            "max_expiries": max_exp,
            "weight_mode": matrix_weight_mode,
            "mny_bins": str(mny_bins),
            "tenors": str(surface_tenors),
            "surface_source": surface_source,
            "surface_model": surface_model,
        }

        def _builder():
            return corr_by_expiry_rank(
                get_slice=self.get_smile_slice,
                tickers=tickers,
                asof=asof,
                max_expiries=max_exp,
                atm_band=atm_band,
            )

        if hasattr(self, "_warm"):
            try:
                self._warm.enqueue("corr", payload, _builder)
            except Exception:
                pass

        view = prepare_correlation_view(
            get_smile_slice=self.get_smile_slice,
            tickers=tickers,
            asof=asof,
            atm_band=atm_band,
            show_values=True,
            clip_negative=clip_negative,
            weight_power=weight_power,
            target=target,
            peers=peers,
            max_expiries=max_exp,
            weight_mode=matrix_weight_mode,
            mny_bins=mny_bins,
            tenors=surface_tenors,
            surface_source=surface_source,
            surface_model=surface_model,
        )
        plot_correlation_details(
            ax,
            view.corr_df,
            weights=view.weights,
            show_values=True,
            view_data=view,
        )

        # cache for other plots
        self.last_corr_df = view.corr_df
        self.last_atm_df = view.atm_df
        self.last_corr_meta = {
            "asof": asof,
            "tickers": list(tickers),
            "pillars": list(pillars or []),
            "weight_mode": matrix_weight_mode,
            "weight_power": weight_power,
            "clip_negative": clip_negative,
        }
        self.last_fit_info = {
            "ticker": target,
            "asof": asof,
            "fit_by_expiry": {},
            "weight_info": _weight_info(target, asof, matrix_weight_mode, view.weights),
            "feature_health": view.context.get("feature_health", {}),
        }

        # Plain-English description
        feature_label = weight_mode.replace("_", " ") if weight_mode else "IV"
        peer_list = ", ".join(peers) if peers else "none"
        has_weights = bool(view.weights is not None and not view.weights.dropna().empty)
        if has_weights:
            right_panel_text = "Right panel shows peer-composite weights."
        else:
            right_panel_text = "No side-panel diagnostics are available for this view."
        self.last_description = (
            f"IV similarity matrix for {target} vs peers ({peer_list}) on {asof}, "
            f"computed using {feature_label} features. "
            f"Brighter squares = more correlated vol dynamics. "
            f"{right_panel_text}"
        )
