# display/gui/gui_plot_manager.py
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

# --- Ensure project root on path so local packages resolve first ---
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Plot helpers
from display.plotting.correlation_detail_plot import (
    plot_correlation_details,
)
from display.plotting.smile_plot import fit_and_plot_smile
from display.plotting.term_plot import (
    plot_atm_term_structure,
    plot_term_structure_comparison,
)
from display.plotting.weights_plot import plot_weights
from display.plotting.rv_plots import (
    plot_surface_residual_heatmap,
    plot_skew_spread,
    plot_rv_signals_table,
)

# Surfaces & synthetic construction
from analysis.syntheticETFBuilder import build_surface_grids, combine_surfaces

# Data/analysis utilities
from analysis.analysis_pipeline import (
    get_smile_slice, prepare_smile_data, prepare_term_data,
    prepare_rv_heatmap_data,
)
from analysis.compute_or_load import compute_or_load
from display.gui.gui_input import plot_id


from analysis.cache_io import  WarmupWorker
from analysis.correlation_view import corr_by_expiry_rank, prepare_correlation_view
from analysis.settings import (
    DEFAULT_ATM_BAND,
    DEFAULT_CLIP_NEGATIVE_WEIGHTS,
    DEFAULT_FEATURE_MODE,
    DEFAULT_MAX_EXPIRIES,
    DEFAULT_MONEYNESS_BINS,
    DEFAULT_WEIGHT_METHOD,
    DEFAULT_WEIGHT_POWER,
)
from analysis.weight_view import resolve_peer_weights
from analysis.explanations import get_explanation


from analysis.model_params_logger import append_params
from analysis.pillars import _fit_smile_get_atm
from volModel.sviFit import fit_svi_slice
from volModel.sabrFit import fit_sabr_slice
from volModel.polyFit import fit_tps_slice
from analysis.confidence_bands import (
    generate_term_structure_confidence_bands,
    svi_confidence_bands,
    sabr_confidence_bands,
    tps_confidence_bands,
    
)

# ---------------------------------------------------------------------------
# Small local helpers
# ---------------------------------------------------------------------------
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

        # animation state
        self._animation: FuncAnimation | None = None
        self._animation_paused = False
        self._animation_speed = 120  # ms between frames

        # preserve last settings for weight computation context
        self.last_settings: dict = {}
        # store latest fit parameters for UI parameter tab
        self.last_fit_info: dict | None = None
        # plain-English description of the current plot (read by the browser)
        self.last_description: str = ""

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
            if hasattr(ax.figure, "_surface_aux_axes"):
                for aux in list(getattr(ax.figure, "_surface_aux_axes")):
                    try:
                        aux.remove()
                    except Exception:
                        pass
                delattr(ax.figure, "_surface_aux_axes")
            for child in list(getattr(ax, "child_axes", [])):
                child.remove()
        except Exception:
            pass

    def invalidate_surface_cache(self):
        """Clear any cached surface grids."""
        self._surface_cache.clear()

    def _get_surface_grids(self, tickers, max_expiries, mny_bins=None):
        """Return surface grids for ``tickers`` using cache if available."""
        bins_key = tuple(tuple(float(x) for x in b) for b in (mny_bins or DEFAULT_MONEYNESS_BINS))
        key = (tuple(sorted(set(tickers))), int(max_expiries), bins_key)
        if key not in self._surface_cache:
            payload = {"tickers": list(key[0]), "max_expiries": key[1], "mny_bins": bins_key}

            def _builder():
                return build_surface_grids(
                    tickers=list(key[0]),
                    tenors=None,
                    mny_bins=bins_key,
                    use_atm_only=False,
                    max_expiries=key[1],
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
        weight_mode = (
            "oi" if weight_method == "oi" else f"{weight_method}_{feature_mode}"
        )
        #TODO: Fix setting- marked as always true. 
        overlay_synth = settings.get("overlay_synth", True)
        overlay_peers = settings.get("overlay_peers", False)
        peers = settings["peers"]
        pillars = settings["pillars"]
        max_expiries = settings.get("max_expiries", DEFAULT_MAX_EXPIRIES)

        # invalidate surface cache if tickers or max_expiries changed
        prev = getattr(self, "last_settings", {}) or {}
        prev_tickers = set([prev.get("target", "")] + prev.get("peers", []))
        curr_tickers = set([target] + peers)
        if (
            prev_tickers != curr_tickers
            or prev.get("max_expiries") != max_expiries
            or prev.get("mny_bins") != mny_bins
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

        self.stop_animation()
        self._clear_child_axes(ax)
        ax.clear()

        # --- Smile: click-through (preload all expiries for date) ---
        if pid == "smile":
            self._clear_correlation_colorbar(ax)

            weights = None
            if peers:
                weights = self._weights_from_ui_or_matrix(
                    target, peers, weight_mode, asof=asof, pillars=self.last_corr_meta.get("pillars") if self.last_corr_meta else None
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
                "v": 2,  # bump to invalidate caches missing cp_arr / expiry
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
                return

            fit_map = data.get("fit_by_expiry", {})
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
            }
            self.last_fit_info = {
                "ticker": target,
                "asof": asof,
                "fit_by_expiry": fit_map,
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
                    sens = {
                        k: row[k]
                        for k in ("atm_vol", "skew", "curv")
                        if k in row and pd.notna(row[k])
                    }
                    if sens:
                        entry["sens"] = sens
                    if entry:
                        fit_map[T_val] = entry
                self.last_fit_info = {
                    "ticker": target,
                    "asof": asof,
                    "fit_by_expiry": fit_map,
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
            )
            return

        # --- Relative Weight Matrix ---
        elif pid == "corr_matrix":
            self._plot_corr_matrix(ax, target, peers, asof, pillars, weight_mode, atm_band)
            return

        # --- Synthetic Surface ---
        elif pid == "synthetic_surface":
            self._clear_correlation_colorbar(ax)
            self._plot_synth_surface(ax, target, peers, asof, T_days, weight_mode)
            return

        # --- ETF Weights only ---
        elif pid == "etf_weights":
            self._clear_correlation_colorbar(ax)
            if not peers:
                ax.text(0.5, 0.5, "No peers", ha="center", va="center")
                return
            weights = self._weights_from_ui_or_matrix(target, peers, weight_mode, asof=asof, pillars=pillars)
            if weights is None or weights.empty:
                ax.text(0.5, 0.5, "No weights", ha="center", va="center")
                return
            raw_scores = None
            try:
                if isinstance(self.last_corr_df, pd.DataFrame) and target in self.last_corr_df.columns:
                    raw_scores = self.last_corr_df.reindex(index=peers)[target]
            except Exception:
                raw_scores = None
            plot_weights(ax, weights, raw_scores=raw_scores)

            feature_label = weight_mode.replace("_", " ") if weight_mode else "IV correlation"
            top_peer = weights.idxmax() if not weights.empty else "—"
            self.last_description = (
                f"Portfolio weights for the synthetic {target} ETF using {feature_label} features on {asof}. "
                f"Each bar shows how much a peer contributes to the composite. "
                f"Largest contributor: {top_peer} ({float(weights[top_peer]):.1%})."
                if not weights.empty else
                f"Portfolio weights for the synthetic {target} ETF using {feature_label} features on {asof}."
            )
            return

        # --- RV Heatmap ---
        elif plot_type.startswith("RV Heatmap"):
            self._clear_correlation_colorbar(ax)
            self._plot_rv_heatmap(ax, target, peers, asof, weight_mode, max_expiries)
            return

        # --- RV Signals ---
        elif plot_type.startswith("RV Signals"):
            self._clear_correlation_colorbar(ax)
            self._plot_rv_signals(ax, target, peers, asof, weight_mode, max_expiries)
            return

        else:
            ax.text(0.5, 0.5, f"Unknown plot: {plot_type}", ha="center", va="center")

    def plot_animated(self, ax: plt.Axes, settings: dict) -> bool:
        """Try to create an animated plot. Returns True if successful, False otherwise."""
        plot_type = settings["plot_type"]
        
        # Stop any existing animation first
        self.stop_animation()
        
        try:
            pid = plot_id(plot_type)
            if pid == "smile":
                return self._create_animated_smile(ax, settings)
            elif pid == "synthetic_surface":
                return self._create_animated_surface(ax, settings)
            else:
                return False  # Animation not supported for this plot type
        except Exception as e:
            print(f"Warning: Animation creation failed: {e}")
            return False

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
        return resolve_peer_weights(
            target,
            peers,
            weight_mode,
            asof=asof,
            pillars=pillars,
            settings=getattr(self, "last_settings", {}),
            last_corr_df=self.last_corr_df,
            last_corr_meta=self.last_corr_meta,
        )

    # -------------------- specific plotters --------------------
    def _plot_smile(self, ax, df, target, asof, model, T_days, ci, overlay_synth, peers, weight_mode):
        dfe = df.copy()
        S = float(dfe["S"].median())
        K = dfe["K"].to_numpy(float)
        IV = dfe["sigma"].to_numpy(float)
        T_used = float(dfe["T"].median())

        m_grid = np.linspace(0.7, 1.3, 121)
        K_grid = m_grid * S
        svi_params = fit_svi_slice(S, K, T_used, IV)
        sabr_params = fit_sabr_slice(S, K, T_used, IV)
        tps_params = fit_tps_slice(S, K, T_used, IV)
        fit_params = {"svi": svi_params, "sabr": sabr_params, "tps": tps_params}.get(model, {})
        bands = None
        if ci and ci > 0:
            if model == "svi":
                bands = svi_confidence_bands(S, K, T_used, IV, K_grid, level=float(ci))
            elif model == "sabr":
                bands = sabr_confidence_bands(S, K, T_used, IV, K_grid, level=float(ci))
            else:
                bands = tps_confidence_bands(S, K, T_used, IV, K_grid, level=float(ci))

        cp = dfe["call_put"].to_numpy() if "call_put" in dfe.columns else None
        info = fit_and_plot_smile(
            ax,
            S=S,
            K=K,
            T=T_used,
            iv=IV,
            model=model,
            params=fit_params,
            bands=bands,
            moneyness_grid=(0.7, 1.3, 121),
            show_points=True,
            call_put=cp,
            enable_toggles=True,
        )
        title = f"{target}  {asof}  T≈{T_used:.3f}y  RMSE={info['rmse']:.4f}"

        # compute and log parameters for both SVI, SABR and sensitivities
        try:
            expiry_dt = None
            if "expiry" in dfe.columns and not dfe["expiry"].empty:
                expiry_dt = dfe["expiry"].iloc[0]

            dfe2 = dfe.copy()
            dfe2["moneyness"] = dfe2["K"].astype(float) / float(S)
            sens = _fit_smile_get_atm(dfe2, model="auto")
            sens_params = {k: sens[k] for k in ("atm_vol", "skew", "curv") if k in sens}

            append_params(
                asof_date=asof,
                ticker=target,
                expiry=str(expiry_dt) if expiry_dt is not None else None,
                model="svi",
                params=svi_params,
                meta={"rmse": svi_params.get("rmse")},
            )
            append_params(
                asof_date=asof,
                ticker=target,
                expiry=str(expiry_dt) if expiry_dt is not None else None,
                model="sabr",
                params=sabr_params,
                meta={"rmse": sabr_params.get("rmse")},
            )
            append_params(
                asof_date=asof,
                ticker=target,
                expiry=str(expiry_dt) if expiry_dt is not None else None,
                model="tps",
                params=tps_params,
                meta={"rmse": tps_params.get("rmse")},
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
                }
            }
            self.last_fit_info = {
                "ticker": target,
                "asof": asof,
                "fit_by_expiry": fit_map,
            }
        except Exception:
            self.last_fit_info = None

        if overlay_synth and peers:
            try:
                w = self._weights_from_ui_or_matrix(
                    target, peers, weight_mode, asof=asof, pillars=self.last_corr_meta.get("pillars") if self.last_corr_meta else None
                )
                tickers = list({target, *peers})
                surfaces = self._get_surface_grids(tickers, self._current_max_expiries, settings.get("mny_bins"))

                tgt_grid = _value_for_asof(surfaces.get(target, {}), asof)
                if tgt_grid is not None:
                    peer_surfaces = {t: surfaces[t] for t in peers if t in surfaces}
                    synth_by_date = combine_surfaces(peer_surfaces, w.to_dict())
                    syn_grid = _value_for_asof(synth_by_date, asof)
                    if syn_grid is not None:

                        tgt_cols = _cols_to_days(tgt_grid.columns)
                        syn_cols = _cols_to_days(syn_grid.columns)
                        i_tgt = _nearest_tenor_idx(tgt_cols, T_days)
                        i_syn = _nearest_tenor_idx(syn_cols, T_days)
                        col_syn = syn_grid.columns[i_syn]

                        x_mny = _mny_from_index_labels(tgt_grid.index)
                        y_syn = syn_grid[col_syn].astype(float).to_numpy()
                        
                        # Improved grid alignment for synthetic smile
                        if not tgt_grid.index.equals(syn_grid.index):
                            try:
                                # Try interpolation-based alignment
                                syn_mny = _mny_from_index_labels(syn_grid.index)
                                syn_iv = syn_grid[col_syn].astype(float).to_numpy()
                                
                                # Filter valid data
                                syn_valid = np.isfinite(syn_mny) & np.isfinite(syn_iv)
                                tgt_valid = np.isfinite(x_mny)
                                
                                if np.sum(syn_valid) >= 2 and np.sum(tgt_valid) >= 2:
                                    from scipy.interpolate import interp1d
                                    syn_mny_clean = syn_mny[syn_valid]
                                    syn_iv_clean = syn_iv[syn_valid]
                                    tgt_mny_clean = x_mny[tgt_valid]
                                    
                                    # Interpolate within range
                                    min_syn = np.min(syn_mny_clean)
                                    max_syn = np.max(syn_mny_clean)
                                    interp_mask = (tgt_mny_clean >= min_syn) & (tgt_mny_clean <= max_syn)
                                    
                                    if np.sum(interp_mask) >= 2:
                                        tgt_mny_interp = tgt_mny_clean[interp_mask]
                                        f_interp = interp1d(syn_mny_clean, syn_iv_clean, 
                                                          kind='linear', bounds_error=False, fill_value=np.nan)
                                        syn_iv_interp = f_interp(tgt_mny_interp)
                                        x_mny = tgt_mny_interp
                                        y_syn = syn_iv_interp
                                        
                            except (ImportError, Exception):
                                # Fallback to intersection-based alignment
                                common = tgt_grid.index.intersection(syn_grid.index)
                                if len(common) >= 3:
                                    x_mny = _mny_from_index_labels(common)
                                    y_syn = syn_grid.loc[common, col_syn].astype(float).to_numpy()

                        # Filter final valid data before plotting
                        final_valid = np.isfinite(x_mny) & np.isfinite(y_syn)
                        if np.sum(final_valid) >= 2:
                            mode_lbl = (weight_mode.split("_")[0] if weight_mode else "")
                            if mode_lbl == "corr":
                                mode_lbl = "relative weight matrix"
                            syn_label = f"Synthetic ETF smile ({mode_lbl})" if mode_lbl else "Synthetic ETF smile"
                            ax.plot(
                                x_mny[final_valid],
                                y_syn[final_valid],
                                "--",
                                linewidth=1.6,
                                alpha=0.95,
                                label=syn_label,
                            )
                        ax.legend(loc="best", fontsize=8)
            except Exception:
                pass

        ax.set_title(title)

    def _plot_synth_surface(self, ax, target, peers, asof, T_days, weight_mode):
        peers = [p for p in peers if p]
        if not peers:
            ax.text(0.5, 0.5, "Provide peers to build synthetic surface", ha="center", va="center")
            return

        w = self._weights_from_ui_or_matrix(
            target, peers, weight_mode, asof=asof, pillars=self.last_corr_meta.get("pillars") if self.last_corr_meta else None
        )

        try:
            tickers = list({target, *peers})
            surfaces = self._get_surface_grids(tickers, self._current_max_expiries, getattr(self, "last_settings", {}).get("mny_bins"))

            tgt_grid = _value_for_asof(surfaces.get(target, {}), asof)
            if tgt_grid is None:
                ax.text(0.5, 0.5, "No target surface for date", ha="center", va="center")
                ax.set_title(f"Synthetic Surface - {target} vs peers")
                return

            peer_surfaces = {t: surfaces[t] for t in peers if t in surfaces}
            synth_by_date = combine_surfaces(peer_surfaces, w.to_dict())
            syn_grid = _value_for_asof(synth_by_date, asof)
            if syn_grid is None:
                ax.text(0.5, 0.5, "No synthetic surface for date", ha="center", va="center")
                ax.set_title(f"Synthetic Surface - {target} vs peers")
                return

            tgt_grid = tgt_grid.copy()
            syn_grid = syn_grid.copy()
            common_rows = tgt_grid.index.intersection(syn_grid.index)
            common_cols = tgt_grid.columns.intersection(syn_grid.columns)
            if len(common_rows) < 2 or len(common_cols) < 2:
                ax.text(0.5, 0.5, "Insufficient common surface grid", ha="center", va="center")
                ax.set_title(f"Synthetic Surface - {target} vs peers")
                return

            tgt = tgt_grid.loc[common_rows, common_cols].astype(float)
            syn = syn_grid.loc[common_rows, common_cols].astype(float)
            spread = tgt - syn

            ax.clear()
            ax.axis("off")
            mode_lbl = (weight_mode.split("_")[0] if weight_mode else "")
            if mode_lbl == "corr":
                mode_lbl = "relative weight matrix"
            ax.set_title(f"{target} — IV Surface vs Synthetic Peer Composite | {asof}", pad=12)

            # Description
            self.last_description = (
                f"Three-panel IV surface view for {target} on {asof}. "
                f"Left: {target} implied vol surface. "
                f"Middle: synthetic peer composite (weighted via {mode_lbl}). "
                f"Right: spread (Target − Synthetic) — red = {target} priced richer than peers, "
                f"blue = {target} cheaper than peers."
            )

            panels = [
                (target, tgt, "viridis"),
                ("Synthetic", syn, "viridis"),
                ("Target - Synthetic", spread, "coolwarm"),
            ]
            finite_iv = np.concatenate([
                tgt.to_numpy(float)[np.isfinite(tgt.to_numpy(float))],
                syn.to_numpy(float)[np.isfinite(syn.to_numpy(float))],
            ])
            vmin = float(np.nanmin(finite_iv)) if finite_iv.size else None
            vmax = float(np.nanmax(finite_iv)) if finite_iv.size else None
            spread_abs = float(np.nanmax(np.abs(spread.to_numpy(float)))) if np.isfinite(spread.to_numpy(float)).any() else 0.01

            for i, (label, grid, cmap) in enumerate(panels):
                child = ax.inset_axes([0.02 + i * 0.325, 0.08, 0.30, 0.78])
                if not hasattr(ax.figure, "_surface_aux_axes"):
                    ax.figure._surface_aux_axes = []
                ax.figure._surface_aux_axes.append(child)
                arr = grid.to_numpy(float)
                kwargs = {"aspect": "auto", "origin": "lower", "cmap": cmap}
                if label != "Target - Synthetic":
                    kwargs.update(vmin=vmin, vmax=vmax)
                else:
                    kwargs.update(vmin=-spread_abs, vmax=spread_abs)
                im = child.imshow(arr, **kwargs)
                child.set_title(label, fontsize=10)
                child.set_xlabel("Tenor (days)", fontsize=8)
                if i == 0:
                    child.set_ylabel("Moneyness", fontsize=8)
                child.set_xticks(range(len(common_cols)))
                child.set_xticklabels([str(c) for c in common_cols], rotation=45, ha="right", fontsize=7)
                child.set_yticks(range(len(common_rows)))
                child.set_yticklabels([str(r) for r in common_rows], fontsize=7)
                cbar = ax.figure.colorbar(im, ax=child, fraction=0.046, pad=0.02)
                ax.figure._surface_aux_axes.append(cbar.ax)
                cbar.ax.tick_params(labelsize=7)
        except Exception:
            ax.text(0.5, 0.5, "Synthetic surface plotting failed", ha="center", va="center")
            ax.set_title(f"Synthetic Surface - {target} vs peers")

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
            mode_lbl = (weight_mode.split("_")[0] if weight_mode else "")
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
            ax.text(0.5, 0.5, f"RV heatmap failed:\n{exc}", ha="center", va="center",
                    wrap=True)
            ax.set_title(f"RV Heatmap - {target}")

    def _plot_rv_signals(self, ax, target, peers, asof, weight_mode, max_expiries):
        """Render ranked RV signals as a matplotlib table."""
        peers = [p for p in peers if p]
        if not peers:
            ax.text(0.5, 0.5, "Provide peers for RV signal computation", ha="center", va="center")
            return
        try:
            from analysis.rv_analysis import generate_rv_signals
            signals = generate_rv_signals(
                target=target,
                peers=peers,
                asof=asof,
                weight_mode=weight_mode,
                max_expiries=max_expiries,
            )
            mode_lbl = (weight_mode.split("_")[0] if weight_mode else "")
            if mode_lbl == "corr":
                mode_lbl = "relative weight matrix"
            plot_rv_signals_table(
                ax, signals,
                title=f"{target} RV signals | {asof} | {mode_lbl}",
            )
        except Exception as exc:
            ax.text(0.5, 0.5, f"RV signals failed:\n{exc}", ha="center", va="center",
                    wrap=True)
            ax.set_title(f"RV Signals - {target}")

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
            self._clear_correlation_colorbar(ax)
            ax.set_title("No data")
            if self.canvas is not None:
                self.canvas.draw_idle()
            return

        ax.clear()
        self._clear_correlation_colorbar(ax)
        S = float(np.nanmedian(S_arr[mask]))
        K = K_arr[mask]
        IV = sigma_arr[mask]
        cp_arr = self._smile_ctx.get("cp_arr")
        cp = cp_arr[mask] if cp_arr is not None else None

        fit_map = self._smile_ctx.get("fit_by_expiry", {})
        pre = fit_map.get(T0)
        pre_params = pre.get(model) if isinstance(pre, dict) else None
        if not pre_params:
            if model == "svi":
                pre_params = fit_svi_slice(S, K, T0, IV)
            elif model == "sabr":
                pre_params = fit_sabr_slice(S, K, T0, IV)
            else:
                pre_params = fit_tps_slice(S, K, T0, IV)
        bands = None
        if ci and ci > 0:
            m_grid = np.linspace(0.7, 1.3, 121)
            K_grid = m_grid * S
            if model == "svi":
                bands = svi_confidence_bands(S, K, T0, IV, K_grid, level=float(ci))
            elif model == "sabr":
                bands = sabr_confidence_bands(S, K, T0, IV, K_grid, level=float(ci))
            else:
                bands = tps_confidence_bands(S, K, T0, IV, K_grid, level=float(ci))
        info = fit_and_plot_smile(
            ax,
            S=S,
            K=K,
            T=T0,
            iv=IV,
            model=model,
            params=pre_params,
            bands=bands,
            moneyness_grid=(0.7, 1.3, 121),
            show_points=True,
            call_put=cp,
            label=f"{target} {model.upper()}",
            enable_toggles=True,
        )

        if fit_map:
            self.last_fit_info = {
                "ticker": target,
                "asof": asof,
                "fit_by_expiry": fit_map,
            }

        # overlay: synthetic smile at this T
        syn_surface = self._smile_ctx.get("syn_surface")
        tgt_surface = self._smile_ctx.get("tgt_surface")
        if settings.get("overlay_synth"):
            if syn_surface is None or tgt_surface is None:
                try:
                    weights = self._smile_ctx.get("weights")
                    tickers = [target] + (settings.get("peers") or [])
                    surfaces = self._get_surface_grids(tickers, self._current_max_expiries, settings.get("mny_bins"))
                    if tgt_surface is None and target in surfaces:
                        tgt_surface = _value_for_asof(surfaces[target], asof)
                        self._smile_ctx["tgt_surface"] = tgt_surface
                    if weights is not None:
                        peer_surfaces = {p: surfaces[p] for p in (settings.get("peers") or []) if p in surfaces}
                        synth_by_date = combine_surfaces(peer_surfaces, weights.to_dict())
                        syn_surface = _value_for_asof(synth_by_date, asof)
                        self._smile_ctx["syn_surface"] = syn_surface
                except Exception:
                    syn_surface = None
            if syn_surface is not None:
                try:
                    syn_cols_days = _cols_to_days(syn_surface.columns)
                    jx = _nearest_tenor_idx(syn_cols_days, T0 * 365.25)
                    col_syn = syn_surface.columns[jx]

                    # Extract synthetic surface data
                    syn_mny = _mny_from_index_labels(syn_surface.index)
                    syn_iv = syn_surface[col_syn].astype(float).to_numpy()

                    # Filter out NaN values
                    valid_mask = np.isfinite(syn_mny) & np.isfinite(syn_iv)
                    if np.sum(valid_mask) >= 2:
                        syn_mny_clean = syn_mny[valid_mask]
                        syn_iv_clean = syn_iv[valid_mask]

                        # If we have target surface, try to align grids
                        if tgt_surface is not None:
                            tgt_mny = _mny_from_index_labels(tgt_surface.index)
                            tgt_valid = np.isfinite(tgt_mny)

                            if np.sum(tgt_valid) >= 2:
                                tgt_mny_clean = tgt_mny[tgt_valid]

                                # Interpolate synthetic IV onto target moneyness grid
                                try:
                                    from scipy.interpolate import interp1d
                                    if len(syn_mny_clean) >= 2 and len(tgt_mny_clean) >= 2:
                                        # Only interpolate within the range of synthetic data
                                        min_syn_mny = np.min(syn_mny_clean)
                                        max_syn_mny = np.max(syn_mny_clean)

                                        # Filter target grid to interpolation range
                                        interp_mask = (tgt_mny_clean >= min_syn_mny) & (tgt_mny_clean <= max_syn_mny)
                                        if np.sum(interp_mask) >= 2:
                                            tgt_mny_interp = tgt_mny_clean[interp_mask]

                                            # Create interpolator and interpolate
                                            f_interp = interp1d(
                                                syn_mny_clean,
                                                syn_iv_clean,
                                                kind="linear",
                                                bounds_error=False,
                                                fill_value=np.nan,
                                            )
                                            syn_iv_interp = f_interp(tgt_mny_interp)

                                            # Use interpolated values
                                            x_mny = tgt_mny_interp
                                            y_syn = syn_iv_interp
                                        else:
                                            x_mny = syn_mny_clean
                                            y_syn = syn_iv_clean
                                    else:
                                        x_mny = syn_mny_clean
                                        y_syn = syn_iv_clean
                                except ImportError:
                                    # Fallback if scipy not available
                                    x_mny = syn_mny_clean
                                    y_syn = syn_iv_clean
                            else:
                                x_mny = syn_mny_clean
                                y_syn = syn_iv_clean
                        else:
                            x_mny = syn_mny_clean
                            y_syn = syn_iv_clean

                        # Plot the synthetic smile with proper alignment
                        final_valid = np.isfinite(x_mny) & np.isfinite(y_syn)
                        if np.sum(final_valid) >= 2:
                            syn_label = f"Synthetic ETF smile ({mode_lbl})" if mode_lbl else "Synthetic ETF smile"
                            ax.plot(
                                x_mny[final_valid],
                                y_syn[final_valid],
                                linestyle="--",
                                linewidth=1.5,
                                alpha=0.9,
                                label=syn_label,
                            )
                except Exception as e:
                    print(f"Warning: Failed to plot synthetic smile overlay: {e}")
                    # Fallback to simple approach
                    try:
                        x_mny = _mny_from_index_labels(syn_surface.index)
                        y_syn = syn_surface[col_syn].astype(float).to_numpy()
                        valid = np.isfinite(x_mny) & np.isfinite(y_syn)
                        if np.sum(valid) >= 2:
                            syn_label = f"Synthetic ETF smile ({mode_lbl})" if mode_lbl else "Synthetic ETF smile"
                            ax.plot(
                                x_mny[valid],
                                y_syn[valid],
                                linestyle="--",
                                linewidth=1.5,
                                alpha=0.9,
                                label=syn_label,
                            )
                    except Exception:
                        pass

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
                if model == "svi":
                    p_params = fit_svi_slice(Sp, Kp, T0p, IVp)
                elif model == "sabr":
                    p_params = fit_sabr_slice(Sp, Kp, T0p, IVp)
                else:
                    p_params = fit_tps_slice(Sp, Kp, T0p, IVp)
                fit_and_plot_smile(
                    ax,
                    S=Sp,
                    K=Kp,
                    T=T0p,
                    iv=IVp,
                    model=model,
                    params=p_params,
                    moneyness_grid=(0.7, 1.3, 121),
                    show_points=False,
                    label=p,
                    line_kwargs={"alpha": 0.7},
                )

        # Add legend only if there are labeled artists
        handles, labels = ax.get_legend_handles_labels()
        if handles and labels:
            ax.legend(loc="best", fontsize=8)
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
            f"{target}  ·  {_title_exp}  ({days}d)  [{_exp_num}/{_total_exp}]"
            f"{_atm_part}{_skew_part}  RMSE {info['rmse']:.4f}\n"
            "← Prev / Next Expiry  or click plot (left=next, right=prev)"
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
            + f"  RMSE {info['rmse']:.4f}"
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
        Ts = self._smile_ctx["Ts"]
        self._smile_ctx["idx"] = max(self._smile_ctx["idx"] - 1, 0)
        self._render_smile_at_index()

    # -------------------- term structure --------------------
    
    def _plot_term(self, ax, data, target, asof, x_units, ci, *, overlay_peers: bool = False, overlay_synth: bool = True):
        """Plot precomputed ATM term structure and optional synthetic overlay."""
        atm_curve = data.get("atm_curve")
        n_exp = len(atm_curve)
        title = f"{target} — ATM Implied Volatility Term Structure | {asof} | {n_exp} expiries"
        peer_curves = data.get("peer_curves") or {}
        weights = data.get("weights")
        synth_curve = data.get("synth_curve") if overlay_synth else None

        # Build plain-English description
        try:
            _vols = atm_curve["atm_vol"].dropna() if "atm_vol" in atm_curve.columns else pd.Series(dtype=float)
            if len(_vols) >= 2:
                _near, _far = float(_vols.iloc[0]), float(_vols.iloc[-1])
                if _far > _near * 1.02:
                    _shape = "upward-sloping (contango) — near-term vol is lower than longer-dated vol"
                elif _near > _far * 1.02:
                    _shape = "downward-sloping (backwardation) — near-term stress is elevated relative to longer maturities"
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
            f"{target} ATM term structure | {asof} | {n_exp} expiries | {_shape}{_desc_stats}"
            + (f"  Peers: {', '.join(peer_curves.keys())}." if peer_curves and overlay_peers else "")
            + f"\n{_term_expl}"
        )

        if peer_curves or (synth_curve is not None and not synth_curve.empty):
            plot_term_structure_comparison(
                ax,
                atm_curve,
                peer_curves=peer_curves if overlay_peers or peer_curves else {},
                synth_curve=synth_curve,
                weights=weights,
                x_units=x_units,
                fit=True,
                show_ci=bool(ci and ci > 0 and {"atm_lo", "atm_hi"}.issubset(atm_curve.columns)),
                title=title,
            )
            return

        plot_atm_term_structure(
            ax,
            atm_curve,
            x_units=x_units,
            fit=True,
            show_ci=bool(ci and ci > 0 and {"atm_lo", "atm_hi"}.issubset(atm_curve.columns)),
        )
        ax.set_title(title)

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

        settings = getattr(self, "last_settings", {})
        weight_power = settings.get("weight_power", DEFAULT_WEIGHT_POWER)
        clip_negative = settings.get("clip_negative", DEFAULT_CLIP_NEGATIVE_WEIGHTS)

        max_exp = self._current_max_expiries or DEFAULT_MAX_EXPIRIES

        payload = {
            "tickers": sorted(tickers),
            "asof": pd.to_datetime(asof).floor("min").isoformat(),
            "atm_band": atm_band,
            "max_expiries": max_exp,
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
            weight_mode=weight_mode,
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
            "weight_mode": weight_mode,
            "weight_power": weight_power,
            "clip_negative": clip_negative,
        }

        # Plain-English description
        feature_label = weight_mode.replace("_", " ") if weight_mode else "IV"
        peer_list = ", ".join(peers) if peers else "none"
        self.last_description = (
            f"IV similarity matrix for {target} vs peers ({peer_list}) on {asof}, "
            f"computed using {feature_label} features. "
            f"Brighter squares = more correlated vol dynamics. "
            f"Right panel shows the resulting portfolio weights for a synthetic {target} ETF."
        )

    # -------------------- synthetic ATM helper --------------------
    # -------------------- animation control --------------------
    def has_animation_support(self, plot_type: str) -> bool:
        """Check if animation is supported for the given plot type."""
        return plot_id(plot_type) in ("smile", "synthetic_surface")

    def is_animation_active(self) -> bool:
        """Check if an animation is currently active."""
        return self._animation is not None

    def stop_animation(self) -> None:
        """Stop any currently running animation."""
        if self._animation is not None:
            try:
                self._animation.event_source.stop()
                self._animation = None
                self._animation_paused = False
            except Exception as e:
                print(f"Warning: Error stopping animation: {e}")
                self._animation = None
                self._animation_paused = False

    def start_animation(self) -> None:
        """Start or resume animation."""
        if self._animation is not None:
            if self._animation_paused:
                try:
                    self._animation.resume()
                    self._animation_paused = False
                except Exception as e:
                    print(f"Warning: Error resuming animation: {e}")

    def pause_animation(self) -> None:
        """Pause animation."""
        if self._animation is not None and not self._animation_paused:
            try:
                self._animation.pause()
                self._animation_paused = True
            except Exception as e:
                print(f"Warning: Error pausing animation: {e}")

    def set_animation_speed(self, speed_ms: int) -> None:
        """Set animation speed in milliseconds between frames."""
        self._animation_speed = max(50, min(2000, speed_ms))  # Clamp between 50ms and 2000ms
        if self._animation is not None:
            try:
                self._animation.event_source.interval = self._animation_speed
            except Exception as e:
                print(f"Warning: Error setting animation speed: {e}")

    def _create_animated_smile(self, ax: plt.Axes, settings: dict) -> bool:
        target = settings["target"]
        try:
            if self._try_animate_smile_over_dates(ax, settings):
                return True
            return self._try_animate_smile_over_expiries(ax, settings)
        except Exception as e:
            print(f"Error creating animated smile: {e}")
            return False

    def _try_animate_smile_over_dates(self, ax: plt.Axes, settings: dict) -> bool:
        from analysis.analysis_pipeline import available_dates

        target = settings["target"]
        T_days = settings.get("T_days", 30)
        dates = available_dates(target)
        if len(dates) < 2:
            return False
        animation_dates = dates[-10:] if len(dates) > 10 else dates

        k_data, iv_data, valid_dates = [], [], []
        for date in animation_dates:
            df = self.get_smile_slice(target, date, T_target_years=T_days / 365.25)
            if df is None or df.empty:
                continue
            K_arr = pd.to_numeric(df["K"], errors="coerce").to_numpy(float)
            sigma_arr = pd.to_numeric(df["sigma"], errors="coerce").to_numpy(float)
            S_arr = pd.to_numeric(df["S"], errors="coerce").to_numpy(float)
            S = np.nanmedian(S_arr)
            if not np.isfinite(S) or S <= 0:
                continue
            k = K_arr / S
            valid_mask = np.isfinite(k) & np.isfinite(sigma_arr)
            if not np.any(valid_mask):
                continue
            k_clean = k[valid_mask]
            iv_clean = sigma_arr[valid_mask]
            sort_idx = np.argsort(k_clean)
            k_data.append(k_clean[sort_idx])
            iv_data.append(iv_clean[sort_idx])
            valid_dates.append(date)

        if len(valid_dates) < 2:
            return False
        return self._create_smile_animation(ax, k_data, iv_data, valid_dates, f"{target} Smile Over Time")

    def _try_animate_smile_over_expiries(self, ax: plt.Axes, settings: dict) -> bool:
        target = settings["target"]
        asof = settings["asof"]
        df = self.get_smile_slice(target, asof, T_target_years=None)
        if df is None or df.empty:
            return False

        T_arr = pd.to_numeric(df["T"], errors="coerce").to_numpy(float)
        K_arr = pd.to_numeric(df["K"], errors="coerce").to_numpy(float)
        sigma_arr = pd.to_numeric(df["sigma"], errors="coerce").to_numpy(float)
        S_arr = pd.to_numeric(df["S"], errors="coerce").to_numpy(float)

        Ts = np.sort(np.unique(T_arr[np.isfinite(T_arr)]))
        if len(Ts) < 2:
            return False

        k_data, iv_data, valid_expiries = [], [], []
        for T in Ts:
            mask = np.isclose(T_arr, T, atol=1e-6)
            if not np.any(mask):
                continue
            K_T = K_arr[mask]
            sigma_T = sigma_arr[mask]
            S_T = S_arr[mask]
            S = np.nanmedian(S_T)
            if not np.isfinite(S) or S <= 0:
                continue
            k = K_T / S
            valid_mask = np.isfinite(k) & np.isfinite(sigma_T)
            if not np.any(valid_mask):
                continue
            k_clean = k[valid_mask]
            iv_clean = sigma_T[valid_mask]
            sort_idx = np.argsort(k_clean)
            k_data.append(k_clean[sort_idx])
            iv_data.append(iv_clean[sort_idx])
            days = int(round(T * 365.25))
            valid_expiries.append(f"T={T:.3f}y ({days}d)")

        if len(valid_expiries) < 2:
            return False
        return self._create_smile_animation(ax, k_data, iv_data, valid_expiries, f"{target} Smile Over Expiries - {asof}")

    def _create_smile_animation(self, ax: plt.Axes, k_data: list, iv_data: list, labels: list, base_title: str) -> bool:
        all_k = np.concatenate(k_data)
        k_min, k_max = np.nanpercentile(all_k, [5, 95])
        k_grid = np.linspace(k_min, k_max, 50)

        iv_grid_data = []
        for k_points, iv_points in zip(k_data, iv_data):
            if len(k_points) > 1:
                iv_interp = np.interp(k_grid, k_points, iv_points, left=np.nan, right=np.nan)
            else:
                iv_interp = np.full_like(k_grid, np.nan)
            iv_grid_data.append(iv_interp)
        iv_tk = np.array(iv_grid_data)

        ax.clear()
        line, = ax.plot(k_grid, iv_tk[0], label="Smile", lw=2)
        ax.set_xlim(k_grid.min(), k_grid.max())
        iv_min, iv_max = np.nanpercentile(iv_tk, [1, 99])
        iv_range = max(iv_max - iv_min, 1e-6)
        ax.set_ylim(iv_min - 0.1 * iv_range, iv_max + 0.1 * iv_range)
        ax.set_xlabel("Moneyness (K/S)")
        ax.set_ylabel("Implied Volatility")
        ax.legend()
        ax.grid(True, alpha=0.3)

        def update_frame(i):
            line.set_ydata(iv_tk[i])
            ax.set_title(f"{base_title} - {labels[i]}")
            return [line]

        fig = ax.figure
        self._animation = FuncAnimation(fig, update_frame, frames=len(labels), interval=self._animation_speed, blit=True, repeat=True)
        update_frame(0)
        return True

    def _create_animated_surface(self, ax: plt.Axes, settings: dict) -> bool:
        from analysis.analysis_pipeline import available_dates

        target = settings["target"]
        peers = settings.get("peers", [])

        dates = available_dates(target)
        if len(dates) < 2:
            return False
        animation_dates = dates[-8:] if len(dates) > 8 else dates

        surfaces_data = []
        valid_dates = []
        for date in animation_dates:
            try:
                # Build grids (no asof_dates arg to keep compatibility)
                surfaces = build_surface_grids(
                    tickers=[target] + (peers or []),
                    max_expiries=settings.get("max_expiries", DEFAULT_MAX_EXPIRIES),
                    mny_bins=settings.get("mny_bins", DEFAULT_MONEYNESS_BINS),
                    use_atm_only=False,
                )
                if target in surfaces and date in surfaces[target]:
                    surface = surfaces[target][date]
                    if not surface.empty:
                        surfaces_data.append(surface)
                        valid_dates.append(date)
            except Exception:
                continue

        if len(valid_dates) < 2:
            return False

        first_surface = surfaces_data[0]
        tau_days = first_surface.columns.values
        k_levels = first_surface.index.values
        tau = np.array([float(t) for t in tau_days])
        k = np.array([float(str(k_str).split('-')[0]) if '-' in str(k_str) else float(k_str) for k_str in k_levels])

        iv_tktau = []
        for surface in surfaces_data:
            aligned = surface.reindex(index=k_levels, columns=tau_days, fill_value=np.nan)
            iv_tktau.append(aligned.values)
        iv_tktau = np.array(iv_tktau)

        ax.clear()
        vmin, vmax = np.nanpercentile(iv_tktau, [1, 99])
        im = ax.imshow(
            iv_tktau[0],
            origin="lower",
            aspect="auto",
            extent=[tau.min(), tau.max(), k.min(), k.max()],
            vmin=vmin,
            vmax=vmax,
            animated=True,
        )
        ax.set_xlabel("Time to Expiry (days)")
        ax.set_ylabel("Moneyness")
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.set_label("Implied Volatility")

        def update_surface(i):
            im.set_array(iv_tktau[i])
            ax.set_title(f"{target} IV Surface Animation - {valid_dates[i]}")
            return [im]

        fig = ax.figure
        self._animation = FuncAnimation(fig, update_surface, frames=len(valid_dates), interval=self._animation_speed, blit=True, repeat=True)
        update_surface(0)
        return True
