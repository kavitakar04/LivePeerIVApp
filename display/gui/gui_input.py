# display/gui/gui_input.py
from __future__ import annotations
import json
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from typing import Callable, List, Any, Dict
from dataclasses import dataclass, field
import sys
from pathlib import Path

# Add project root to sys.path if not already there
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

@dataclass
class InputManager:
    """Lightweight store for GUI settings.

    The GUI updates this manager whenever a control changes so that callers
    can grab a coherent snapshot of the current configuration without polling
    each widget individually. This reduces the delay between editing a field
    and using the new values for plotting or data ingestion."""

    settings: Dict[str, Any] = field(default_factory=dict)

    def update(self, **kwargs: Any) -> None:
        """Merge provided key/value pairs into the settings store."""
        self.settings.update(kwargs)

    def as_dict(self) -> Dict[str, Any]:
        """Return a shallow copy of all current settings."""
        return dict(self.settings)

from data.ticker_groups import (
    save_ticker_group, load_ticker_group, list_ticker_groups,
    delete_ticker_group, create_default_groups
)
from data.interest_rates import (
    save_interest_rate, load_interest_rate, get_default_interest_rate,
    list_interest_rates, delete_interest_rate, set_default_interest_rate,
    get_interest_rate_names, create_default_interest_rates, STANDARD_RISK_FREE_RATE, STANDARD_DIVIDEND_YIELD
)
from data.db_utils import get_conn, ensure_initialized
from analysis.settings import (
    DEFAULT_ATM_BAND,
    DEFAULT_CI,
    DEFAULT_CLIP_NEGATIVE_WEIGHTS,
    DEFAULT_FEATURE_MODE,
    DEFAULT_MAX_EXPIRIES,
    DEFAULT_MODEL,
    DEFAULT_OVERLAY,
    DEFAULT_PILLAR_DAYS,
    DEFAULT_PILLAR_TOLERANCE_DAYS,
    DEFAULT_SMILE_MONEYNESS_RANGE,
    DEFAULT_UNDERLYING_LOOKBACK_DAYS,
    DEFAULT_WEIGHT_METHOD,
    DEFAULT_WEIGHT_POWER,
    DEFAULT_X_UNITS,
    format_moneyness_bins,
    format_moneyness_range,
    parse_int_list,
    parse_moneyness_bins,
    parse_moneyness_range,
)
from volModel.models import GUI_MODELS


DEFAULT_PILLARS = list(DEFAULT_PILLAR_DAYS)
PREFERENCES_PATH = ROOT / "data" / "gui_settings.json"
PERSISTED_SETTING_KEYS = (
    "target",
    "peers",
    "plot_type",
    "model",
    "T_days",
    "ci",
    "x_units",
    "atm_band",
    "smile_moneyness_range",
    "pillar_tolerance_days",
    "underlying_lookback_days",
    "mny_bins",
    "weight_method",
    "feature_mode",
    "weight_power",
    "clip_negative",
    "overlay_synth",
    "overlay_peers",
    "show_term_fit",
    "pillars",
    "max_expiries",
)
WEIGHT_METHODS = ("corr", "pca", "oi", "cosine", "equal")
WEIGHT_METHOD_LABELS = {
    "corr": "Correlation",
    "pca": "PCA",
    "oi": "Open Interest",
    "cosine": "Cosine Similarity",
    "equal": "Equal Weight",
}
WEIGHT_METHOD_IDS_BY_LABEL = {label: key for key, label in WEIGHT_METHOD_LABELS.items()}
WEIGHT_METHOD_DISPLAY = tuple(WEIGHT_METHOD_LABELS[key] for key in WEIGHT_METHODS)

MODEL_LABELS = {
    "svi": "SVI",
    "sabr": "SABR",
    "tps": "TPS",
}
MODEL_IDS_BY_LABEL = {label: key for key, label in MODEL_LABELS.items()}

FEATURE_MODE_LABELS = {
    "iv_atm": "Term Structure",
    "ul": "Underlying Returns",
    "surface": "Surface",
    "surface_grid": "Surface Grid",
}
FEATURE_MODE_IDS_BY_LABEL = {label: key for key, label in FEATURE_MODE_LABELS.items()}
FEATURE_MODE_DISPLAY = tuple(FEATURE_MODE_LABELS[key] for key in ("iv_atm", "ul", "surface", "surface_grid"))


def weight_method_label(method: str) -> str:
    return WEIGHT_METHOD_LABELS.get(str(method or "").lower(), str(method or ""))


def weight_method_id(label_or_method: str) -> str:
    value = str(label_or_method or "").strip()
    if value in WEIGHT_METHOD_IDS_BY_LABEL:
        return WEIGHT_METHOD_IDS_BY_LABEL[value]
    lowered = value.lower()
    return lowered if lowered in WEIGHT_METHODS else DEFAULT_WEIGHT_METHOD


def model_label(model: str) -> str:
    return MODEL_LABELS.get(str(model or "").lower(), str(model or ""))


def model_id(label_or_model: str) -> str:
    value = str(label_or_model or "").strip()
    if value in MODEL_IDS_BY_LABEL:
        return MODEL_IDS_BY_LABEL[value]
    lowered = value.lower()
    return lowered if lowered in GUI_MODELS else DEFAULT_MODEL


def feature_mode_label(feature_mode: str) -> str:
    return FEATURE_MODE_LABELS.get(str(feature_mode or "").lower(), str(feature_mode or ""))


def feature_mode_id(label_or_mode: str) -> str:
    value = str(label_or_mode or "").strip()
    if value in FEATURE_MODE_IDS_BY_LABEL:
        return FEATURE_MODE_IDS_BY_LABEL[value]
    lowered = value.lower()
    return lowered if lowered in FEATURE_MODE_LABELS else DEFAULT_FEATURE_MODE


def _load_gui_preferences() -> dict[str, Any]:
    try:
        if not PREFERENCES_PATH.is_file():
            return {}
        with PREFERENCES_PATH.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _persistable_settings(settings: dict[str, Any]) -> dict[str, Any]:
    return {
        key: _json_safe(settings[key])
        for key in PERSISTED_SETTING_KEYS
        if key in settings
    }


def _save_gui_preferences(settings: dict[str, Any], path: Path = PREFERENCES_PATH) -> None:
    data = _persistable_settings(settings)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)
            fh.write("\n")
        tmp.replace(path)
    except Exception:
        pass


def _json_safe(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    return value


def _format_pref_bins(value: Any) -> str:
    bins = parse_moneyness_bins(
        ",".join(f"{lo}-{hi}" for lo, hi in value) if isinstance(value, list) else str(value)
    )
    return format_moneyness_bins(bins)


def _format_pref_range(value: Any) -> str:
    if isinstance(value, list) and len(value) == 2:
        text = f"{value[0]}-{value[1]}"
    else:
        text = str(value)
    return format_moneyness_range(parse_moneyness_range(text))
PLOT_TYPES = (
    "Smile (K/S vs IV)",
    "Term (ATM vs T)",
    "Relative Weight Matrix",
    "Peer Composite Surface",
    "RV Heatmap",
)

# Stable IDs decoupled from display labels. All routing logic must use these;
# never match against PLOT_TYPES strings directly so label renames stay safe.
PLOT_ID: dict[str, str] = {
    "Smile (K/S vs IV)":        "smile",
    "Term (ATM vs T)":           "term",
    "Relative Weight Matrix":    "corr_matrix",
    "Peer Composite Surface": "synthetic_surface",
    "RV Heatmap":                "rv_heatmap",
}


def plot_id(label: str) -> str:
    """Return the stable routing ID for a display label."""
    if label in PLOT_ID:
        return PLOT_ID[label]
    text = str(label or "")
    if text.startswith("Smile"):
        return "smile"
    if text.startswith("Term"):
        return "term"
    if text.startswith("Relative Weight Matrix"):
        return "corr_matrix"
    if text.startswith("Peer Composite Surface"):
        return "synthetic_surface"
    return text


def _derive_feature_scope(plot_type: str, feature_mode: str) -> str:
    """Infer feature scope from plot and feature mode.

    Returns one of: 'smile', 'term', 'surface'."""
    pid = plot_id(plot_type)
    if pid == "smile":
        return "smile"
    if pid == "term":
        return "term"
    if pid == "synthetic_surface":
        return "surface"
    if pid == "corr_matrix":
        return "term" if feature_mode in ("iv_atm", "ul") else "surface"
    return "term"


def gui_model_values() -> list[str]:
    """Return user-selectable smile model names."""
    return [model_label(model) for model in GUI_MODELS]


def gui_feature_mode_values() -> list[str]:
    """Return user-selectable feature mode names."""
    return list(FEATURE_MODE_DISPLAY)


def model_selection_state(plot_type: str, feature_mode: str) -> str:
    """Return combobox state for plots that support explicit smile models."""
    pid = plot_id(plot_type)
    if pid in ("smile", "synthetic_surface"):
        return "readonly"
    return "disabled"

class InputPanel(ttk.Frame):
    """
    Encapsulates all GUI inputs and exposes getters/setters + callbacks.
    Browser/runner can:
      - read current settings via getters
      - set date list via set_dates(...)
      - bind target-entry changes and button clicks

    Parameters
    ----------
    master : tk.Widget
        Parent widget.
    overlay_synth : bool, optional
        Initial state for the synthetic overlay checkbox.
    overlay_peers : bool, optional
        Initial state for the peer overlay checkbox.
    ci_percent : float, optional
        Confidence interval expressed in percentage (e.g. 68 for 68%).
    """

    def __init__(self, master, *, overlay_synth: bool = True, overlay_peers: bool = False,
                 ci_percent: float = 68.0, settings_parent=None):
        super().__init__(master)
        self.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        # Central store for current settings
        self.manager = InputManager()
        self._preferences = _load_gui_preferences()
        
        # Initialize database and create default groups if needed
        self._init_ticker_groups()

        # =======================
        # Row 0: Presets
        # =======================
        row0 = ttk.Frame(self); row0.pack(side=tk.TOP, fill=tk.X, pady=(0,6))

        ttk.Label(row0, text="Preset").grid(row=0, column=0, sticky="w")
        self.cmb_presets = ttk.Combobox(row0, values=[], width=28, state="readonly")
        self.cmb_presets.grid(row=0, column=1, padx=(4, 6))
        self.cmb_presets.bind("<<ComboboxSelected>>", self._on_preset_selected)

        self.btn_save_preset = ttk.Button(row0, text="Save", command=self._save_preset)
        self.btn_save_preset.grid(row=0, column=2, padx=2)

        self.btn_delete_preset = ttk.Button(row0, text="Delete", command=self._delete_preset)
        self.btn_delete_preset.grid(row=0, column=3, padx=2)

        self._refresh_presets()

        # =======================
        # Row 1: Universe & Download
        # =======================
        row1 = ttk.Frame(self); row1.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(row1, text="Target").grid(row=0, column=0, sticky="w")
        self.ent_target = ttk.Entry(row1, width=12)
        if self._preferences.get("target"):
            self.ent_target.insert(0, str(self._preferences.get("target", "")).upper())
        self.ent_target.grid(row=0, column=1, padx=(4, 10))
        self.ent_target.bind("<KeyRelease>", self._sync_settings)

        ttk.Label(row1, text="Peers").grid(row=0, column=2, sticky="w")
        self.ent_peers = ttk.Entry(row1, width=44)
        pref_peers = self._preferences.get("peers")
        if isinstance(pref_peers, list):
            self.ent_peers.insert(0, ", ".join(str(p).upper() for p in pref_peers if str(p).strip()))
        elif pref_peers:
            self.ent_peers.insert(0, str(pref_peers))
        self.ent_peers.grid(row=0, column=3, padx=(4, 10))
        self.ent_peers.bind("<KeyRelease>", self._sync_settings)

        ttk.Label(row1, text="r").grid(row=0, column=4, sticky="w")
        self.ent_r = ttk.Entry(row1, width=8)
        self.ent_r.grid(row=0, column=5, padx=(4, 6))
        self.ent_r.bind("<KeyRelease>", self._sync_settings)

        ttk.Label(row1, text="q").grid(row=0, column=6, sticky="w")
        self.ent_q = ttk.Entry(row1, width=6)
        self.ent_q.insert(0, "0.0")
        self.ent_q.grid(row=0, column=7, padx=(4, 10))
        self.ent_q.bind("<KeyRelease>", self._sync_settings)

        self.btn_download = ttk.Button(row1, text="Download / Ingest")
        self.btn_download.grid(row=0, column=8, padx=8)

        # stub widget kept for API compatibility (not shown)
        self.cmb_r_presets = ttk.Combobox(row1, values=[], width=1, state="readonly")

        self._init_interest_rates()

        # =======================
        # Row 2: Plot controls
        # =======================
        row2 = ttk.Frame(self); row2.pack(side=tk.TOP, fill=tk.X, pady=(6, 0))

        ttk.Label(row2, text="Date").grid(row=0, column=0, sticky="w")
        self.cmb_date = ttk.Combobox(row2, values=[], width=12, state="readonly")
        self.cmb_date.grid(row=0, column=1, padx=6)
        self.cmb_date.bind("<<ComboboxSelected>>", self._sync_settings)

        ttk.Label(row2, text="Plot").grid(row=0, column=2, sticky="w")
        self.cmb_plot = ttk.Combobox(row2, values=PLOT_TYPES, width=21, state="readonly")
        pref_plot = str(self._preferences.get("plot_type") or "")
        self.cmb_plot.set(pref_plot if pref_plot in PLOT_TYPES else PLOT_TYPES[0])
        self.cmb_plot.grid(row=0, column=3, padx=6)
        self.cmb_plot.bind("<<ComboboxSelected>>", lambda e: (self._sync_settings(), self._refresh_visibility()))

        ttk.Label(row2, text="Model").grid(row=0, column=4, sticky="w")
        self.cmb_model = ttk.Combobox(
            row2,
            values=gui_model_values(),
            width=8,
            state="readonly",
        )
        self.cmb_model.set(model_label(str(self._preferences.get("model") or DEFAULT_MODEL)))
        self.cmb_model.grid(row=0, column=5, padx=6)
        self.cmb_model.bind("<<ComboboxSelected>>", self._sync_settings)

        # T-days is an internal smile focus used by click-through/drilldown.
        # The visible expiry controls are Prev/Next Expiry.
        self.var_T_days = tk.StringVar(value=str(self._preferences.get("T_days") or "30"))

        # stub widgets kept for API compatibility (not shown)
        self.cmb_xunits = ttk.Combobox(row2, values=["years", "days"], width=1, state="readonly")
        self.cmb_xunits.set(DEFAULT_X_UNITS)

        # =======================
        # Row 3: Weights & actions
        # =======================
        row3 = ttk.Frame(self); row3.pack(side=tk.TOP, fill=tk.X, pady=(6, 0))

        ttk.Label(row3, text="Weights").grid(row=0, column=0, sticky="w")
        self.cmb_weight_method = ttk.Combobox(
            row3, values=WEIGHT_METHOD_DISPLAY,
            width=18, state="readonly",
        )
        self.cmb_weight_method.set(weight_method_label(str(self._preferences.get("weight_method") or DEFAULT_WEIGHT_METHOD)))
        self.cmb_weight_method.grid(row=0, column=1, padx=6)
        self.cmb_weight_method.bind(
            "<<ComboboxSelected>>",
            lambda e: (self._sync_settings(), self._refresh_visibility(), self._replot_if_weights()),
        )

        ttk.Label(row3, text="Features").grid(row=0, column=2, sticky="w")
        self.cmb_feature_mode = ttk.Combobox(
            row3, values=gui_feature_mode_values(),
            width=20, state="readonly",
        )
        self.cmb_feature_mode.set(feature_mode_label(str(self._preferences.get("feature_mode") or DEFAULT_FEATURE_MODE)))
        self.cmb_feature_mode.grid(row=0, column=3, padx=6)
        self.cmb_feature_mode.bind(
            "<<ComboboxSelected>>",
            lambda e: (self._sync_settings(), self._refresh_visibility(), self._replot_if_weights()),
        )

        self.var_overlay_synth = tk.BooleanVar(value=bool(self._preferences.get("overlay_synth", overlay_synth)))
        self.chk_overlay_synth = ttk.Checkbutton(row3, text="Overlay synth",
                                                  variable=self.var_overlay_synth)
        self.chk_overlay_synth.grid(row=0, column=4, padx=8, sticky="w")
        self.var_overlay_synth.trace_add("write", lambda *args: self._sync_settings())

        self.var_overlay_peers = tk.BooleanVar(value=bool(self._preferences.get("overlay_peers", overlay_peers)))
        self.chk_overlay_peers = ttk.Checkbutton(row3, text="Show individual peers",
                                                  variable=self.var_overlay_peers)
        self.chk_overlay_peers.grid(row=0, column=5, padx=4, sticky="w")
        self.var_overlay_peers.trace_add("write", lambda *args: self._sync_settings())

        self.var_show_term_fit = tk.BooleanVar(value=bool(self._preferences.get("show_term_fit", False)))
        self.chk_show_term_fit = ttk.Checkbutton(row3, text="Show term fit",
                                                 variable=self.var_show_term_fit)
        self.chk_show_term_fit.grid(row=0, column=6, padx=4, sticky="w")
        self.var_show_term_fit.trace_add("write", lambda *args: self._sync_settings())

        self.btn_plot = ttk.Button(row3, text="Plot")
        self.btn_plot.grid(row=0, column=7, padx=12)

        # stub vars for API compatibility
        self.var_weight_power = tk.DoubleVar(value=float(self._preferences.get("weight_power", DEFAULT_WEIGHT_POWER)))
        self.var_clip_negative = tk.BooleanVar(value=bool(self._preferences.get("clip_negative", DEFAULT_CLIP_NEGATIVE_WEIGHTS)))

        # =======================
        # Row 4: Analysis settings
        # =======================
        settings_container = settings_parent if settings_parent is not None else self
        row4 = ttk.LabelFrame(settings_container, text="Analysis Settings")
        row4.pack(side=tk.TOP, fill=tk.X, padx=8, pady=8)

        ttk.Label(row4, text="Max exp").grid(row=0, column=0, sticky="w")
        self.ent_maxexp = ttk.Entry(row4, width=5)
        self.ent_maxexp.insert(0, str(self._preferences.get("max_expiries", DEFAULT_MAX_EXPIRIES)))
        self.ent_maxexp.grid(row=0, column=1, padx=(4, 10))
        self.ent_maxexp.bind("<KeyRelease>", self._sync_settings)

        ttk.Label(row4, text="Pillars").grid(row=0, column=2, sticky="w")
        self.ent_pillars = ttk.Entry(row4, width=22)
        pref_pillars = self._preferences.get("pillars", DEFAULT_PILLARS)
        if isinstance(pref_pillars, list):
            self.ent_pillars.insert(0, ",".join(str(x) for x in pref_pillars))
        else:
            self.ent_pillars.insert(0, str(pref_pillars))
        self.ent_pillars.grid(row=0, column=3, padx=(4, 10))
        self.ent_pillars.bind("<KeyRelease>", self._sync_settings)

        ttk.Label(row4, text="Tol").grid(row=0, column=4, sticky="w")
        self.ent_pillar_tol = ttk.Entry(row4, width=5)
        self.ent_pillar_tol.insert(0, f"{float(self._preferences.get('pillar_tolerance_days', DEFAULT_PILLAR_TOLERANCE_DAYS)):g}")
        self.ent_pillar_tol.grid(row=0, column=5, padx=(4, 10))
        self.ent_pillar_tol.bind("<KeyRelease>", self._sync_settings)

        ttk.Label(row4, text="ATM band").grid(row=0, column=6, sticky="w")
        self.ent_atm_band = ttk.Entry(row4, width=6)
        self.ent_atm_band.insert(0, f"{float(self._preferences.get('atm_band', DEFAULT_ATM_BAND)):.2f}")
        self.ent_atm_band.grid(row=0, column=7, padx=(4, 10))
        self.ent_atm_band.bind("<KeyRelease>", self._sync_settings)

        ttk.Label(row4, text="Mny bins").grid(row=0, column=8, sticky="w")
        self.ent_mny_bins = ttk.Entry(row4, width=30)
        pref_bins = self._preferences.get("mny_bins")
        self.ent_mny_bins.insert(0, _format_pref_bins(pref_bins) if pref_bins else format_moneyness_bins())
        self.ent_mny_bins.grid(row=0, column=9, padx=(4, 10))
        self.ent_mny_bins.bind("<KeyRelease>", self._sync_settings)

        ttk.Label(row4, text="Smile K/S").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.ent_smile_mny_range = ttk.Entry(row4, width=10)
        pref_smile_range = self._preferences.get("smile_moneyness_range")
        self.ent_smile_mny_range.insert(0, _format_pref_range(pref_smile_range) if pref_smile_range else format_moneyness_range())
        self.ent_smile_mny_range.grid(row=1, column=1, padx=(4, 10), pady=(6, 0))
        self.ent_smile_mny_range.bind("<KeyRelease>", self._sync_settings)

        ttk.Label(row4, text="CI (%)").grid(row=1, column=2, sticky="w", pady=(6, 0))
        self.ent_ci = ttk.Entry(row4, width=6)
        pref_ci = self._preferences.get("ci")
        if pref_ci is not None:
            try:
                pref_ci_value = float(pref_ci)
                if pref_ci_value <= 1:
                    pref_ci_value *= 100.0
                self.ent_ci.insert(0, f"{pref_ci_value:.0f}")
            except Exception:
                self.ent_ci.insert(0, f"{ci_percent:.0f}")
        else:
            self.ent_ci.insert(0, f"{ci_percent:.0f}")
        self.ent_ci.grid(row=1, column=3, padx=(4, 10), pady=(6, 0))
        self.ent_ci.bind("<KeyRelease>", self._sync_settings)

        ttk.Label(row4, text="UL days").grid(row=1, column=4, sticky="w", pady=(6, 0))
        self.ent_underlying_days = ttk.Entry(row4, width=7)
        self.ent_underlying_days.insert(
            0,
            str(self._preferences.get("underlying_lookback_days", DEFAULT_UNDERLYING_LOOKBACK_DAYS)),
        )
        self.ent_underlying_days.grid(row=1, column=5, padx=(4, 10), pady=(6, 0))
        self.ent_underlying_days.bind("<KeyRelease>", self._sync_settings)

        # initial sync
        self._sync_settings()
        self._refresh_visibility()


    # ---------- bindings ----------
    def bind_download(self, fn: Callable[[], None]):
        self.btn_download.configure(command=fn)

    def bind_plot(self, fn: Callable[[], None]):
        self.btn_plot.configure(command=fn)
        self._plot_fn = fn

    def bind_target_change(self, fn: Callable):
        # run when user confirms/enters target
        self.ent_target.bind("<FocusOut>", fn)
        self.ent_target.bind("<Return>", fn)

    # ---------- setters ----------
    def set_dates(self, dates: List[str]):
        self.cmb_date["values"] = dates or []
        if dates:
            self.cmb_date.current(len(dates) - 1)
        self._sync_settings()

    def set_rates(self, r: float = STANDARD_RISK_FREE_RATE, q: float = STANDARD_DIVIDEND_YIELD) -> None:
        """Set the risk-free and dividend rates displayed in the UI."""
        self.ent_r.delete(0, tk.END)
        self.ent_r.insert(0, f"{r:.4f}")
        self.ent_q.delete(0, tk.END)
        self.ent_q.insert(0, f"{q:.4f}")
        self._sync_settings()

    def set_T_days(self, days: float) -> None:
        """Set the internal smile focus tenor used by drilldowns."""
        try:
            self.var_T_days.set(str(int(round(float(days)))))
        except Exception:
            self.var_T_days.set("30")
        self._sync_settings()

    def _parse_rate(self, text: str, default: float) -> float:
        """Parse user-entered rate; accepts percents or decimals."""
        try:
            txt = text.strip().replace('%', '')
            if not txt:
                return default
            val = float(txt)
            if val > 1:
                val /= 100.0
            return val
        except Exception:
            return default

    # ---------- getters ----------
    def get_target(self) -> str:
        return (self.ent_target.get() or "").strip().upper()

    def get_peers(self) -> list[str]:
        txt = (self.ent_peers.get() or "").strip()
        if not txt:
            return []
        return [p.strip().upper() for p in txt.split(",") if p.strip()]

    def get_overlay_synth(self) -> bool:
        return bool(self.var_overlay_synth.get())

    def get_overlay_peers(self) -> bool:
        return bool(self.var_overlay_peers.get())

    def get_overlay(self) -> bool:
        """Backward-compatible synthetic overlay getter."""
        return self.get_overlay_synth()

    def get_max_exp(self) -> int:
        try:
            return int(float(self.ent_maxexp.get()))
        except Exception:
            return DEFAULT_MAX_EXPIRIES

    def get_interest_rate(self) -> float:
        """Get the current interest rate value from the persistent system."""
        try:
            rate_str = self.ent_r.get().strip()
            if not rate_str:
                return get_default_interest_rate()
            
            rate_value = float(rate_str)
            
            # Convert to decimal if percentage (values > 1 are assumed to be percentages)
            if rate_value > 1:
                rate_value = rate_value / 100.0
            
            return rate_value
            
        except ValueError:
            # Return default if parsing fails
            return get_default_interest_rate()

    def get_rates(self) -> tuple[float, float]:
        """Get interest rate and dividend yield. Uses persistent interest rate system."""
        r = self.get_interest_rate()  # Use our new persistent interest rate method
        try:
            q = float(self.ent_q.get())
        except Exception:
            q = 0.0
        return r, q

    def get_plot_type(self) -> str:
        return self.cmb_plot.get()

    def get_asof(self) -> str:
        return (self.cmb_date.get() or "").strip()

    def get_model(self) -> str:
        return model_id(self.cmb_model.get() or DEFAULT_MODEL)

    def get_T_days(self) -> float:
        try:
            return float(self.var_T_days.get())
        except Exception:
            return 30.0

    def get_ci(self) -> float:
        """Return CI level as decimal; accepts percentage inputs."""
        try:
            val = float(self.ent_ci.get())
            if val > 1:
                val /= 100.0
            return val
        except Exception:
            return DEFAULT_CI


    def get_x_units(self) -> str:
        return self.cmb_xunits.get() or DEFAULT_X_UNITS

    def get_weight_method(self) -> str:
        return weight_method_id(self.cmb_weight_method.get())

    def get_feature_mode(self) -> str:
        return feature_mode_id(self.cmb_feature_mode.get() or DEFAULT_FEATURE_MODE)

    def get_weight_power(self) -> float:
        return self.var_weight_power.get()

    def get_clip_negative(self) -> bool:
        return self.var_clip_negative.get()

    def get_show_term_fit(self) -> bool:
        return bool(self.var_show_term_fit.get())

    def get_pillars(self) -> list[int]:
        return parse_int_list(self.ent_pillars.get(), tuple(DEFAULT_PILLARS))

    def get_pillar_tolerance_days(self) -> float:
        try:
            return float(self.ent_pillar_tol.get())
        except Exception:
            return DEFAULT_PILLAR_TOLERANCE_DAYS

    def get_atm_band(self) -> float:
        try:
            return float(self.ent_atm_band.get())
        except Exception:
            return DEFAULT_ATM_BAND

    def get_mny_bins(self):
        return parse_moneyness_bins(self.ent_mny_bins.get())

    def get_smile_moneyness_range(self) -> tuple[float, float]:
        return parse_moneyness_range(self.ent_smile_mny_range.get(), DEFAULT_SMILE_MONEYNESS_RANGE)

    def get_underlying_lookback_days(self) -> int:
        try:
            return max(1, int(float(self.ent_underlying_days.get())))
        except Exception:
            return DEFAULT_UNDERLYING_LOOKBACK_DAYS

    def get_settings(self) -> dict:
        """Return a snapshot of all current settings."""
        self._sync_settings()
        return self.manager.as_dict()

    def export_settings_for_engine(self) -> dict:
        st = self.get_settings()
        legacy_mode = {"smile": "atm", "term": "term", "surface": "surface"}[st.get("feature_scope", "term")]
        st["mode"] = legacy_mode
        return st

    def _sync_settings(self, *_):
        """Synchronize widgets to the central InputManager."""
        try:
            plot_type = self.get_plot_type()
            weight_method = self.get_weight_method()
            feature_mode = self.get_feature_mode()
            pillars = self.get_pillars()

            feature_scope = _derive_feature_scope(plot_type, feature_mode)
            if plot_id(plot_type) == "corr_matrix" and feature_mode in ("iv_atm", "ul"):
                feature_scope = "term" if len(pillars) >= 2 else "smile"

            self.manager.update(
                target=self.get_target(),
                peers=self.get_peers(),
                plot_type=plot_type,
                asof=self.get_asof(),
                model=self.get_model(),
                T_days=self.get_T_days(),
                ci=self.get_ci(),
                x_units=self.get_x_units(),
                atm_band=self.get_atm_band(),
                smile_moneyness_range=self.get_smile_moneyness_range(),
                pillar_tolerance_days=self.get_pillar_tolerance_days(),
                underlying_lookback_days=self.get_underlying_lookback_days(),
                mny_bins=self.get_mny_bins(),
                weight_method=weight_method,
                feature_mode=feature_mode,
                weight_power=self.get_weight_power(),
                clip_negative=self.get_clip_negative(),
                overlay_synth=self.get_overlay_synth(),
                overlay_peers=self.get_overlay_peers(),
                show_term_fit=self.get_show_term_fit(),
                pillars=pillars,
                max_expiries=self.get_max_exp(),
                feature_scope=feature_scope,
            )
            _save_gui_preferences(self.manager.as_dict())
        except Exception:
            # Avoid raising UI errors from sync process
            pass

    def _replot_if_weights(self):
        """Re-render plots that directly depend on weight/feature controls."""
        if plot_id(self.get_plot_type()) in {"smile", "term", "synthetic_surface", "corr_matrix", "rv_heatmap"} and hasattr(self, "_plot_fn"):
            self._plot_fn()

    def _refresh_visibility(self):
        wm = self.get_weight_method()
        feat = self.get_feature_mode()
        pid = plot_id(self.get_plot_type())

        self.cmb_model.configure(state=model_selection_state(pid, feat))
        self.chk_show_term_fit.configure(state=("normal" if pid == "term" else "disabled"))
        self.cmb_feature_mode.configure(
            state=("disabled" if wm == "oi" else "readonly")
        )
    
    # ---------- preset management ----------
    def _init_ticker_groups(self):
        """Initialize database for ticker groups and create defaults if needed."""
        try:
            conn = get_conn()
            ensure_initialized(conn)
            
            # Check if we have any groups, if not create defaults
            groups = list_ticker_groups(conn)
            if not groups:
                create_default_groups(conn)
            
            conn.close()
        except Exception as e:
            print(f"Error initializing ticker groups: {e}")
    
    def _refresh_presets(self):
        """Refresh the preset dropdown with current groups from database."""
        try:
            groups = list_ticker_groups()
            group_names = [group["group_name"] for group in groups]
            self.cmb_presets["values"] = group_names
            if group_names:
                self.cmb_presets.set("")  # Clear selection
        except Exception as e:
            print(f"Error refreshing presets: {e}")
            messagebox.showerror("Error", f"Failed to refresh presets: {e}")
    
    def _on_preset_selected(self, event=None):
        """Called when user selects a preset from dropdown."""
        # Auto-load when selection changes
        self._load_preset()
    
    def _load_preset(self):
        """Load the selected preset into the target and peers fields."""
        selected = self.cmb_presets.get()
        if not selected:
            return
            
        try:
            group = load_ticker_group(selected)
            if group is None:
                messagebox.showerror("Error", f"Preset '{selected}' not found!")
                self._refresh_presets()
                return

            # Update the GUI fields
            self.ent_target.delete(0, tk.END)
            self.ent_target.insert(0, group["target_ticker"])

            self.ent_peers.delete(0, tk.END)
            self.ent_peers.insert(0, ", ".join(group["peer_tickers"]))

            # Show description if available
            if group.get("description"):
                print(f"Loaded preset: {selected} - {group['description']}")

            self._sync_settings()

        except Exception as e:
            print(f"Error loading preset: {e}")
            messagebox.showerror("Error", f"Failed to load preset: {e}")
    
    def _save_preset(self):
        """Save current target and peers as a new preset."""
        target = self.get_target()
        peers = self.get_peers()
        
        if not target:
            messagebox.showerror("Error", "Please enter a target ticker before saving preset.")
            return
            
        if not peers:
            messagebox.showerror("Error", "Please enter peer tickers before saving preset.")
            return
        
        # Ask for preset name
        group_name = simpledialog.askstring(
            "Save Preset", 
            "Enter a name for this preset:",
            initialvalue=f"{target} vs peers"
        )
        
        if not group_name:
            return
            
        # Ask for optional description
        description = simpledialog.askstring(
            "Save Preset", 
            "Enter an optional description:",
            initialvalue=""
        ) or ""
        
        try:
            success = save_ticker_group(
                group_name=group_name,
                target_ticker=target,
                peer_tickers=peers,
                description=description
            )
            
            if success:
                self._refresh_presets()
                self.cmb_presets.set(group_name)
                print(f"Success: Preset '{group_name}' saved successfully!")
            else:
                messagebox.showerror("Error", "Failed to save preset.")
                
        except Exception as e:
            print(f"Error saving preset: {e}")
            messagebox.showerror("Error", f"Failed to save preset: {e}")
    
    def _delete_preset(self):
        """Delete the selected preset."""
        selected = self.cmb_presets.get()
        if not selected:
            messagebox.showerror("Error", "Please select a preset to delete.")
            return
        
        # Confirm deletion
        if not messagebox.askyesno("Confirm Delete", 
                                   f"Are you sure you want to delete preset '{selected}'?"):
            return
        
        try:
            success = delete_ticker_group(selected)
            if success:
                self._refresh_presets()
                print(f"Success: Preset '{selected}' deleted successfully!")
            else:
                messagebox.showerror("Error", f"Preset '{selected}' not found or could not be deleted.")
                
        except Exception as e:
            print(f"Error deleting preset: {e}")
            messagebox.showerror("Error", f"Failed to delete preset: {e}")
    
    def get_selected_preset_name(self) -> str:
        """Get the currently selected preset name."""
        return self.cmb_presets.get()
    
    # ==========================================
    # Interest Rate Management Methods
    # ==========================================
    
    def _init_interest_rates(self):
        """Initialize interest rates and load default."""
        try:
            conn = get_conn()
            ensure_initialized(conn)
            create_default_interest_rates()
            self._refresh_interest_rates()
            
            # Load and set the default rate
            default_rate = get_default_interest_rate()
            self.ent_r.delete(0, tk.END)
            self.ent_r.insert(0, f"{default_rate:.4f}")
            
        except Exception as e:
            print(f"Error initializing interest rates: {e}")
    
    def _refresh_interest_rates(self):
        """Refresh the interest rate dropdown with current rates."""
        try:
            rate_names = get_interest_rate_names()
            self.cmb_r_presets['values'] = rate_names
            
            # Set to default if exists
            for rate_id, rate_value, description, is_default in list_interest_rates():
                if is_default:
                    self.cmb_r_presets.set(rate_id)
                    break
                    
        except Exception as e:
            print(f"Error refreshing interest rates: {e}")
    
    def _on_rate_preset_selected(self, event=None):
        """Handle selection of an interest rate preset."""
        selected = self.cmb_r_presets.get()
        if not selected:
            return
        
        try:
            rate_data = load_interest_rate(selected)
            if rate_data:
                rate_value, description, is_default = rate_data
                self.ent_r.delete(0, tk.END)
                self.ent_r.insert(0, f"{rate_value:.4f}")
                self._sync_settings()

        except Exception as e:
            print(f"Error loading interest rate: {e}")
            messagebox.showerror("Error", f"Failed to load interest rate: {e}")
    
    def _save_interest_rate(self):
        """Save the current interest rate as a new preset."""
        try:
            # Get current rate value
            rate_str = self.ent_r.get().strip()
            if not rate_str:
                messagebox.showerror("Error", "Please enter an interest rate value.")
                return
            
            try:
                rate_value = float(rate_str)
            except ValueError:
                messagebox.showerror("Error", "Please enter a valid numeric interest rate.")
                return
            
            # Convert to percentage if needed (values > 1 are assumed to be percentages)
            if rate_value > 1:
                rate_value = rate_value / 100.0
                print(f"Info: Converted {rate_str}% to {rate_value:.4f} (decimal form)")
            
            # Ask for rate name
            rate_id = simpledialog.askstring(
                "Save Interest Rate", 
                "Enter a name for this interest rate:",
                initialvalue=f"rate_{rate_value*100:.2f}pct"
            )
            
            if not rate_id:
                return
            
            # Ask for description
            description = simpledialog.askstring(
                "Save Interest Rate", 
                "Enter an optional description:",
                initialvalue=f"{rate_value*100:.2f}% interest rate"
            ) or ""
            
            # Ask if this should be the default
            is_default = messagebox.askyesno(
                "Set as Default", 
                "Set this as the default interest rate?"
            )
            
            # Save the rate
            save_interest_rate(rate_id, rate_value, description, is_default)
            
            # Refresh the dropdown
            self._refresh_interest_rates()
            self.cmb_r_presets.set(rate_id)
            
            
        except Exception as e:
            print(f"Error saving interest rate: {e}")
            messagebox.showerror("Error", f"Failed to save interest rate: {e}")
