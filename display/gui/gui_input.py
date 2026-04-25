# display/gui/gui_input.py
from __future__ import annotations
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


DEFAULT_MODEL = "svi"
DEFAULT_ATM_BAND = 0.05
DEFAULT_CI = 0.68
DEFAULT_X_UNITS = "years"
DEFAULT_WEIGHT_METHOD = "corr"
DEFAULT_FEATURE_MODE = "iv_atm"
DEFAULT_PILLARS = [7,30,60,90,180,365]
DEFAULT_OVERLAY = False
PLOT_TYPES = (
    "Smile (K/S vs IV)",
    "Term (ATM vs T)",
    "Relative Weight Matrix",
    "Synthetic Surface (Smile)",
    "ETF Weights",
)

# Stable IDs decoupled from display labels. All routing logic must use these;
# never match against PLOT_TYPES strings directly so label renames stay safe.
PLOT_ID: dict[str, str] = {
    "Smile (K/S vs IV)":      "smile",
    "Term (ATM vs T)":         "term",
    "Relative Weight Matrix":  "corr_matrix",
    "Synthetic Surface (Smile)": "synthetic_surface",
    "ETF Weights":             "etf_weights",
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
    if text.startswith("Synthetic Surface"):
        return "synthetic_surface"
    if text.startswith("ETF Weights"):
        return "etf_weights"
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
    if pid == "etf_weights":
        return "surface" if feature_mode.startswith("surface") else "term"
    return "term"

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

    def __init__(self, master, *, overlay_synth: bool = True, overlay_peers: bool = True,
                 ci_percent: float = 68.0):
        super().__init__(master)
        self.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        # Central store for current settings
        self.manager = InputManager()
        
        # Initialize database and create default groups if needed
        self._init_ticker_groups()

        # =======================
        # Row 0: Presets
        # =======================
        row0 = ttk.Frame(self); row0.pack(side=tk.TOP, fill=tk.X, pady=(0,6))
        
        ttk.Label(row0, text="Presets").grid(row=0, column=0, sticky="w")
        self.cmb_presets = ttk.Combobox(row0, values=[], width=25, state="readonly")
        self.cmb_presets.grid(row=0, column=1, padx=(4,6))
        self.cmb_presets.bind("<<ComboboxSelected>>", self._on_preset_selected)
        
        self.btn_load_preset = ttk.Button(row0, text="Load", command=self._load_preset)
        self.btn_load_preset.grid(row=0, column=2, padx=2)
        
        self.btn_save_preset = ttk.Button(row0, text="Save", command=self._save_preset)
        self.btn_save_preset.grid(row=0, column=3, padx=2)
        
        self.btn_delete_preset = ttk.Button(row0, text="Delete", command=self._delete_preset)
        self.btn_delete_preset.grid(row=0, column=4, padx=2)
        
        self.btn_refresh_presets = ttk.Button(row0, text="Refresh", command=self._refresh_presets)
        self.btn_refresh_presets.grid(row=0, column=5, padx=2)
        
        # Load initial presets
        self._refresh_presets()

        # =======================
        # Row 1: Universe & Download
        # =======================
        row1 = ttk.Frame(self); row1.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(row1, text="Target").grid(row=0, column=0, sticky="w")
        self.ent_target = ttk.Entry(row1, width=12)
        self.ent_target.grid(row=0, column=1, padx=(4,10))
        self.ent_target.bind("<KeyRelease>", self._sync_settings)

        ttk.Label(row1, text="Peers (comma)").grid(row=0, column=2, sticky="w")
        self.ent_peers = ttk.Entry(row1, width=40)
        self.ent_peers.grid(row=0, column=3, padx=(4,10))
        self.ent_peers.bind("<KeyRelease>", self._sync_settings)

        ttk.Label(row1, text="Max expiries").grid(row=0, column=4, sticky="w")
        self.ent_maxexp = ttk.Entry(row1, width=6)
        self.ent_maxexp.insert(0, "6")
        self.ent_maxexp.grid(row=0, column=5, padx=(4,10))
        self.ent_maxexp.bind("<KeyRelease>", self._sync_settings)

        ttk.Label(row1, text="r").grid(row=0, column=6, sticky="w")
        self.ent_r = ttk.Entry(row1, width=8)
        self.ent_r.grid(row=0, column=7, padx=(4,4))
        self.ent_r.bind("<KeyRelease>", self._sync_settings)
        
        # Interest rate dropdown and management
        self.cmb_r_presets = ttk.Combobox(row1, values=[], width=12, state="readonly")
        self.cmb_r_presets.grid(row=0, column=8, padx=(2,2))
        self.cmb_r_presets.bind("<<ComboboxSelected>>", self._on_rate_preset_selected)
        self.cmb_r_presets.bind("<<ComboboxSelected>>", self._sync_settings)
        
        self.btn_save_rate = ttk.Button(row1, text="Save R", command=self._save_interest_rate, width=6)
        self.btn_save_rate.grid(row=0, column=9, padx=2)

        ttk.Label(row1, text="q").grid(row=0, column=10, sticky="w")
        self.ent_q = ttk.Entry(row1, width=6)
        self.ent_q.insert(0, "0.0")
        self.ent_q.grid(row=0, column=11, padx=(4,10))
        self.ent_q.bind("<KeyRelease>", self._sync_settings)

        self.btn_download = ttk.Button(row1, text="Download / Ingest")
        self.btn_download.grid(row=0, column=12, padx=8)

        # Initialize interest rates now that the related widgets exist
        self._init_interest_rates()

        # =======================
        # Row 2: Plot controls
        # =======================
        row2 = ttk.Frame(self); row2.pack(side=tk.TOP, fill=tk.X, pady=(6,0))

        ttk.Label(row2, text="Date").grid(row=0, column=0, sticky="w")
        self.cmb_date = ttk.Combobox(row2, values=[], width=12, state="readonly")
        self.cmb_date.grid(row=0, column=1, padx=6)
        self.cmb_date.bind("<<ComboboxSelected>>", self._sync_settings)

        ttk.Label(row2, text="Plot").grid(row=0, column=2, sticky="w")
        self.cmb_plot = ttk.Combobox(row2, values=PLOT_TYPES, width=21, state="readonly")
        self.cmb_plot.set(PLOT_TYPES[0])
        self.cmb_plot.grid(row=0, column=3, padx=6)
        self.cmb_plot.bind("<<ComboboxSelected>>", lambda e: (self._sync_settings(), self._refresh_visibility()))

        ttk.Label(row2, text="Model").grid(row=0, column=4, sticky="w")
        self.cmb_model = ttk.Combobox(row2, values=["svi", "sabr", "tps"], width=8, state="readonly")
        self.cmb_model.set(DEFAULT_MODEL)
        self.cmb_model.grid(row=0, column=5, padx=6)
        self.cmb_model.bind("<<ComboboxSelected>>", self._sync_settings)

        ttk.Label(row2, text="Target T (days)").grid(row=0, column=6, sticky="w")
        self.ent_days = ttk.Entry(row2, width=6)
        self.ent_days.insert(0, "30")
        self.ent_days.grid(row=0, column=7, padx=6)
        self.ent_days.bind("<KeyRelease>", self._sync_settings)

        ttk.Label(row2, text="CI (%)").grid(row=0, column=8, sticky="w")
        self.ent_ci = ttk.Entry(row2, width=6)
        self.ent_ci.insert(0, f"{ci_percent:.0f}")
        self.ent_ci.grid(row=0, column=9, padx=6)
        self.ent_ci.bind("<KeyRelease>", self._sync_settings)

        ttk.Label(row2, text="X units").grid(row=0, column=10, sticky="w")
        self.cmb_xunits = ttk.Combobox(row2, values=["years", "days"], width=8, state="readonly")
        self.cmb_xunits.set("years")
        self.cmb_xunits.grid(row=0, column=11, padx=6)
        self.cmb_xunits.bind("<<ComboboxSelected>>", self._sync_settings)
        
        row3 = ttk.Frame(self); row3.pack(side=tk.TOP, fill=tk.X, pady=(6,0))

        ttk.Label(row3, text="Pillars (days)").grid(row=0, column=0, sticky="w")
        self.ent_pillars = ttk.Entry(row3, width=18)
        self.ent_pillars.insert(0, "7,30,60,90,180,365")
        self.ent_pillars.grid(row=0, column=1, padx=6)
        self.ent_pillars.bind("<KeyRelease>", lambda e: (self._sync_settings(), self._refresh_visibility()))

        ttk.Label(row3, text="Weight method").grid(row=0, column=2, sticky="w")
        self.cmb_weight_method = ttk.Combobox(
            row3,
            values=["corr", "pca", "cosine", "equal", "oi"],
            width=10,
            state="readonly",
        )
        self.cmb_weight_method.set(DEFAULT_WEIGHT_METHOD)
        self.cmb_weight_method.grid(row=0, column=3, padx=6)
        self.cmb_weight_method.bind(
            "<<ComboboxSelected>>",
            lambda e: (self._sync_settings(), self._refresh_visibility(), self._replot_if_weights()),
        )

        ttk.Label(row3, text="Feature mode").grid(row=0, column=4, sticky="w")
        self.cmb_feature_mode = ttk.Combobox(
            row3,
            values=["iv_atm", "ul", "surface", "surface_grid"],
            width=14,
            state="readonly",
        )
        self.cmb_feature_mode.set(DEFAULT_FEATURE_MODE)
        self.cmb_feature_mode.grid(row=0, column=5, padx=6)
        self.cmb_feature_mode.bind(
            "<<ComboboxSelected>>",
            lambda e: (self._sync_settings(), self._refresh_visibility(), self._replot_if_weights()),
        )

        ttk.Label(row3, text="ATM band").grid(row=0, column=6, sticky="w")
        self.ent_atm_band = ttk.Entry(row3, width=6)
        self.ent_atm_band.insert(0, f"{DEFAULT_ATM_BAND:.2f}")
        self.ent_atm_band.grid(row=0, column=7, padx=6)
        self.ent_atm_band.bind("<KeyRelease>", self._sync_settings)

        # Row 4: Weight power and clipping controls
        row4 = ttk.Frame(self); row4.pack(side=tk.TOP, fill=tk.X, pady=(6,0))
        
        ttk.Label(row4, text="Weight power").grid(row=0, column=0, sticky="w")
        self.var_weight_power = tk.DoubleVar(value=1.0)
        self.scl_weight_power = ttk.Scale(row4, from_=1.0, to=3.0, variable=self.var_weight_power, 
                                          orient=tk.HORIZONTAL, length=120)
        self.scl_weight_power.grid(row=0, column=1, padx=6, sticky="w")
        self.var_weight_power.trace_add("write", lambda *args: self._sync_settings())
        
        self.var_clip_negative = tk.BooleanVar(value=True)
        self.chk_clip_negative = ttk.Checkbutton(row4, text="Clip negative weights", 
                                                variable=self.var_clip_negative)
        self.chk_clip_negative.grid(row=0, column=2, padx=8, sticky="w")
        self.var_clip_negative.trace_add("write", lambda *args: self._sync_settings())

        self.var_overlay_synth = tk.BooleanVar(value=bool(overlay_synth))
        self.chk_overlay_synth = ttk.Checkbutton(row4, text="Overlay synth", variable=self.var_overlay_synth)
        self.chk_overlay_synth.grid(row=0, column=3, padx=8, sticky="w")
        self.var_overlay_synth.trace_add("write", lambda *args: self._sync_settings())

        self.var_overlay_peers = tk.BooleanVar(value=bool(overlay_peers))
        self.chk_overlay_peers = ttk.Checkbutton(row4, text="Overlay peers", variable=self.var_overlay_peers)
        self.chk_overlay_peers.grid(row=0, column=4, padx=4, sticky="w")
        self.var_overlay_peers.trace_add("write", lambda *args: self._sync_settings())

        self.btn_plot = ttk.Button(row4, text="Plot")
        self.btn_plot.grid(row=0, column=5, padx=8)

        # initial sync of settings
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
            return 6

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
        return self.cmb_model.get() or DEFAULT_MODEL

    def get_T_days(self) -> float:
        try:
            return float(self.ent_days.get())
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
        return self.cmb_weight_method.get() or DEFAULT_WEIGHT_METHOD

    def get_feature_mode(self) -> str:
        return self.cmb_feature_mode.get() or DEFAULT_FEATURE_MODE

    def get_weight_power(self) -> float:
        return self.var_weight_power.get()

    def get_clip_negative(self) -> bool:
        return self.var_clip_negative.get()

    def get_pillars(self) -> list[int]:
        try:
            txt = self.ent_pillars.get().strip()
            if not txt:
                return list(DEFAULT_PILLARS)
            return [int(p.strip()) for p in txt.split(",") if p.strip().isdigit()]
        except Exception:
            return list(DEFAULT_PILLARS)

    def get_atm_band(self) -> float:
        try:
            return float(self.ent_atm_band.get())
        except Exception:
            return DEFAULT_ATM_BAND

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
                weight_method=weight_method,
                feature_mode=feature_mode,
                weight_power=self.get_weight_power(),
                clip_negative=self.get_clip_negative(),
                overlay_synth=self.get_overlay_synth(),
                overlay_peers=self.get_overlay_peers(),
                pillars=pillars,
                max_expiries=self.get_max_exp(),
                feature_scope=feature_scope,
            )
        except Exception:
            # Avoid raising UI errors from sync process
            pass

    def _replot_if_weights(self):
        """Re-render the plot immediately when weight/feature controls change and ETF Weights is active."""
        if plot_id(self.get_plot_type()) == "etf_weights" and hasattr(self, "_plot_fn"):
            self._plot_fn()

    def _refresh_visibility(self):
        wm = self.get_weight_method()
        feat = self.get_feature_mode()
        pid = plot_id(self.get_plot_type())

        show_T = pid == "smile"
        show_pillars = pid in ("term", "synthetic_surface") or (
            pid == "corr_matrix" and feat.startswith("surface")
        )

        if pid == "corr_matrix" and feat in ("iv_atm", "ul"):
            show_pillars = len(self.get_pillars()) >= 2
            show_T = not show_pillars

        show_model = (
            feat == "surface"
            and pid in ("smile", "synthetic_surface")
        )

        self.ent_days.configure(state=("normal" if show_T else "disabled"))
        self.ent_pillars.configure(state=("normal" if show_pillars else "disabled"))
        self.cmb_model.configure(state=("readonly" if show_model else "disabled"))

        self.cmb_xunits.configure(
            state=("readonly" if feat.startswith("surface") else "disabled")
        )
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
