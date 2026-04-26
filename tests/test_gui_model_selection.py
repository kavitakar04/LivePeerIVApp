from display.gui.gui_input import (
    feature_mode_id,
    feature_mode_label,
    gui_feature_mode_values,
    gui_model_values,
    model_id,
    model_label,
    model_selection_state,
)
from volModel.models import GUI_MODELS


def test_model_dropdown_uses_registered_gui_models():
    assert [model_id(label) for label in gui_model_values()] == list(GUI_MODELS)
    assert gui_model_values() == ["SVI", "SABR", "TPS"]


def test_model_display_labels_round_trip_to_backend_ids():
    assert model_label("svi") == "SVI"
    assert model_id("SVI") == "svi"
    assert model_id("sabr") == "sabr"


def test_feature_dropdown_uses_professional_display_labels():
    assert gui_feature_mode_values() == [
        "Term Structure",
        "Underlying Returns",
        "Surface",
        "Surface Grid",
    ]
    assert feature_mode_label("iv_atm") == "Term Structure"
    assert feature_mode_id("Term Structure") == "iv_atm"
    assert feature_mode_id("surface_grid") == "surface_grid"


def test_model_dropdown_enabled_for_smile_modes():
    assert model_selection_state("Smile (K/S vs IV)", "iv_atm") == "readonly"
    assert model_selection_state("Smile (K/S vs IV)", "surface") == "readonly"
    assert model_selection_state("smile", "iv_atm") == "readonly"
    assert model_selection_state("Peer Composite Surface", "surface") == "readonly"
    assert model_selection_state("synthetic_surface", "surface") == "readonly"


def test_model_dropdown_disabled_for_non_model_plots():
    assert model_selection_state("Term (ATM vs T)", "iv_atm") == "disabled"
    assert model_selection_state("Relative Weight Matrix", "surface") == "disabled"
