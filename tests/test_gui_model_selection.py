from display.gui.gui_input import gui_model_values, model_selection_state
from volModel.models import GUI_MODELS


def test_model_dropdown_uses_registered_gui_models():
    assert gui_model_values() == list(GUI_MODELS)
    assert gui_model_values() == ["svi", "sabr", "tps"]


def test_model_dropdown_enabled_for_smile_modes():
    assert model_selection_state("Smile (K/S vs IV)", "iv_atm") == "readonly"
    assert model_selection_state("Smile (K/S vs IV)", "surface") == "readonly"
    assert model_selection_state("smile", "iv_atm") == "readonly"
    assert model_selection_state("Peer Composite Surface", "surface") == "readonly"
    assert model_selection_state("synthetic_surface", "surface") == "readonly"


def test_model_dropdown_disabled_for_non_model_plots():
    assert model_selection_state("Term (ATM vs T)", "iv_atm") == "disabled"
    assert model_selection_state("Relative Weight Matrix", "surface") == "disabled"
