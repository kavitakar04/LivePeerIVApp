from display.gui.app.browser import BROWSER_TAB_LABELS
from display.gui.controls.gui_input import PLOT_TYPES


def test_browser_combines_settings_and_health_tab():
    assert BROWSER_TAB_LABELS == (
        "IV Explorer",
        "Settings / Data & Model Health",
        "Parameter Summary",
        "Spillover",
        "RV Signals",
    )
    assert "Data & Model Health" not in BROWSER_TAB_LABELS


def test_rv_signals_is_dedicated_tab_not_explorer_plot():
    assert "RV Signals" in BROWSER_TAB_LABELS
    assert "RV Signals" not in PLOT_TYPES


def test_peer_composite_weights_is_not_explorer_plot():
    assert "Peer Composite Weights" not in PLOT_TYPES
