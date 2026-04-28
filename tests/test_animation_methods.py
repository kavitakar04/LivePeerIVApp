"""Regression guard that animation playback APIs stay removed."""

from display.gui.controllers.gui_plot_manager import PlotManager


def test_animation_methods_removed():
    mgr = PlotManager()
    methods = [
        "has_animation_support",
        "plot_animated",
        "start_animation",
        "stop_animation",
        "pause_animation",
        "set_animation_speed",
        "is_animation_active",
    ]

    for method in methods:
        assert not hasattr(mgr, method)
