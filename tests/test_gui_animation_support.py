import matplotlib
matplotlib.use("Agg")

from display.gui.gui_plot_manager import PlotManager


def test_plot_manager_animation_support_removed():
    mgr = PlotManager()
    removed_methods = [
        "has_animation_support",
        "plot_animated",
        "start_animation",
        "stop_animation",
        "pause_animation",
        "set_animation_speed",
        "is_animation_active",
    ]

    for method in removed_methods:
        assert not hasattr(mgr, method)
