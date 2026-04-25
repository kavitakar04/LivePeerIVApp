import matplotlib
matplotlib.use("Agg")

from display.gui.gui_plot_manager import PlotManager


def test_has_animation_support_smile_and_surface():
    mgr = PlotManager()
    # Labels are accepted directly; plot_id() is applied internally
    assert mgr.has_animation_support("Smile (K/S vs IV)")
    assert mgr.has_animation_support("Synthetic Surface (Smile)")
    assert not mgr.has_animation_support("Term (ATM vs T)")
    # Raw IDs also work
    assert mgr.has_animation_support("smile")
    assert mgr.has_animation_support("synthetic_surface")
    assert not mgr.has_animation_support("term")
