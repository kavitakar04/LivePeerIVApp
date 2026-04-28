import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from display.plotting.utils.legend_utils import add_legend_toggles


def test_legend_toggle_single_connection():
    fig, ax = plt.subplots()
    (line,) = ax.plot([0, 1], [0, 1], label="Series")
    series = {"Series": [line]}

    leg1 = add_legend_toggles(ax, series)
    add_legend_toggles(ax, series)

    handle = leg1.legend_handles[0] if hasattr(leg1, "legend_handles") else leg1.legendHandles[0]
    event = type("E", (), {"artist": handle, "canvas": fig.canvas, "name": "pick_event"})()
    fig.canvas.callbacks.process("pick_event", event)
    assert series["Series"][0].get_visible() is False
    plt.close(fig)


def test_legend_toggle_after_axes_clear():
    fig, ax = plt.subplots()
    (line,) = ax.plot([0, 1], [0, 1], label="Series")
    series = {"Series": [line]}

    add_legend_toggles(ax, series)
    ax.cla()
    (line2,) = ax.plot([0, 1], [1, 0], label="Series2")
    series2 = {"Series2": [line2]}

    add_legend_toggles(ax, series2)
    assert hasattr(ax, "_legend_toggle_text")
    plt.close(fig)
