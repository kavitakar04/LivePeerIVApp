"""Utilities for interactive legend toggling."""

from typing import Dict, List

import matplotlib.pyplot as plt
from matplotlib.legend import Legend


def add_legend_toggles(ax: plt.Axes, series_map: Dict[str, List[plt.Artist]]) -> Legend:
    """Make legend entries clickable to toggle series with visual feedback."""
    leg = ax.get_legend()
    if leg is None:
        handles, labels = ax.get_legend_handles_labels()
        if len(labels) >= 5:
            if len(ax.figure.axes) == 1:
                ax.figure.subplots_adjust(right=0.78)
            leg = ax.legend(
                loc="upper left",
                bbox_to_anchor=(1.01, 1.0),
                borderaxespad=0.0,
                fontsize=8,
                framealpha=0.92,
            )
        else:
            leg = ax.legend(loc="upper right", fontsize=8, framealpha=0.92)
    fig = ax.figure
    handles = leg.legend_handles if hasattr(leg, "legend_handles") else leg.legendHandles
    texts = leg.get_texts()

    # Make legend entries pickable.
    for handle in handles:
        handle.set_picker(True)
        if hasattr(handle, "set_pickradius"):
            handle.set_pickradius(15)
    for text in texts:
        text.set_picker(True)
        if hasattr(text, "set_pickradius"):
            text.set_pickradius(15)

    def on_pick(event):
        artist = event.artist
        if hasattr(artist, "get_label"):
            label_text = artist.get_label()
        else:
            try:
                text_index = texts.index(artist)
                label_text = texts[text_index].get_text()
            except (ValueError, IndexError):
                return

        matched_key = None
        for key in series_map.keys():
            if key == label_text or label_text in key or key in label_text:
                matched_key = key
                break
        if not matched_key:
            label_words = label_text.lower().split()
            for key in series_map.keys():
                key_words = key.lower().split()
                if any(word in key_words for word in label_words):
                    matched_key = key
                    break
                if ("ci" in label_text.lower() or "%" in label_text) and "confidence" in key.lower():
                    matched_key = key
                    break
                if "fit" in label_text.lower() and "fit" in key.lower():
                    matched_key = key
                    break
        if not matched_key:
            print(f"Warning: Could not find series for legend label '{label_text}'")
            print(f"Available series: {list(series_map.keys())}")
            return

        arts = series_map[matched_key]
        if not arts:
            return
        new_visible = not arts[0].get_visible()
        for art in arts:
            art.set_visible(new_visible)

        for i, handle in enumerate(handles):
            if (hasattr(handle, "get_label") and handle.get_label() == label_text) or (
                i < len(texts) and texts[i].get_text() == label_text
            ):
                handle.set_alpha(1.0 if new_visible else 0.3)
                if i < len(texts):
                    texts[i].set_alpha(1.0 if new_visible else 0.5)
                    texts[i].set_weight("normal" if new_visible else "normal")
                    texts[i].set_style("normal" if new_visible else "italic")
                break

        fig.canvas.draw_idle()

    if hasattr(ax, "_legend_toggle_cid"):
        fig.canvas.mpl_disconnect(ax._legend_toggle_cid)

    ax._legend_toggle_cid = fig.canvas.mpl_connect("pick_event", on_pick)

    if hasattr(ax, "_legend_toggle_text"):
        text = ax._legend_toggle_text
        if getattr(text, "figure", None) is not None and getattr(text, "axes", None) is not None:
            try:
                text.remove()
            except (ValueError, NotImplementedError, RuntimeError):
                pass

    ax._legend_toggle_text = ax.text(
        0.02,
        0.98,
        "Click legend entries to toggle visibility",
        transform=ax.transAxes,
        fontsize=8,
        alpha=0.7,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", alpha=0.1),
    )

    return leg
