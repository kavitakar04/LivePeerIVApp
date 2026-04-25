from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt
from typing import Mapping, Sequence


def plot_weights(
    ax: plt.Axes,
    weights: Mapping[str, float] | Sequence[float] | pd.Series,
    *,
    raw_scores: Mapping[str, float] | pd.Series | None = None,
) -> None:
    """Render weights as a sorted bar chart.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axis to draw on. It will be cleared.
    weights : mapping or sequence or pd.Series
        Mapping/series of label -> weight values or sequence of weights (labels
        will be numeric indices).
    """
    ax.clear()
    if weights is None:
        ax.text(0.5, 0.5, "No weights", ha="center", va="center")
        return

    if isinstance(weights, pd.Series):
        s = weights.astype(float)
    elif isinstance(weights, Mapping):
        s = pd.Series(dict(weights), dtype=float)
    else:
        s = pd.Series(list(weights), dtype=float)

    if s.empty:
        ax.text(0.5, 0.5, "No weights", ha="center", va="center")
        return

    s = s.sort_values(ascending=False)
    x = range(len(s))
    bars = ax.bar(x, s.values, color="steelblue")
    ax.set_title("ETF Weights")
    ax.set_ylabel("Weight")
    ax.set_ylim(0, max(float(s.max()) * 1.18, 0.05))
    ax.set_xticks(list(x))
    ax.set_xticklabels(s.index.astype(str), rotation=45, ha="right")

    raw = pd.Series(raw_scores, dtype=float) if raw_scores is not None else pd.Series(dtype=float)
    for bar, ticker, val in zip(bars, s.index, s.values):
        label = f"{val:.3f}"
        if ticker in raw.index and pd.notna(raw.loc[ticker]):
            label = f"{val:.3f}\nscore {raw.loc[ticker]:.2f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.grid(axis="y", alpha=0.2)
