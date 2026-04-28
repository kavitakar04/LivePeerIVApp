# Display Package Layout

The display package separates GUI orchestration from chart rendering while
keeping implementation modules grouped by responsibility.

## Public Modules

GUI code should import from the canonical implementation packages below.
Plotting code continues to expose chart renderers from `display.plotting`.

## Implementation Packages

- `gui/app/`: top-level application shell and browser frame.
- `gui/controls/`: input widgets, settings persistence, and option labels.
- `gui/controllers/`: GUI dispatch and plot orchestration.
- `gui/panels/`: secondary tabs and panels such as parameters, RV signals, and
  spillover views.
- `plotting/charts/`: Matplotlib chart renderers for smile, term, correlation,
  RV heatmaps, and peer-composite views.
- `plotting/utils/`: shared plotting utilities.

## Import Rule

Import implementation modules directly, for example:

```python
from display.gui.controllers.gui_plot_manager import PlotManager
from display.gui.controls.gui_input import InputPanel
from display.plotting.charts.smile_plot import fit_and_plot_smile
```
