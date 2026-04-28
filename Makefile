.PHONY: gui gui-help test

PYTHON ?= ./venv/bin/python

gui:
	@env MPLCONFIGDIR=/tmp/mplconfig $(PYTHON) -m display.gui.app.browser

gui-help:
	@env MPLCONFIGDIR=/tmp/mplconfig $(PYTHON) -m display.gui.app.browser --help

test:
	@env MPLCONFIGDIR=/tmp/mplconfig PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $(PYTHON) -m pytest
