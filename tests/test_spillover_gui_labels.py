from display.gui.spillover_gui import (
    EVENT_PLOT_NOTE,
    EVENT_RESPONSE_Y_LABEL,
    EVENT_RESPONSE_TITLE_SUFFIX,
    MEDIAN_RESPONSE_LABEL,
    ROLLING_SIGNAL_ABNORMAL_LABEL,
    ROLLING_SIGNAL_DIRECTION_LABEL,
    ROLLING_SIGNAL_TITLE_SUFFIX,
    ROLLING_SIGNAL_WINDOW_EVENTS,
    RESPONSE_PLOT_LABEL,
    RESPONSE_PLOT_TITLE,
    RESPONSE_Y_LABEL,
    SPILLOVER_EXPLANATION,
    SPILLOVER_SUMMARY_NOTE,
    SUMMARY_HEADINGS,
)


def test_spillover_summary_labels_define_response_metrics():
    assert SUMMARY_HEADINGS["resp"] == "Median peer response (%)"
    assert SUMMARY_HEADINGS["abn"] == "Abnormal vs baseline (%)"
    assert SUMMARY_HEADINGS["ci"] == "Median response 95% CI"
    assert "selected series" in SPILLOVER_EXPLANATION
    assert "prior trading day" in SPILLOVER_EXPLANATION
    assert "event date + horizon" in SPILLOVER_EXPLANATION
    assert "minus the same pair's baseline median" in SPILLOVER_SUMMARY_NOTE


def test_spillover_plot_labels_are_not_event_level_averages():
    assert RESPONSE_PLOT_LABEL == "Avg pair median response"
    assert RESPONSE_PLOT_TITLE == "Average of pair-level median responses by horizon"
    assert RESPONSE_Y_LABEL == "Peer response (%)"
    assert EVENT_RESPONSE_Y_LABEL == "Response (%)"
    assert EVENT_RESPONSE_TITLE_SUFFIX == "trigger and peer responses"
    assert "one selected event trajectory" in EVENT_PLOT_NOTE
    assert "medians across all events" in EVENT_PLOT_NOTE
    assert MEDIAN_RESPONSE_LABEL == "All-event median response"
    assert ROLLING_SIGNAL_WINDOW_EVENTS == 30
    assert ROLLING_SIGNAL_TITLE_SUFFIX == "rolling spillover signal"
    assert ROLLING_SIGNAL_ABNORMAL_LABEL == "Rolling abnormal response"
    assert ROLLING_SIGNAL_DIRECTION_LABEL == "Rolling same-direction probability"
