"""Plot click selection helpers."""

from review.app import _epoch_from_plot_click


def test_epoch_from_flux_marker_click():
    click = {"points": [{"customdata": 42, "curveNumber": 1}]}
    assert _epoch_from_plot_click(click) == 42


def test_deselect_on_background_click():
    assert _epoch_from_plot_click(None) is None
    assert _epoch_from_plot_click({"points": []}) is None


def test_deselect_on_overlay_click():
    click = {"points": [{"x": 2450.5, "y": 1.0, "curveNumber": 3}]}
    assert _epoch_from_plot_click(click) is None
