"""Plot click selection helpers."""

import pandas as pd

from review.app import _epoch_from_plot_click, find_epoch_idx_by_product_id
from review.overlay_layers import PRIMARY_LAYER_KEY, point_from_plot_click


def test_epoch_from_flux_marker_click():
    click = {"points": [{"customdata": [PRIMARY_LAYER_KEY, 42], "curveNumber": 1}]}
    assert point_from_plot_click(click) == {"layer": PRIMARY_LAYER_KEY, "epoch_idx": 42}
    assert _epoch_from_plot_click(click) == 42


def test_epoch_from_legacy_primary_customdata():
    click = {"points": [{"customdata": 42, "curveNumber": 1}]}
    assert _epoch_from_plot_click(click) == 42


def test_deselect_on_background_click():
    assert _epoch_from_plot_click(None) is None
    assert _epoch_from_plot_click({"points": []}) is None


def test_deselect_on_overlay_click():
    click = {"points": [{"x": 2450.5, "y": 1.0, "curveNumber": 3}]}
    assert _epoch_from_plot_click(click) is None


def test_find_epoch_idx_by_product_id_exact():
    epochs = pd.DataFrame(
        {
            "epoch_idx": [0, 1],
            "product_id": ["tess2020019142923", "tess2020019142924"],
        }
    )
    idx, msg = find_epoch_idx_by_product_id(epochs, "tess2020019142923")
    assert idx == 0
    assert msg == ""


def test_find_epoch_idx_by_product_id_partial_unique():
    epochs = pd.DataFrame(
        {
            "epoch_idx": [0, 1],
            "product_id": ["tess2020019142923", "tess2020019142924"],
        }
    )
    idx, msg = find_epoch_idx_by_product_id(epochs, "9142924")
    assert idx == 1
    assert msg == ""


def test_find_epoch_idx_by_product_id_missing():
    epochs = pd.DataFrame({"epoch_idx": [0], "product_id": ["tess2020019142923"]})
    idx, msg = find_epoch_idx_by_product_id(epochs, "tess9999")
    assert idx is None
    assert "No epoch" in msg
