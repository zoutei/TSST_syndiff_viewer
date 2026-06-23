"""Compare-layer overlay helpers and plotting."""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import yaml

from review.config import ReviewConfig
from review.overlay_layers import (
    MAX_OVERLAY_LAYERS,
    PRIMARY_LAYER_KEY,
    append_layer_if_new,
    can_add_layer,
    has_layer_identity,
    layer_is_visible_for_plot,
    load_layer_index_payload,
    make_layer,
    point_from_plot_click,
    resolve_active_context,
    set_layer_offset,
)
from review.plot_lc import ACTIVE_PRIMARY_MARKER, PRIMARY_MARKER, add_syndiff_traces
from review.smoothing import apply_smoothing


def _write_compare_event(tmp: Path) -> Path:
    event = tmp / "events" / "s0023_test"
    for workspace in ("ws", "ws_alt"):
        ws = event / workspace / "hp_d"
        master = event / workspace / "master"
        lc_dir = event / workspace / "lc_prf_on_diffs"
        ws.mkdir(parents=True)
        master.mkdir(parents=True)
        lc_dir.mkdir(parents=True)

        diff_cfg = {
            "pipeline": [
                {
                    "kind": "hotpants",
                    "output": {"diffs": "hp_d", "convolved": "hp_c", "bkg": "hp_b"},
                    "write_bkg": True,
                    "write_convolved": False,
                },
                {"kind": "forced_photometry", "inputs": {"diffs": "hp_d"}, "output": "lc_prf_on_diffs"},
            ],
        }
        (event / workspace / "diff_config.yaml").write_text(yaml.dump(diff_cfg))

        flux = 10.0 if workspace == "ws" else 12.0
        lc = pd.DataFrame(
            [
                {
                    "btjd": 1928.94,
                    "flux": flux,
                    "eflux": 1.0,
                    "filename": str(ws / "tess2020019142923_hp_d.fits"),
                    "group_id": 0,
                }
            ]
        )
        lc.to_csv(lc_dir / "lightcurve.csv", index=False)

        diff_fits = ws / "tess2020019142923_hp_d.fits"
        diff_fits.write_bytes(b"SIMPLE  =                    T / syn diff test")
        (master / "tess2020019142923_hp_d.fits").symlink_to(diff_fits.resolve())
        (master / "tess2020019142923-s0023-1-3-0165-s_ffic.fits").symlink_to(diff_fits.resolve())

    manifest = pd.DataFrame(
        [
            {
                "filename": "tess2020019142923-s0023-1-3-0165-s_ffic.fits",
                "path": "/data/tess2020019142923-s0023-1-3-0165-s_ffic.fits",
                "group_id": 0,
                "group_dx": 0.0,
                "group_dy": 0.01,
                "hotpants_hp_d_ok": True,
            }
        ]
    )
    manifest.to_csv(event / "syndiff_ffi_frames.csv", index=False)
    return event


def test_append_layer_dedupes_identity():
    layers = append_layer_if_new([], workspace="ws", lc_dir="lc_prf_on_diffs", lc_name="primary")
    assert len(layers) == 1
    again = append_layer_if_new(layers, workspace="ws", lc_dir="lc_prf_on_diffs", lc_name="primary")
    assert again == layers


def test_can_add_layer_respects_cap():
    layers = [
        make_layer(workspace=f"ws{i}", lc_dir="lc_prf_on_diffs", lc_name="primary")
        for i in range(MAX_OVERLAY_LAYERS)
    ]
    assert not can_add_layer(layers)
    assert has_layer_identity(layers, ("ws0", "lc_prf_on_diffs", "primary"))


def test_layer_is_visible_for_plot_skips_primary_match():
    layer = make_layer(workspace="ws", lc_dir="lc_prf_on_diffs", lc_name="primary")
    primary = ("ws", "lc_prf_on_diffs", "primary")
    assert not layer_is_visible_for_plot(layer, primary)
    assert layer_is_visible_for_plot(layer, ("ws_alt", "lc_prf_on_diffs", "primary"))


def test_load_layer_index_payload(tmp_path):
    _write_compare_event(tmp_path)
    cfg = ReviewConfig(
        mount_root=str(tmp_path),
        cache_root=str(tmp_path),
        default_event="s0023_test",
        mount_root_strict=True,
    )
    layers = [
        make_layer(workspace="ws", lc_dir="lc_prf_on_diffs", lc_name="primary"),
        make_layer(workspace="ws_alt", lc_dir="lc_prf_on_diffs", lc_name="primary"),
    ]
    payload = load_layer_index_payload(cfg, "s0023_test", layers)
    assert set(payload) == {layer["id"] for layer in layers}
    assert payload[layers[0]["id"]]["epochs"]["flux"] == [10.0]
    assert payload[layers[1]["id"]]["epochs"]["flux"] == [12.0]
    assert payload[layers[0]["id"]]["workspace"] == "ws"


def test_resolve_active_context_uses_compare_layer():
    primary = {
        "workspace": "ws_alt",
        "lc_dir": "lc_prf_on_diffs",
        "lc_name": "primary",
        "epochs": {"epoch_idx": [0], "flux": [12.0]},
    }
    layer_id = "layer-abc"
    layer_index = {
        layer_id: {
            "workspace": "ws",
            "epochs": {"epoch_idx": [0], "flux": [10.0]},
        }
    }
    point = {"layer": layer_id, "epoch_idx": 0}
    store, epoch_idx = resolve_active_context(layer_id, primary, layer_index, point)
    assert store["workspace"] == "ws"
    assert epoch_idx == 0


def test_point_from_plot_click_layer_customdata():
    click = {"points": [{"customdata": ["layer-abc", 3]}]}
    assert point_from_plot_click(click) == {"layer": "layer-abc", "epoch_idx": 3}


def test_set_layer_offset():
    layer = make_layer(workspace="ws", lc_dir="lc_prf_on_diffs", lc_name="primary")
    updated = set_layer_offset([layer], layer["id"], 0.5)
    assert updated[0]["flux_offset"] == 0.5


def test_add_syndiff_traces_active_vs_inactive():
    df = pd.DataFrame(
        {
            "btjd": [1.0, 2.0],
            "flux": [10.0, 11.0],
            "eflux": [0.5, 0.5],
            "epoch_idx": [0, 1],
        }
    )
    smooth = apply_smoothing(df["btjd"], df["flux"], mode="none")
    fig = go.Figure()
    add_syndiff_traces(
        fig,
        df,
        layer_key=PRIMARY_LAYER_KEY,
        name="Syndiff",
        marker=PRIMARY_MARKER,
        show_errorbars=False,
        smooth=smooth,
        mode="none",
        show_diagnostics=False,
    )
    add_syndiff_traces(
        fig,
        df,
        layer_key="layer-1",
        name="ws / primary",
        marker=ACTIVE_PRIMARY_MARKER,
        show_errorbars=False,
        smooth=smooth,
        mode="none",
        selected_epoch=1,
        show_diagnostics=True,
        color_index=0,
    )
    names = [trace.name for trace in fig.data]
    assert "Syndiff" in names
    assert "ws / primary" in names
    assert "selected" in names
    assert fig.data[0].customdata[0] == [PRIMARY_LAYER_KEY, 0]
