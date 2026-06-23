import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from astropy.io import fits

from review.event_index import (
    EventIndex,
    clear_index_cache,
    epoch_file_exists,
    get_master_index,
    get_workspace_context,
    master_index_is_cached,
)
from review.pipeline_labels import list_lightcurve_options, parse_diff_config


def _write_minimal_event(tmp: Path) -> Path:
    event = tmp / "s0023_test"
    ws = event / "ws" / "hp_d"
    master = event / "ws" / "master"
    lc_dir = event / "ws" / "lc_prf_on_diffs"
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
        "additional_forced_targets": [{"name": "offset_top", "position_mode": "offset", "dx": 0, "dy": -7}],
    }
    (event / "ws" / "diff_config.yaml").write_text(yaml.dump(diff_cfg))

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

    lc = pd.DataFrame(
        [
            {
                "btjd": 1928.94,
                "flux": 10.0,
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

    return event


def test_parse_diff_config(tmp_path):
    event = _write_minimal_event(tmp_path)
    labels = parse_diff_config(event / "ws" / "diff_config.yaml")
    assert labels.diff_label == "hp_d"
    assert labels.lc_dir == "lc_prf_on_diffs"
    assert labels.additional_targets == ["offset_top"]
    options = dict(list_lightcurve_options(labels))
    assert options["offset_top"] == "lightcurve_offset_top.csv"


def test_event_index_resolves_master_paths(tmp_path):
    event = _write_minimal_event(tmp_path)
    idx = EventIndex.load(event)
    assert len(idx.epochs) == 1
    row = idx.epochs.iloc[0]
    assert row["product_id"] == "tess2020019142923"
    assert epoch_file_exists(row)["diff_exists"]
    assert "tess2020019142923_hp_d.fits" in row["diff_path"]
    assert row["group_dx"] == 0.0
    assert row["group_dy"] == 0.01


def test_event_index_resolves_template_from_ws_templates(tmp_path):
    event = _write_minimal_event(tmp_path)
    physical = tmp_path / "template_data"
    physical.mkdir()
    tmpl_name = "syndiff_template_s0023_1_3_dx0.000_dy0.010.fits"
    data = np.zeros((10, 10), dtype=np.float32)
    fits.PrimaryHDU(data=data).writeto(physical / tmpl_name, overwrite=True)
    (event / "ws" / "templates").symlink_to(physical)

    idx = EventIndex.load(event)
    row = idx.epochs.iloc[0]
    assert row["template_path"] is not None
    assert tmpl_name in row["template_path"]
    assert idx.template_dir == physical.resolve()


def test_master_index_cache_reuses_scan(tmp_path):
    clear_index_cache()
    event = _write_minimal_event(tmp_path)
    fits_event = event.resolve()
    assert not master_index_is_cached(fits_event, "ws")
    first = get_master_index(fits_event, "ws")
    assert master_index_is_cached(fits_event, "ws")
    second = get_master_index(fits_event, "ws")
    assert first is second
    clear_index_cache()
    assert not master_index_is_cached(fits_event, "ws")


def test_target_swap_reuses_epoch_paths(tmp_path):
    clear_index_cache()
    event = _write_minimal_event(tmp_path)
    idx_primary = EventIndex.load(event, lc_name="primary")
    idx_primary_again = EventIndex.load(event, lc_name="primary")
    assert len(idx_primary.epochs) == len(idx_primary_again.epochs)
    assert idx_primary.epochs.iloc[0]["flux"] == idx_primary_again.epochs.iloc[0]["flux"]
    clear_index_cache()


def test_workspace_context_reused_across_targets(tmp_path):
    clear_index_cache()
    event = _write_minimal_event(tmp_path)
    ctx1 = get_workspace_context(event, event, "ws")
    ctx2 = get_workspace_context(event, event, "ws")
    assert ctx1 is ctx2
    EventIndex.load(event, lc_name="primary")
    EventIndex.load(event, lc_name="primary")
    ctx3 = get_workspace_context(event, event, "ws")
    assert ctx3 is ctx1
    clear_index_cache()


def _write_kernel_subtract_hotpants_event(tmp: Path) -> Path:
    event = tmp / "s0020_test"
    mk_ws = event / "ws" / "mk_d"
    ks_ws = event / "ws" / "ks_d"
    kb_ws = event / "ws" / "mk_b"
    ks_b_ws = event / "ws" / "ks_b"
    master = event / "ws" / "master"
    lc_dir = event / "ws" / "lc_prf_on_mk_diffs"
    for path in (mk_ws, ks_ws, kb_ws, ks_b_ws, master, lc_dir):
        path.mkdir(parents=True)

    diff_cfg = yaml.safe_load(
        textwrap.dedent(
            """
            pipeline:
              - kind: kernel_subtract
                output:
                  diffs: ks_d
                  phot_bkg: ks_b
              - kind: hotpants
                write_convolved: false
                write_bkg: true
                output:
                  diffs: mk_d
                  convolved: mk_c
                  bkg: mk_b
              - kind: forced_photometry
                inputs:
                  diffs: mk_d
                output: lc_prf_on_mk_diffs
            """
        )
    )
    (event / "ws" / "diff_config.yaml").write_text(yaml.dump(diff_cfg))

    manifest = pd.DataFrame(
        [
            {
                "filename": "tess2020019142923-s0023-1-3-0165-s_ffic.fits",
                "path": "/data/tess2020019142923-s0023-1-3-0165-s_ffic.fits",
                "group_id": 0,
                "group_dx": 0.0,
                "group_dy": 0.01,
                "hotpants_mk_d_ok": True,
            }
        ]
    )
    manifest.to_csv(event / "syndiff_ffi_frames.csv", index=False)

    lc = pd.DataFrame(
        [
            {
                "btjd": 1928.94,
                "flux": 10.0,
                "eflux": 1.0,
                "filename": str(mk_ws / "tess2020019142923_mk_d.fits"),
                "group_id": 0,
            }
        ]
    )
    lc.to_csv(lc_dir / "lightcurve.csv", index=False)

    for label, directory in (
        ("mk_d", mk_ws),
        ("ks_d", ks_ws),
        ("mk_b", kb_ws),
        ("ks_b", ks_b_ws),
    ):
        fname = f"tess2020019142923_{label}.fits"
        fits_path = directory / fname
        fits_path.write_bytes(b"SIMPLE  =                    T / syn diff test")
        (master / fname).symlink_to(fits_path.resolve())

    return event


def test_event_index_multi_stage_products(tmp_path):
    clear_index_cache()
    event = _write_kernel_subtract_hotpants_event(tmp_path)
    idx = EventIndex.load(event, lc_dir="lc_prf_on_mk_diffs")
    row = idx.epochs.iloc[0]
    assert "tess2020019142923_mk_d.fits" in row["diff_path"]
    products = {product["key"]: product["path"] for product in row["products"]}
    assert products["mk_d"] is not None and "mk_d" in products["mk_d"]
    assert products["ks_d"] is not None and "ks_d" in products["ks_d"]
    assert products["mk_b"] is not None and "mk_b" in products["mk_b"]
    assert products["ks_b"] is not None and "ks_b" in products["ks_b"]
    assert idx.kernel_workspace_paths()["epoch_products"][0]["key"] == "mk_d"
    clear_index_cache()
