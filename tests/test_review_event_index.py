import textwrap
from pathlib import Path

import pandas as pd
import yaml

from review.event_index import EventIndex, epoch_file_exists
from review.pipeline_labels import parse_diff_config


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


def test_event_index_resolves_master_paths(tmp_path):
    event = _write_minimal_event(tmp_path)
    idx = EventIndex.load(event)
    assert len(idx.epochs) == 1
    row = idx.epochs.iloc[0]
    assert row["product_id"] == "tess2020019142923"
    assert epoch_file_exists(row)["diff_exists"]
    assert "tess2020019142923_hp_d.fits" in row["diff_path"]
