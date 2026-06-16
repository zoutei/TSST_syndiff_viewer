import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from astropy.io import fits

from review.crop_cache import ensure_cropped_fits
from review.pipeline_labels import list_lightcurve_options, parse_diff_config
from review.support.templates import (
    find_template_by_offset,
    lookup_convolved_template,
    parse_crop_bounds_from_targets_reg,
    parse_syndiff_template_filename,
)


def test_offset_lightcurve_filename_not_doubled(tmp_path):
    cfg = {
        "pipeline": [
            {"kind": "hotpants", "output": {"diffs": "hp_d"}},
            {"kind": "forced_photometry", "inputs": {"diffs": "hp_d"}, "output": "lc_prf_on_diffs"},
        ],
        "additional_forced_targets": [
            {"name": "offset_top", "position_mode": "offset", "dx": 0, "dy": -7},
        ],
    }
    path = tmp_path / "diff_config.yaml"
    path.write_text(yaml.dump(cfg))
    labels = parse_diff_config(path)
    options = dict(list_lightcurve_options(labels))
    assert options["offset_top"] == "lightcurve_offset_top.csv"
    assert "offset_offset_top" not in options


def test_parse_syndiff_template_filename():
    name = "syndiff_template_s0023_1_3_dx0.000_dy0.010.fits"
    parsed = parse_syndiff_template_filename(name)
    assert parsed is not None
    assert parsed.sector == 23
    assert parsed.dx == 0.0
    assert parsed.dy == 0.01


def test_find_template_by_offset(tmp_path):
    tmpl_dir = tmp_path / "templates"
    tmpl_dir.mkdir()
    fname = "syndiff_template_s0023_1_3_dx0.000_dy0.010.fits"
    data = np.zeros((10, 10), dtype=np.float32)
    fits.PrimaryHDU(data=data).writeto(tmpl_dir / fname, overwrite=True)
    hit = find_template_by_offset(tmpl_dir, dx=0.0, dy=0.01)
    assert hit is not None
    assert hit.name == fname


def test_lookup_convolved_template(tmp_path):
    conv_dir = tmp_path / "tmpl_conv"
    conv_dir.mkdir()
    conv_path = conv_dir / "convolved_template_dx0.000_dy0.010.fits"
    conv_path.write_bytes(b"SIMPLE  =                    T")
    table = pd.DataFrame(
        [
            {
                "group_id": 0,
                "group_dx": 0.0,
                "group_dy": 0.01,
                "template_path": "/data/tmpl.fits",
                "convolved_path": str(conv_path),
            }
        ]
    )
    table.to_csv(conv_dir / "convolved_templates.csv", index=False)
    hit = lookup_convolved_template(conv_dir, 0.0, 0.01)
    assert hit == conv_path


def test_parse_crop_bounds_from_targets_reg(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "targets.reg").write_text(
        textwrap.dedent(
            """\
            # Region file format: DS9 version 4.1
            # FFI ROI origin x_min=100 y_min=200 size=64x32
            """
        )
    )
    bounds = parse_crop_bounds_from_targets_reg(ws)
    assert bounds == {"x_min": 100, "x_max": 164, "y_min": 200, "y_max": 232, "shape": (32, 64)}


def _write_ffi(path: Path, shape: tuple[int, int] = (300, 300)) -> None:
    ny, nx = shape
    data = np.zeros((ny, nx), dtype=np.float32)
    phdu = fits.PrimaryHDU()
    ihdu = fits.ImageHDU(data=data)
    fits.HDUList([phdu, ihdu]).writeto(path, overwrite=True)


def test_crop_cache_creates_and_reuses(tmp_path):
    src = tmp_path / "ffi.fits"
    _write_ffi(src)
    bounds = {"x_min": 10, "x_max": 20, "y_min": 5, "y_max": 15, "shape": (10, 10)}
    cache_root = tmp_path / "cache"
    out1 = ensure_cropped_fits(
        src,
        kind="ffi",
        crop_bounds=bounds,
        cache_root=cache_root,
        event_key="evt",
        workspace="ws",
    )
    assert out1.is_file()
    with fits.open(out1) as hdul:
        assert hdul[0].data.shape == (10, 10)
    mtime = out1.stat().st_mtime_ns
    out2 = ensure_cropped_fits(
        src,
        kind="ffi",
        crop_bounds=bounds,
        cache_root=cache_root,
        event_key="evt",
        workspace="ws",
    )
    assert out2 == out1
    assert out2.stat().st_mtime_ns == mtime
