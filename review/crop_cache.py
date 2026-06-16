"""On-demand FITS cropping with a local disk cache for DS9 display."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Literal

import numpy as np
from astropy.io import fits

from review.support.template_coverage import (
    crop_bounds_subset_of_coverage,
    template_coverage_ffi_bounds,
)

log = logging.getLogger(__name__)

CropKind = Literal["ffi", "template"]


def _cache_key(source: Path, crop_bounds: dict, kind: CropKind) -> str:
    try:
        mtime = source.stat().st_mtime_ns
    except OSError:
        mtime = 0
    payload = json.dumps(
        {
            "source": str(source.resolve()),
            "mtime_ns": mtime,
            "kind": kind,
            "bounds": {k: crop_bounds[k] for k in ("x_min", "x_max", "y_min", "y_max")},
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def _crop_ffi_array(ffi_path: Path, bounds: dict) -> np.ndarray:
    x0, x1 = bounds["x_min"], bounds["x_max"]
    y0, y1 = bounds["y_min"], bounds["y_max"]
    with fits.open(ffi_path, memmap=True) as hdul:
        return hdul[1].data[y0:y1, x0:x1].astype(np.float64)


def _crop_template_array(tmpl_path: Path, bounds: dict) -> np.ndarray:
    coverage = template_coverage_ffi_bounds(str(tmpl_path))
    if not crop_bounds_subset_of_coverage(bounds, coverage):
        raise ValueError(
            f"Diff crop {bounds} extends outside template coverage {coverage} for {tmpl_path}"
        )
    ox = coverage["x_min"]
    oy = coverage["y_min"]
    x0, x1 = bounds["x_min"] - ox, bounds["x_max"] - ox
    y0, y1 = bounds["y_min"] - oy, bounds["y_max"] - oy
    with fits.open(tmpl_path, memmap=True) as hdul:
        if hdul[0].data is not None:
            return hdul[0].data[y0:y1, x0:x1].astype(np.float64)
        return hdul[1].data[y0:y1, x0:x1].astype(np.float64)


def _write_cropped_fits(out_path: Path, data: np.ndarray, header_source: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    hdr = fits.Header()
    try:
        with fits.open(header_source, memmap=True) as hdul:
            src_hdr = hdul[1].header if len(hdul) > 1 else hdul[0].header
            for key in ("CTYPE1", "CTYPE2", "CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2", "CD1_1", "CD1_2", "CD2_1", "CD2_2"):
                if key in src_hdr:
                    hdr[key] = src_hdr[key]
    except Exception:
        pass
    hdr["NAXIS"] = 2
    hdr["NAXIS1"] = data.shape[1]
    hdr["NAXIS2"] = data.shape[0]
    hdu = fits.PrimaryHDU(data=data.astype(np.float64), header=hdr)
    hdu.writeto(out_path, overwrite=True)


def ensure_cropped_fits(
    source: str | Path,
    *,
    kind: CropKind,
    crop_bounds: dict,
    cache_root: Path,
    event_key: str,
    workspace: str,
) -> Path:
    """Return cached cropped FITS path; create or refresh when source mtime changes."""
    src = Path(source).expanduser()
    resolved = src.resolve()
    cache_dir = cache_root / "crops" / event_key / workspace
    out_path = cache_dir / f"{_cache_key(resolved, crop_bounds, kind)}_{kind}.fits"

    if out_path.is_file():
        try:
            if out_path.stat().st_mtime_ns >= resolved.stat().st_mtime_ns:
                return out_path
        except OSError:
            pass

    if kind == "ffi":
        data = _crop_ffi_array(resolved, crop_bounds)
    else:
        data = _crop_template_array(resolved, crop_bounds)

    _write_cropped_fits(out_path, data, resolved)
    log.info("Wrote cropped %s cache: %s", kind, out_path)
    return out_path
