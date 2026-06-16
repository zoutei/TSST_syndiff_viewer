"""Template FITS coverage in FFI pixel coordinates (vendored from syndiff_pipeline)."""

from __future__ import annotations

from astropy.io import fits


def template_coverage_ffi_bounds(tmpl_path: str) -> dict:
    """Return FFI-coordinate bounds covered by a syndiff template FITS."""
    with fits.open(tmpl_path, memmap=True) as hdul:
        if hdul[0].data is not None:
            data = hdul[0].data
            hdr = hdul[0].header
        else:
            data = hdul[1].data
            hdr = hdul[1].header
        ny, nx = data.shape

    if all(k in hdr for k in ("XMIN", "XMAX", "YMIN", "YMAX")):
        return {
            "x_min": int(hdr["XMIN"]),
            "x_max": int(hdr["XMAX"]),
            "y_min": int(hdr["YMIN"]),
            "y_max": int(hdr["YMAX"]),
            "shape": (int(hdr["YMAX"]) - int(hdr["YMIN"]), int(hdr["XMAX"]) - int(hdr["XMIN"])),
        }
    return {
        "x_min": 0,
        "x_max": nx,
        "y_min": 0,
        "y_max": ny,
        "shape": (ny, nx),
    }


def crop_bounds_subset_of_coverage(crop_bounds: dict, coverage: dict) -> bool:
    """True when *crop_bounds* lies inside template *coverage* (FFI coords)."""
    return (
        crop_bounds["x_min"] >= coverage["x_min"]
        and crop_bounds["y_min"] >= coverage["y_min"]
        and crop_bounds["x_max"] <= coverage["x_max"]
        and crop_bounds["y_max"] <= coverage["y_max"]
    )
