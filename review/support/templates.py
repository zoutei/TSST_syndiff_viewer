"""Workspace-scoped syndiff template and convolved-template path resolution."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

_SYNDIFF_TEMPLATE_RE = re.compile(
    r"^syndiff_template_s(?P<sector>\d+)_(?P<camera>\d+)_(?P<ccd>\d+)"
    r"(?P<roi>_x(?P<x0>\d+)-(?P<x1>\d+)_y(?P<y0>\d+)-(?P<y1>\d+))?"
    r"(?:_os\d+)?"
    r"_dx(?P<dx>[+-]?\d*\.?\d+)_dy(?P<dy>[+-]?\d*\.?\d+)\.fits(?:\.gz)?$",
    re.IGNORECASE,
)

_TARGETS_REG_CROP_RE = re.compile(
    r"FFI ROI origin x_min=(\d+) y_min=(\d+) size=(\d+)x(\d+)"
)


@dataclass(frozen=True)
class ParsedSyndiffTemplate:
    sector: int
    camera: int
    ccd: int
    x_min: Optional[int]
    x_max: Optional[int]
    y_min: Optional[int]
    y_max: Optional[int]
    dx: float
    dy: float
    path: str


def parse_syndiff_template_filename(path_or_basename: str) -> Optional[ParsedSyndiffTemplate]:
    """Parse a ``syndiff_template_*.fits`` basename or path; return None if no match."""
    name = Path(path_or_basename).name
    m = _SYNDIFF_TEMPLATE_RE.match(name)
    if not m:
        return None
    sec = int(m.group("sector"))
    cam = int(m.group("camera"))
    ccd = int(m.group("ccd"))
    if m.group("roi"):
        x0, x1 = int(m.group("x0")), int(m.group("x1"))
        y0, y1 = int(m.group("y0")), int(m.group("y1"))
    else:
        x0 = x1 = y0 = y1 = None
    dx = float(m.group("dx"))
    dy = float(m.group("dy"))
    p = Path(path_or_basename)
    path = str(p.resolve()) if p.is_file() else str(path_or_basename)
    return ParsedSyndiffTemplate(sec, cam, ccd, x0, x1, y0, y1, dx, dy, path)


def _offset_match(a: float, b: float, tol: float = 1e-3) -> bool:
    return abs(float(a) - float(b)) <= max(1e-5, tol)


def resolve_template_dir(fits_ws: Path) -> Path | None:
    """Resolve ``{fits_ws}/templates`` symlink or directory; None when missing."""
    link = fits_ws / "templates"
    if link.is_symlink() or link.is_dir():
        return link.resolve()
    return None


def find_template_by_offset(
    template_dir: Path,
    *,
    dx: float = 0.0,
    dy: float = 0.0,
    offset_tol: float = 1e-3,
) -> Path | None:
    """Find a syndiff template FITS with the requested (dx, dy) sub-pixel offset."""
    root = template_dir.expanduser().resolve()
    if not root.is_dir():
        return None

    matches: list[str] = []
    for full in sorted(root.iterdir()):
        if not full.is_file():
            continue
        parsed = parse_syndiff_template_filename(str(full))
        if parsed is None:
            continue
        if _offset_match(parsed.dx, dx, offset_tol) and _offset_match(parsed.dy, dy, offset_tol):
            matches.append(str(full.resolve()))

    if not matches:
        return None
    prefer_gz = [p for p in matches if p.lower().endswith(".fits.gz")]
    return Path(prefer_gz[0] if prefer_gz else matches[0])


def lookup_convolved_template(
    conv_dir: Path,
    group_dx: float,
    group_dy: float,
    *,
    tol: float = 1e-3,
) -> Path | None:
    """Return convolved template path for manifest group offsets via CSV manifest."""
    csv_path = conv_dir / "convolved_templates.csv"
    if not csv_path.is_file():
        return None
    table = pd.read_csv(csv_path)
    for _, row in table.iterrows():
        if abs(float(row["group_dx"]) - group_dx) <= tol and abs(
            float(row["group_dy"]) - group_dy
        ) <= tol:
            p = Path(str(row["convolved_path"]))
            if p.is_file():
                return p
    return None


def parse_crop_bounds_from_targets_reg(ws_dir: Path) -> dict | None:
    """Parse diff ROI crop bounds from ``targets.reg``; None when missing or unparsable."""
    path = ws_dir / "targets.reg"
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            m = _TARGETS_REG_CROP_RE.search(line)
            if not m:
                continue
            xm, ym, nx, ny = (int(m.group(i)) for i in range(1, 5))
            return {
                "x_min": xm,
                "x_max": xm + nx,
                "y_min": ym,
                "y_max": ym + ny,
                "shape": (ny, nx),
            }
    return None
