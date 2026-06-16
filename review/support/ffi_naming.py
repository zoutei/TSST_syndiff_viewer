"""Vendored FFI product-id helpers (mirrors syndiff_pipeline ffi_naming)."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional, Tuple

_TESS_PRODUCT_ID_RE = re.compile(r"^(tess\d+)", re.IGNORECASE)
_WORKSPACE_FRAME_STEM_RE = re.compile(r"^(tess\d+)_(.+)$", re.IGNORECASE)


def tess_product_id_from_ffi_path(path_or_basename: str) -> Optional[str]:
    name = Path(str(path_or_basename)).name
    stem = name[:-5] if name.lower().endswith(".fits") else os.path.splitext(name)[0]
    m = _TESS_PRODUCT_ID_RE.match(stem)
    return m.group(1) if m else None


def sanitize_workspace_label(label: str) -> str:
    return str(label).replace(" ", "_")


def workspace_frame_stem(product_id: str, label: str) -> str:
    return f"{product_id}_{sanitize_workspace_label(label)}"


def parse_workspace_frame_stem(frame_stem: str) -> Optional[Tuple[str, str]]:
    m = _WORKSPACE_FRAME_STEM_RE.match(str(frame_stem))
    if not m:
        return None
    return m.group(1), m.group(2)
