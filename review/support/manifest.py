"""Vendored manifest loader (mirrors syndiff_pipeline manifest)."""

from __future__ import annotations

import os

import pandas as pd

from .paths import DEFAULT_MANIFEST_BASENAME


def load_frame_manifest(output_dir: str, manifest_path: str | None = None) -> pd.DataFrame:
    path = manifest_path or os.path.join(os.path.abspath(output_dir), DEFAULT_MANIFEST_BASENAME)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing frame manifest {path!r}")
    return pd.read_csv(path)
