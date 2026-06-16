"""Vendored workspace path constants (mirrors syndiff_pipeline paths)."""

from __future__ import annotations

import os

WORKSPACE_SUBDIR = "ws"
MASTER_SUBDIR = "master"
DEFAULT_MANIFEST_BASENAME = "syndiff_ffi_frames.csv"
TARGETS_DS9_REGION_BASENAME = "targets.reg"


def workspace_root(output_dir: str, workspace_subdir: str = WORKSPACE_SUBDIR) -> str:
    return os.path.join(os.path.abspath(output_dir), workspace_subdir)


def master_root(output_dir: str, workspace_subdir: str = WORKSPACE_SUBDIR) -> str:
    return os.path.join(workspace_root(output_dir, workspace_subdir), MASTER_SUBDIR)
