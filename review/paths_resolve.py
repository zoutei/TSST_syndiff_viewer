"""Resolve cluster absolute paths and symlinks for STScI NFS mounts."""

from __future__ import annotations

import os
from pathlib import Path

CLUSTER_PREFIX = "/astro/armin/koji/syndiff"
LOCAL_CLUSTER_PREFIXES: tuple[str, ...] = (
    "/System/Volumes/Data/astro/armin/koji/syndiff",
    CLUSTER_PREFIX,
)


def remap_cluster_path(path: str | Path | None) -> Path | None:
    """Map ``/astro/armin/koji/syndiff/...`` to a readable local NFS path when needed."""
    if not path:
        return None
    p = Path(os.path.expanduser(str(path)))
    if p.is_file():
        return p
    s = str(p)
    if s.startswith(CLUSTER_PREFIX):
        suffix = s[len(CLUSTER_PREFIX) :]
        for prefix in LOCAL_CLUSTER_PREFIXES:
            alt = Path(prefix + suffix)
            if alt.is_file():
                return alt
    return p if p.is_file() else None


def resolve_fits_path(path: str | Path | None) -> Path | None:
    """Return a readable local FITS path (follows master symlinks + cluster remap)."""
    if not path:
        return None
    p = Path(os.path.expanduser(str(path)))
    if p.is_file():
        return p
    if p.is_symlink():
        target = remap_cluster_path(os.readlink(p))
        if target and target.is_file():
            return target
    return remap_cluster_path(p)
