"""NFS workspace path resolution and health checks for the review UI."""

from __future__ import annotations

import os
from pathlib import Path

from review.pipeline_labels import (
    list_lightcurve_selections,
    parse_diff_config,
    parse_lightcurve_selection,
    resolve_lightcurve_filename,
)

# STScI Macs automount cs10d at /System/Volumes/Data/astro (also /astro).
DEFAULT_MOUNT_ROOT = "/System/Volumes/Data/astro/armin/koji/syndiff/workspace"
FALLBACK_MOUNT_ROOTS: tuple[str, ...] = (
    "/astro/armin/koji/syndiff/workspace",
)


def mount_root_usable(path: Path) -> bool:
    """Return True if *path* looks like a SynDiff workspace mount (has ``events/``)."""
    return path.is_dir() and (path / "events").is_dir()


def resolve_mount_root(
    preferred: str | Path | None = None,
    *,
    allow_fallback: bool = True,
) -> Path:
    """Pick the first usable workspace mount, preferring *preferred* when set."""
    tried: list[Path] = []
    seen: set[str] = set()

    def consider(raw: str | Path) -> Path | None:
        p = Path(os.path.expanduser(str(raw)))
        try:
            resolved = p.resolve()
        except OSError:
            resolved = p
        key = str(resolved)
        if key in seen:
            return None
        seen.add(key)
        tried.append(resolved)
        return resolved

    if preferred is not None:
        p = consider(preferred)
        if p and mount_root_usable(p):
            return p
        if not allow_fallback:
            return p or tried[-1]

    if allow_fallback:
        for raw in (DEFAULT_MOUNT_ROOT, *FALLBACK_MOUNT_ROOTS):
            p = consider(raw)
            if p and mount_root_usable(p):
                return p

    return tried[0] if tried else Path(DEFAULT_MOUNT_ROOT)


def is_workspace_dir(path: Path) -> bool:
    """Return True if *path* is a workspace directory (``ws`` or ``ws_*`` with config)."""
    name = path.name
    return (name == "ws" or name.startswith("ws_")) and (path / "diff_config.yaml").is_file()


def is_photometry_dir(path: Path) -> bool:
    """Return True if *path* is a photometry output dir (``lc_*`` with light-curve CSVs)."""
    if not path.is_dir() or not path.name.startswith("lc_"):
        return False
    if (path / "lightcurve.csv").is_file():
        return True
    return any(
        entry.is_file() and entry.name.startswith("lightcurve_") and entry.suffix == ".csv"
        for entry in path.iterdir()
    )


def list_workspaces(event_dir: str | Path) -> list[str]:
    """Return workspace subdir names under an event (``ws``, ``ws_*`` with ``diff_config.yaml``)."""
    event = Path(os.path.expanduser(str(event_dir)))
    if not event.is_dir():
        return []
    names = sorted(entry.name for entry in event.iterdir() if entry.is_dir() and is_workspace_dir(entry))
    if "ws" in names:
        names.remove("ws")
        return ["ws", *names]
    return names


def list_photometry_dirs(workspace_dir: str | Path) -> list[str]:
    """Return photometry subdir names (``lc_*``) that contain light-curve CSVs."""
    ws = Path(os.path.expanduser(str(workspace_dir)))
    if not ws.is_dir():
        return []
    return sorted(entry.name for entry in ws.iterdir() if is_photometry_dir(entry))


def is_healthy(
    mount_root: str | Path,
    test_event: str = "s0023_c1_k3_2020ftl",
    *,
    workspace_subdir: str | None = None,
    lc_dir: str | None = None,
    metadata_root: str | Path | None = None,
    fits_root: str | Path | None = None,
) -> tuple[bool, str]:
    """Return ``(ok, message)`` after checking that workspace data is reviewable.

  *metadata_root* (default *mount_root*) supplies CSV/YAML; *fits_root* (default
  *mount_root*) supplies ``master/`` FITS for DS9.
    """
    meta_root = Path(os.path.expanduser(str(metadata_root or mount_root)))
    fits_mount = Path(os.path.expanduser(str(fits_root or mount_root)))
    meta_event = meta_root / "events" / test_event
    fits_event = fits_mount / "events" / test_event

    if not meta_root.is_dir():
        return False, f"Metadata root missing: {meta_root}"
    if not meta_event.is_dir():
        return False, f"Event missing: {meta_event}"

    workspaces = list_workspaces(meta_event)
    if not workspaces:
        return False, f"No workspace in {meta_event}"

    ws_name = workspace_subdir if workspace_subdir in workspaces else workspaces[0]
    meta_ws = meta_event / ws_name
    fits_ws = fits_event / ws_name

    phot_dirs = list_photometry_dirs(meta_ws)
    if not phot_dirs:
        return False, f"No photometry dir in {meta_ws}"

    lc_dir_name = lc_dir if lc_dir in phot_dirs else phot_dirs[0]
    meta_lc_dir = meta_ws / lc_dir_name
    labels = parse_diff_config(meta_ws / "diff_config.yaml")
    selections = list_lightcurve_selections(labels, meta_lc_dir)
    if not selections:
        return False, f"No light curves in {meta_lc_dir}"

    lc_filename = resolve_lightcurve_filename(
        *parse_lightcurve_selection(selections[0], labels, meta_lc_dir),
        labels,
        meta_lc_dir,
    )
    lc = meta_lc_dir / lc_filename
    master = fits_ws / "master"

    if not lc.is_file():
        return False, f"Light curve missing: {lc}"
    if not master.is_dir():
        return False, f"master/ missing: {master}"

    has_fits = next((True for p in master.iterdir() if p.suffix.lower() == ".fits"), False)
    if not has_fits:
        return False, f"No FITS files in {master}"
    return True, f"OK: {test_event}/{ws_name}/{lc_dir_name}"


def list_events(mount_root: str | Path) -> list[str]:
    """Return event labels under ``{mount_root}/events`` that have at least one workspace."""
    events_dir = Path(os.path.expanduser(str(mount_root))) / "events"
    if not events_dir.is_dir():
        return []
    out: list[str] = []
    for entry in sorted(events_dir.iterdir()):
        if not entry.is_dir():
            continue
        if list_workspaces(entry):
            out.append(entry.name)
    return out
