"""Sync non-FITS workspace metadata from NFS source to a local cache."""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from review.mount import list_events, list_photometry_dirs, list_workspaces
from review.support.paths import DEFAULT_MANIFEST_BASENAME, TARGETS_DS9_REGION_BASENAME

log = logging.getLogger(__name__)

CLUSTER_TEMPLATE_JOB_BASENAME = "cluster_template_job.json"
CONVOLVED_TEMPLATES_CSV_BASENAME = "convolved_templates.csv"

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
DEFAULT_CACHE_ROOT = PROJECT_ROOT / ".cache" / "workspace"


@dataclass
class SyncResult:
    copied: int = 0
    skipped: int = 0
    errors: list[str] = field(default_factory=list)


def resolve_cache_root(cache_root: str | Path | None = None) -> Path:
    if cache_root is None:
        return DEFAULT_CACHE_ROOT
    return Path(cache_root).expanduser().resolve()


def needs_update(src: Path, dst: Path) -> bool:
    """Return True if *dst* is missing or its mtime differs from *src*."""
    if not dst.is_file():
        return True
    try:
        return src.stat().st_mtime_ns != dst.stat().st_mtime_ns
    except OSError:
        return True


def discover_metadata_files(source_root: Path) -> list[Path]:
    """Return metadata file paths under *source_root* (CSVs, YAML, REG; no FITS)."""
    root = source_root.resolve()
    events_dir = root / "events"
    if not events_dir.is_dir():
        return []

    files: list[Path] = []
    for event_label in list_events(root):
        event_dir = events_dir / event_label
        manifest = event_dir / DEFAULT_MANIFEST_BASENAME
        if manifest.is_file():
            files.append(manifest)

        cluster_job = event_dir / CLUSTER_TEMPLATE_JOB_BASENAME
        if cluster_job.is_file():
            files.append(cluster_job)

        for ws_name in list_workspaces(event_dir):
            ws_dir = event_dir / ws_name
            diff_cfg = ws_dir / "diff_config.yaml"
            if diff_cfg.is_file():
                files.append(diff_cfg)
            targets_reg = ws_dir / TARGETS_DS9_REGION_BASENAME
            if targets_reg.is_file():
                files.append(targets_reg)
            for csv_path in sorted(ws_dir.glob(f"**/{CONVOLVED_TEMPLATES_CSV_BASENAME}")):
                if csv_path.is_file():
                    files.append(csv_path)
            for lc_name in list_photometry_dirs(ws_dir):
                lc_dir = ws_dir / lc_name
                for csv_path in sorted(lc_dir.glob("*.csv")):
                    if csv_path.is_file():
                        files.append(csv_path)
    return files


def _copy_file(src: Path, dst: Path) -> str | None:
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return None
    except OSError as exc:
        return f"{src} -> {dst}: {exc}"


def sync_workspace_metadata(source_root: Path, cache_root: Path) -> SyncResult:
    """Copy metadata files from *source_root* to *cache_root*, skipping unchanged mtimes."""
    source = source_root.resolve()
    cache = cache_root.resolve()
    result = SyncResult()

    for src in discover_metadata_files(source):
        rel = src.relative_to(source)
        dst = cache / rel
        if not needs_update(src, dst):
            result.skipped += 1
            continue
        err = _copy_file(src, dst)
        if err:
            result.errors.append(err)
            log.warning("Cache sync failed: %s", err)
        else:
            result.copied += 1

    log.info(
        "Cache sync: %d copied, %d skipped (up to date)%s",
        result.copied,
        result.skipped,
        f", {len(result.errors)} errors" if result.errors else "",
    )
    return result
