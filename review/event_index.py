"""Join manifest + light curve and resolve FITS paths via ``ws/master/``."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from review.support.ffi_naming import (
    sanitize_workspace_label,
    tess_product_id_from_ffi_path,
    workspace_frame_stem,
)
from review.support.manifest import load_frame_manifest
from review.support.paths import (
    DEFAULT_MANIFEST_BASENAME,
    TARGETS_DS9_REGION_BASENAME,
    master_root,
)
from review.support.templates import (
    find_template_by_offset,
    lookup_convolved_template,
    parse_crop_bounds_from_targets_reg,
    resolve_template_dir,
)

from .paths_resolve import resolve_fits_path
from .pipeline_labels import PipelineLabels, list_lightcurve_options, parse_diff_config

log = logging.getLogger(__name__)

CLUSTER_TEMPLATE_JOB_BASENAME = "cluster_template_job.json"


@dataclass
class EventIndex:
    event_dir: Path
    target_label: str
    workspace_subdir: str
    labels: PipelineLabels
    epochs: pd.DataFrame
    lc_name: str
    lc_dir: str
    fits_event_dir: Path
    crop_bounds: dict | None = None
    template_dir: Path | None = None
    conv_templates_dir: Path | None = None

    @classmethod
    def load(
        cls,
        event_dir: str | Path,
        *,
        workspace_subdir: str = "ws",
        lc_dir: str | None = None,
        lc_name: str = "primary",
        lc_filename: str | None = None,
        fits_event_dir: str | Path | None = None,
    ) -> EventIndex:
        event_path = Path(event_dir).resolve()
        fits_path = Path(fits_event_dir).resolve() if fits_event_dir else event_path
        ws = event_path / workspace_subdir
        fits_ws = fits_path / workspace_subdir
        labels = parse_diff_config(ws / "diff_config.yaml")
        resolved_lc_dir = lc_dir or labels.lc_dir
        if lc_filename is None:
            lc_filename = _lc_filename_for_name(labels, lc_name)

        lc_path = ws / resolved_lc_dir / lc_filename
        lc_df = _read_lightcurve(lc_path)
        manifest_df = load_frame_manifest(str(event_path))

        crop_bounds = parse_crop_bounds_from_targets_reg(ws)
        template_dir = resolve_template_dir(fits_ws)
        conv_templates_dir = (
            fits_ws / labels.conv_template_label if labels.conv_template_label else None
        )

        cluster_job_path = event_path / CLUSTER_TEMPLATE_JOB_BASENAME
        if not cluster_job_path.is_file():
            log.debug("No %s under %s", CLUSTER_TEMPLATE_JOB_BASENAME, event_path)

        epochs = _build_epoch_table(
            metadata_event_path=event_path,
            fits_event_path=fits_path,
            workspace_subdir=workspace_subdir,
            labels=labels,
            lc_df=lc_df,
            manifest_df=manifest_df,
            template_dir=template_dir,
            conv_templates_dir=conv_templates_dir,
        )
        return cls(
            event_dir=event_path,
            target_label=event_path.name,
            workspace_subdir=workspace_subdir,
            labels=labels,
            epochs=epochs,
            lc_name=lc_name,
            lc_dir=resolved_lc_dir,
            fits_event_dir=fits_path,
            crop_bounds=crop_bounds,
            template_dir=template_dir,
            conv_templates_dir=conv_templates_dir,
        )

    @property
    def workspace_dir(self) -> Path:
        return self.event_dir / self.workspace_subdir

    @property
    def fits_workspace_dir(self) -> Path:
        return self.fits_event_dir / self.workspace_subdir

    @property
    def regions_path(self) -> Path:
        return self.workspace_dir / TARGETS_DS9_REGION_BASENAME

    @property
    def mask_path(self) -> Path:
        return self.fits_workspace_dir / "shared_mask.fits"

    @property
    def kernel_fit_dir(self) -> Path | None:
        if not self.labels.kernel_fit_dir:
            return None
        return self.fits_workspace_dir / self.labels.kernel_fit_dir

    @property
    def has_kernel_fit(self) -> bool:
        kf = self.kernel_fit_dir
        return kf is not None and kf.is_dir()

    @property
    def kernel_reference_path(self) -> Path | None:
        kf = self.kernel_fit_dir
        if kf is None:
            return None
        p = kf / "ffi.fits"
        return p if p.is_file() else None

    @property
    def kernel_sci1_clean_path(self) -> Path | None:
        kf = self.kernel_fit_dir
        if kf is None:
            return None
        p = kf / "sci1_clean.fits"
        return p if p.is_file() else None

    @property
    def kernel_phot_bkg_fine_path(self) -> Path | None:
        kf = self.kernel_fit_dir
        if kf is None:
            return None
        p = kf / "phot_bkg_fine_on_hp1_diff.fits"
        return p if p.is_file() else None

    def _kernel_fit_file(self, basename: str) -> Path | None:
        kf = self.kernel_fit_dir
        if kf is None:
            return None
        p = kf / basename
        return p if p.is_file() else None

    @property
    def kernel_template_path(self) -> Path | None:
        return self._kernel_fit_file("template.fits")

    @property
    def kernel_hp1_diff_path(self) -> Path | None:
        return self._kernel_fit_file("hp1_diff.fits")

    @property
    def kernel_hp1_bkg_path(self) -> Path | None:
        return self._kernel_fit_file("hp1_bkg.fits")

    @property
    def kernel_hp2_diff_path(self) -> Path | None:
        return self._kernel_fit_file("hp2_diff.fits")

    @property
    def kernel_hp2_bkg_path(self) -> Path | None:
        return self._kernel_fit_file("hp2_bkg.fits")

    def kernel_workspace_paths(self) -> dict[str, str | None]:
        """Workspace-level kernel_fit paths for the Dash store."""
        return {
            "has_kernel_fit": self.has_kernel_fit,
            "kernel_fit_dir": str(self.kernel_fit_dir) if self.kernel_fit_dir else None,
            "kernel_reference_path": str(self.kernel_reference_path)
            if self.kernel_reference_path
            else None,
            "kernel_template_path": str(self.kernel_template_path) if self.kernel_template_path else None,
            "kernel_hp1_diff_path": str(self.kernel_hp1_diff_path) if self.kernel_hp1_diff_path else None,
            "kernel_hp1_bkg_path": str(self.kernel_hp1_bkg_path) if self.kernel_hp1_bkg_path else None,
            "kernel_hp2_diff_path": str(self.kernel_hp2_diff_path) if self.kernel_hp2_diff_path else None,
            "kernel_hp2_bkg_path": str(self.kernel_hp2_bkg_path) if self.kernel_hp2_bkg_path else None,
            "kernel_sci1_clean_path": str(self.kernel_sci1_clean_path)
            if self.kernel_sci1_clean_path
            else None,
            "kernel_phot_bkg_fine_path": str(self.kernel_phot_bkg_fine_path)
            if self.kernel_phot_bkg_fine_path
            else None,
            "mask_path": str(self.mask_path) if self.mask_path.is_file() else None,
            "hotpants_stages": [
                {"diffs": s.diffs, "bkg": s.bkg, "convolved": s.convolved}
                for s in self.labels.hotpants_stages
            ],
        }


@dataclass
class _MasterIndex:
    """Single-pass index of ``master/`` and optional ``ffis/`` filenames."""

    master_dir: Path
    master_names: set[str]
    sci_by_product_id: dict[str, Path]
    ffis_dir: Path | None = None
    ffis_names: set[str] = field(default_factory=set)

    @classmethod
    def build(cls, master: Path, fits_ws: Path) -> _MasterIndex:
        master_names: set[str] = set()
        sci_by_product_id: dict[str, Path] = {}
        if master.is_dir():
            for entry in master.iterdir():
                name = entry.name
                master_names.add(name)
                if "_ffic" in name:
                    pid = tess_product_id_from_ffi_path(name)
                    if pid and pid not in sci_by_product_id:
                        sci_by_product_id[pid] = entry

        ffis = fits_ws / "ffis"
        ffis_names: set[str] = set()
        ffis_dir: Path | None = None
        if ffis.is_dir():
            ffis_dir = ffis
            ffis_names = {p.name for p in ffis.iterdir() if p.is_file()}

        return cls(
            master_dir=master,
            master_names=master_names,
            sci_by_product_id=sci_by_product_id,
            ffis_dir=ffis_dir,
            ffis_names=ffis_names,
        )

    def master_path(self, filename: str) -> Path | None:
        if filename in self.master_names:
            return self.master_dir / filename
        return None


@dataclass
class _TemplatePathCache:
    """Cache offset-based template lookups per (group_dx, group_dy)."""

    template_dir: Path | None
    labels: PipelineLabels
    _by_offset: dict[tuple[float, float], Path | None] = field(default_factory=dict)
    _by_group_yaml: dict[int, Path | None] = field(default_factory=dict)

    def lookup(
        self,
        group_id: int | None,
        group_dx: float | None,
        group_dy: float | None,
    ) -> Path | None:
        if group_dx is not None and group_dy is not None and self.template_dir is not None:
            key = (round(float(group_dx), 6), round(float(group_dy), 6))
            if key not in self._by_offset:
                self._by_offset[key] = find_template_by_offset(
                    self.template_dir, dx=key[0], dy=key[1]
                )
            hit = self._by_offset[key]
            if hit is not None:
                return hit

        if group_id is not None:
            if group_id not in self._by_group_yaml:
                self._by_group_yaml[group_id] = _lookup_yaml_template_path(
                    self.labels, group_id, self.template_dir
                )
            return self._by_group_yaml[group_id]
        return None


def epoch_file_exists(row: Mapping[str, Any]) -> dict[str, bool]:
    """Resolve FITS existence for one epoch (deferred from index build)."""
    return {
        "diff_exists": bool(resolve_fits_path(row.get("diff_path"))),
        "sci_exists": bool(resolve_fits_path(row.get("sci_path"))),
        "template_exists": bool(resolve_fits_path(row.get("template_path"))),
    }


def _lc_filename_for_name(labels: PipelineLabels, lc_name: str) -> str:
    for name, filename in list_lightcurve_options(labels):
        if name == lc_name:
            return filename
    if lc_name.endswith(".csv"):
        return lc_name
    raise ValueError(f"Unknown light curve name: {lc_name!r}")


def _read_lightcurve(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "btjd" not in df.columns:
        raise ValueError(f"Light curve missing btjd column: {path}")
    ok = df["btjd"].notna() & np.isfinite(df["btjd"].astype(float))
    if "flux" in df.columns:
        ok &= df["flux"].notna()
    return df.loc[ok].reset_index(drop=True)


def _float_or_none(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _build_epoch_table(
    *,
    metadata_event_path: Path,
    fits_event_path: Path,
    workspace_subdir: str,
    labels: PipelineLabels,
    lc_df: pd.DataFrame,
    manifest_df: pd.DataFrame,
    template_dir: Path | None,
    conv_templates_dir: Path | None,
) -> pd.DataFrame:
    meta_ws = metadata_event_path / workspace_subdir
    fits_ws = fits_event_path / workspace_subdir
    master = Path(master_root(str(fits_event_path), workspace_subdir))
    master_index = _MasterIndex.build(master, fits_ws)
    template_cache = _TemplatePathCache(template_dir=template_dir, labels=labels)

    regions = meta_ws / TARGETS_DS9_REGION_BASENAME
    mask = fits_ws / "shared_mask.fits"
    regions_path = str(regions) if regions.is_file() else None
    mask_path = str(mask) if mask.is_file() else None

    diff_safe = sanitize_workspace_label(labels.diff_label)
    hotpants_ok_col = f"hotpants_{diff_safe}_ok"

    manifest_by_pid: dict[str, dict[str, Any]] = {}
    for _, row in manifest_df.iterrows():
        fname = str(row.get("filename") or row.get("path") or "")
        pid = tess_product_id_from_ffi_path(fname)
        if pid:
            manifest_by_pid[pid] = row.to_dict()

    records: list[dict[str, Any]] = []
    for i, lc_row in lc_df.iterrows():
        lc_fname = str(lc_row.get("filename") or "")
        pid = tess_product_id_from_ffi_path(lc_fname)
        man = manifest_by_pid.get(pid or "", {})

        group_id = lc_row.get("group_id", man.get("group_id"))
        try:
            group_id = int(group_id) if pd.notna(group_id) else None
        except (TypeError, ValueError):
            group_id = None

        group_dx = _float_or_none(man.get("group_dx"))
        group_dy = _float_or_none(man.get("group_dy"))

        flux = lc_row.get("flux")
        eflux = lc_row.get("eflux")
        btjd = lc_row.get("btjd")
        snr = (
            float(flux) / float(eflux)
            if pd.notna(flux) and pd.notna(eflux) and float(eflux) != 0
            else np.nan
        )

        diff_path = _resolve_master_or_workspace(master_index, fits_ws, pid, labels.diff_label)
        conv_template_path = (
            lookup_convolved_template(conv_templates_dir, group_dx, group_dy)
            if conv_templates_dir is not None and group_dx is not None and group_dy is not None
            else None
        )
        conv_path = (
            _resolve_master_or_workspace(master_index, fits_ws, pid, labels.conv_label)
            if labels.conv_label and labels.write_convolved
            else conv_template_path
        )
        bkg_path = (
            _resolve_master_or_workspace(master_index, fits_ws, pid, labels.bkg_label)
            if labels.bkg_label and labels.write_bkg
            else None
        )

        sci_basename = str(man.get("filename") or "")
        sci_path = _resolve_sci_path(master_index, fits_ws, sci_basename, pid)
        template_path = template_cache.lookup(group_id, group_dx, group_dy)

        hotpants_ok = man.get(hotpants_ok_col)
        if pd.isna(hotpants_ok):
            hotpants_ok = None
        else:
            hotpants_ok = bool(hotpants_ok)

        records.append(
            {
                "epoch_idx": int(i),
                "btjd": float(btjd) if pd.notna(btjd) else np.nan,
                "flux": float(flux) if pd.notna(flux) else np.nan,
                "eflux": float(eflux) if pd.notna(eflux) else np.nan,
                "snr": snr,
                "product_id": pid,
                "group_id": group_id,
                "group_dx": group_dx,
                "group_dy": group_dy,
                "filename": sci_basename,
                "diff_path": str(diff_path) if diff_path else None,
                "conv_path": str(conv_path) if conv_path else None,
                "conv_template_path": str(conv_template_path) if conv_template_path else None,
                "bkg_path": str(bkg_path) if bkg_path else None,
                "sci_path": str(sci_path) if sci_path else None,
                "template_path": str(template_path) if template_path else None,
                "regions_path": regions_path,
                "mask_path": mask_path,
                "hotpants_ok": hotpants_ok,
            }
        )

    return pd.DataFrame.from_records(records)


def _resolve_master_or_workspace(
    index: _MasterIndex,
    ws: Path,
    product_id: str | None,
    label: str | None,
) -> Path | None:
    if not product_id or not label:
        return None
    stem = workspace_frame_stem(product_id, label)
    fname = f"{stem}.fits"
    hit = index.master_path(fname)
    if hit is not None:
        return hit
    ws_path = ws / label / fname
    if ws_path.is_file():
        return ws_path
    return None


def _resolve_sci_path(
    index: _MasterIndex,
    ws: Path,
    sci_basename: str,
    product_id: str | None,
) -> Path | None:
    if sci_basename:
        hit = index.master_path(sci_basename)
        if hit is not None:
            return hit
        if index.ffis_dir is not None and sci_basename in index.ffis_names:
            return index.ffis_dir / sci_basename
    if product_id:
        return index.sci_by_product_id.get(product_id)
    return None


def _lookup_yaml_template_path(
    labels: PipelineLabels,
    group_id: int,
    template_dir: Path | None,
) -> Path | None:
    key = str(group_id)
    if key not in labels.template_paths:
        return None
    p = Path(labels.template_paths[key])
    if not p.is_file():
        return None
    if template_dir is not None:
        try:
            p.resolve().relative_to(template_dir.resolve())
        except ValueError:
            return None
    return p


def resolve_labeled_epoch_path(
    *,
    fits_event_dir: str | Path,
    workspace_subdir: str,
    product_id: str | None,
    label: str | None,
) -> Path | None:
    """Resolve a per-epoch FITS path for a workspace label (master/ then ws/label/)."""
    if not product_id or not label:
        return None
    fits_event = Path(fits_event_dir)
    fits_ws = fits_event / workspace_subdir
    master = Path(master_root(str(fits_event), workspace_subdir))
    stem = workspace_frame_stem(product_id, label)
    fname = f"{stem}.fits"
    for candidate in (master / fname, fits_ws / label / fname):
        resolved = resolve_fits_path(candidate)
        if resolved is not None:
            return resolved
    return None


def manifest_path_for_event(event_dir: Path) -> Path:
    return event_dir / DEFAULT_MANIFEST_BASENAME
