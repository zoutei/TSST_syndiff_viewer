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
    ConvTemplateIndex,
    find_template_by_offset,
    parse_crop_bounds_from_targets_reg,
    resolve_template_dir,
)

from .paths_resolve import resolve_fits_path
from .pipeline_labels import (
    EpochProduct,
    PipelineLabels,
    list_epoch_products,
    list_lightcurve_options,
    parse_diff_config,
)

log = logging.getLogger(__name__)

CLUSTER_TEMPLATE_JOB_BASENAME = "cluster_template_job.json"

_master_index_cache: dict[tuple[str, str], "_MasterIndex"] = {}
_epoch_paths_cache: dict[tuple[str, str, str, str], pd.DataFrame] = {}
_workspace_context_cache: dict[tuple[str, str, str], "_WorkspaceContext"] = {}

_FLUX_COLUMNS = ("flux", "eflux", "snr", "btjd")


def clear_index_cache() -> None:
    """Drop all in-memory NFS index caches (call after metadata refresh)."""
    _master_index_cache.clear()
    _epoch_paths_cache.clear()
    _workspace_context_cache.clear()


def master_index_is_cached(fits_event_path: str | Path, workspace_subdir: str) -> bool:
    key = (str(Path(fits_event_path).resolve()), workspace_subdir)
    return key in _master_index_cache


def get_master_index(fits_event_path: Path, workspace_subdir: str) -> _MasterIndex:
    """Return cached master/ index, building it on first access only."""
    key = (str(fits_event_path.resolve()), workspace_subdir)
    if key not in _master_index_cache:
        master = Path(master_root(str(fits_event_path), workspace_subdir))
        fits_ws = fits_event_path / workspace_subdir
        _master_index_cache[key] = _MasterIndex.build(master, fits_ws)
    return _master_index_cache[key]


@dataclass
class _WorkspaceContext:
    """Cached metadata and path indexes shared across targets in one workspace."""

    labels: PipelineLabels
    manifest_df: pd.DataFrame
    manifest_by_pid: dict[str, dict[str, Any]]
    crop_bounds: dict | None
    template_dir: Path | None
    conv_templates_dir: Path | None
    conv_template_index: ConvTemplateIndex
    template_cache: "_TemplatePathCache"
    regions_path: str | None
    mask_path: str | None
    hotpants_ok_col: str
    kernel_paths: dict[str, Any]


def _workspace_context_key(
    event_path: Path, fits_path: Path, workspace_subdir: str
) -> tuple[str, str, str]:
    return (str(event_path.resolve()), str(fits_path.resolve()), workspace_subdir)


def _manifest_by_pid(manifest_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for _, row in manifest_df.iterrows():
        fname = str(row.get("filename") or row.get("path") or "")
        pid = tess_product_id_from_ffi_path(fname)
        if pid:
            out[pid] = row.to_dict()
    return out


def _resolve_epoch_product_path(
    product: EpochProduct,
    *,
    master_index: _MasterIndex,
    fits_ws: Path,
    product_id: str | None,
    sci_path: Path | None,
    template_path: Path | None,
    conv_template_path: Path | None,
    mask_path: str | None,
) -> Path | None:
    if product.kind == "sci":
        return sci_path
    if product.kind == "template":
        return template_path
    if product.kind == "conv_template":
        return conv_template_path
    if product.kind == "mask":
        return Path(mask_path) if mask_path else None
    if product.workspace_label:
        return _resolve_master_or_workspace(
            master_index, fits_ws, product_id, product.workspace_label
        )
    return None


def _build_epoch_products(
    labels: PipelineLabels,
    *,
    master_index: _MasterIndex,
    fits_ws: Path,
    product_id: str | None,
    sci_path: Path | None,
    template_path: Path | None,
    conv_template_path: Path | None,
    mask_path: str | None,
) -> list[dict[str, Any]]:
    products = labels.epoch_products or list_epoch_products(labels)
    rows: list[dict[str, Any]] = []
    for product in products:
        path = _resolve_epoch_product_path(
            product,
            master_index=master_index,
            fits_ws=fits_ws,
            product_id=product_id,
            sci_path=sci_path,
            template_path=template_path,
            conv_template_path=conv_template_path,
            mask_path=mask_path,
        )
        rows.append(
            {
                "key": product.key,
                "button_label": product.button_label,
                "kind": product.kind,
                "path": str(path) if path else None,
                "needs_epoch": product.needs_epoch,
            }
        )
    return rows


def _kernel_workspace_paths(fits_ws: Path, labels: PipelineLabels) -> dict[str, Any]:
    kernel_fit_dir = fits_ws / labels.kernel_fit_dir if labels.kernel_fit_dir else None

    def _file_in_kf(basename: str) -> Path | None:
        if kernel_fit_dir is None or not kernel_fit_dir.is_dir():
            return None
        p = kernel_fit_dir / basename
        return p if p.is_file() else None

    has_kernel_fit = kernel_fit_dir is not None and kernel_fit_dir.is_dir()
    mask = fits_ws / "shared_mask.fits"
    return {
        "has_kernel_fit": has_kernel_fit,
        "kernel_fit_dir": str(kernel_fit_dir) if kernel_fit_dir else None,
        "kernel_reference_path": str(p) if (p := _file_in_kf("ffi.fits")) else None,
        "kernel_template_path": str(p) if (p := _file_in_kf("template.fits")) else None,
        "kernel_hp1_diff_path": str(p) if (p := _file_in_kf("hp1_diff.fits")) else None,
        "kernel_hp1_bkg_path": str(p) if (p := _file_in_kf("hp1_bkg.fits")) else None,
        "kernel_hp2_diff_path": str(p) if (p := _file_in_kf("hp2_diff.fits")) else None,
        "kernel_hp2_bkg_path": str(p) if (p := _file_in_kf("hp2_bkg.fits")) else None,
        "kernel_sci1_clean_path": str(p) if (p := _file_in_kf("sci1_clean.fits")) else None,
        "kernel_phot_bkg_fine_path": str(p)
        if (p := _file_in_kf("phot_bkg_fine_on_hp1_diff.fits"))
        else None,
        "mask_path": str(mask) if mask.is_file() else None,
        "hotpants_stages": [
            {"diffs": s.diffs, "bkg": s.bkg, "convolved": s.convolved}
            for s in labels.hotpants_stages
        ],
        "epoch_products": [
            {
                "key": p.key,
                "button_label": p.button_label,
                "kind": p.kind,
                "needs_epoch": p.needs_epoch,
            }
            for p in (labels.epoch_products or list_epoch_products(labels))
        ],
    }


def get_workspace_context(
    event_path: Path,
    fits_path: Path,
    workspace_subdir: str,
) -> _WorkspaceContext:
    key = _workspace_context_key(event_path, fits_path, workspace_subdir)
    cached = _workspace_context_cache.get(key)
    if cached is not None:
        return cached

    meta_ws = event_path / workspace_subdir
    fits_ws = fits_path / workspace_subdir
    labels = parse_diff_config(meta_ws / "diff_config.yaml")
    manifest_df = load_frame_manifest(str(event_path))
    template_dir = resolve_template_dir(fits_ws)
    conv_templates_dir = (
        fits_ws / labels.conv_template_label if labels.conv_template_label else None
    )
    regions = meta_ws / TARGETS_DS9_REGION_BASENAME
    mask = fits_ws / "shared_mask.fits"
    diff_safe = sanitize_workspace_label(labels.diff_label)

    ctx = _WorkspaceContext(
        labels=labels,
        manifest_df=manifest_df,
        manifest_by_pid=_manifest_by_pid(manifest_df),
        crop_bounds=parse_crop_bounds_from_targets_reg(meta_ws),
        template_dir=template_dir,
        conv_templates_dir=conv_templates_dir,
        conv_template_index=ConvTemplateIndex.from_dir(conv_templates_dir),
        template_cache=_TemplatePathCache(template_dir=template_dir, labels=labels),
        regions_path=str(regions) if regions.is_file() else None,
        mask_path=str(mask) if mask.is_file() else None,
        hotpants_ok_col=f"hotpants_{diff_safe}_ok",
        kernel_paths=_kernel_workspace_paths(fits_ws, labels),
    )
    _workspace_context_cache[key] = ctx
    return ctx


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
        ws_ctx = get_workspace_context(event_path, fits_path, workspace_subdir)
        labels = ws_ctx.labels
        resolved_lc_dir = lc_dir or labels.lc_dir
        if lc_filename is None:
            lc_filename = _lc_filename_for_name(labels, lc_name)

        lc_path = event_path / workspace_subdir / resolved_lc_dir / lc_filename
        lc_df = _read_lightcurve(lc_path)

        cluster_job_path = event_path / CLUSTER_TEMPLATE_JOB_BASENAME
        if not cluster_job_path.is_file():
            log.debug("No %s under %s", CLUSTER_TEMPLATE_JOB_BASENAME, event_path)

        paths_key = (
            str(event_path.resolve()),
            str(fits_path.resolve()),
            workspace_subdir,
            resolved_lc_dir,
        )
        cached_paths = _epoch_paths_cache.get(paths_key)
        if cached_paths is not None and _lc_rows_compatible(cached_paths, lc_df):
            epochs = _merge_lc_flux(cached_paths, lc_df)
        else:
            master_index = get_master_index(fits_path, workspace_subdir)
            epochs = _build_epoch_table(
                ws_ctx=ws_ctx,
                metadata_event_path=event_path,
                fits_event_path=fits_path,
                workspace_subdir=workspace_subdir,
                lc_df=lc_df,
                master_index=master_index,
            )
            _epoch_paths_cache[paths_key] = epochs.drop(columns=list(_FLUX_COLUMNS), errors="ignore")
        return cls(
            event_dir=event_path,
            target_label=event_path.name,
            workspace_subdir=workspace_subdir,
            labels=labels,
            epochs=epochs,
            lc_name=lc_name,
            lc_dir=resolved_lc_dir,
            fits_event_dir=fits_path,
            crop_bounds=ws_ctx.crop_bounds,
            template_dir=ws_ctx.template_dir,
            conv_templates_dir=ws_ctx.conv_templates_dir,
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
        ctx = get_workspace_context(self.event_dir, self.fits_event_dir, self.workspace_subdir)
        return ctx.kernel_paths


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


def _lc_rows_compatible(paths_df: pd.DataFrame, lc_df: pd.DataFrame) -> bool:
    if len(paths_df) != len(lc_df):
        return False
    if "filename" not in paths_df.columns or "filename" not in lc_df.columns:
        return len(paths_df) == len(lc_df)
    left = paths_df["filename"].fillna("").astype(str)
    right = lc_df["filename"].fillna("").astype(str)
    return left.equals(right)


def _merge_lc_flux(paths_df: pd.DataFrame, lc_df: pd.DataFrame) -> pd.DataFrame:
    out = paths_df.copy()
    flux = lc_df["flux"]
    eflux = lc_df["eflux"] if "eflux" in lc_df.columns else pd.Series(np.nan, index=lc_df.index)
    btjd = lc_df["btjd"]
    out["btjd"] = btjd.astype(float).values
    out["flux"] = flux.astype(float).values
    out["eflux"] = eflux.astype(float).values
    out["snr"] = [
        float(f) / float(e) if pd.notna(f) and pd.notna(e) and float(e) != 0 else np.nan
        for f, e in zip(out["flux"], out["eflux"], strict=True)
    ]
    return out


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
    ws_ctx: _WorkspaceContext,
    metadata_event_path: Path,
    fits_event_path: Path,
    workspace_subdir: str,
    lc_df: pd.DataFrame,
    master_index: _MasterIndex | None = None,
) -> pd.DataFrame:
    del metadata_event_path  # retained for call-site clarity
    fits_ws = fits_event_path / workspace_subdir
    labels = ws_ctx.labels
    if master_index is None:
        master_index = get_master_index(fits_event_path, workspace_subdir)

    manifest_by_pid = ws_ctx.manifest_by_pid
    regions_path = ws_ctx.regions_path
    mask_path = ws_ctx.mask_path
    hotpants_ok_col = ws_ctx.hotpants_ok_col
    template_cache = ws_ctx.template_cache
    conv_index = ws_ctx.conv_template_index

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
            conv_index.lookup(group_dx, group_dy)
            if group_dx is not None and group_dy is not None
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

        products = _build_epoch_products(
            labels,
            master_index=master_index,
            fits_ws=fits_ws,
            product_id=pid,
            sci_path=sci_path,
            template_path=template_path,
            conv_template_path=conv_template_path,
            mask_path=mask_path,
        )

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
                "products": products,
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
