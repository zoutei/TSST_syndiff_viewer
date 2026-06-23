"""Parse frozen ``ws/diff_config.yaml`` into workspace label maps."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

_EPOCH_STAGE_KINDS = frozenset({"hotpants", "kernel_subtract"})


@dataclass(frozen=True)
class EpochProduct:
    """One DS9 sidebar product derived from ``diff_config.yaml``."""

    key: str
    button_label: str
    kind: str
    workspace_label: str | None = None
    needs_epoch: bool = True


@dataclass(frozen=True)
class HotpantsLabels:
    """Output labels from one hotpants pipeline stage."""

    diffs: str
    bkg: str | None = None
    convolved: str | None = None


@dataclass(frozen=True)
class PipelineLabels:
    """Workspace labels extracted from a frozen diff config."""

    diff_label: str
    conv_label: str | None
    bkg_label: str | None
    write_bkg: bool
    write_convolved: bool
    lc_label: str
    lc_dir: str
    conv_template_label: str | None = None
    kernel_fit_dir: str | None = None
    hotpants_stages: list[HotpantsLabels] = field(default_factory=list)
    additional_targets: list[str] = field(default_factory=list)
    template_paths: dict[str | int, str] = field(default_factory=dict)
    template_dir: str | None = None
    pipeline: list[dict[str, Any]] = field(default_factory=list)
    epoch_products: list[EpochProduct] = field(default_factory=list)


def _stage_by_kind(pipeline: list[Any], kind: str) -> dict[str, Any] | None:
    for entry in pipeline:
        if isinstance(entry, dict) and entry.get("kind") == kind:
            return entry
    return None


def _stages_by_kind(pipeline: list[Any], kind: str) -> list[dict[str, Any]]:
    return [entry for entry in pipeline if isinstance(entry, dict) and entry.get("kind") == kind]


def _hotpants_labels(stage: dict[str, Any]) -> HotpantsLabels:
    hp_out = stage.get("output") or {}
    diffs = hp_out.get("diffs")
    if not diffs:
        diffs = "hp_d"
    return HotpantsLabels(
        diffs=str(diffs),
        bkg=str(hp_out["bkg"]) if hp_out.get("bkg") else None,
        convolved=str(hp_out["convolved"]) if hp_out.get("convolved") else None,
    )


def _fallback_hotpants_stage(index: int) -> HotpantsLabels:
    if index == 0:
        return HotpantsLabels(diffs="hp1_d", bkg="hp1_b", convolved="hp1_c")
    return HotpantsLabels(diffs="hp2_d", bkg="hp2_b", convolved="hp2_c")


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "t", "true", "yes", "y"}
    return bool(value)


def primary_diff_label(pipeline: list[Any], phot: dict[str, Any]) -> str:
    """Return the primary diff label from forced photometry or the last subtract stage."""
    phot_input = (phot.get("inputs") or {}).get("diffs")
    if phot_input:
        return str(phot_input)
    last: str | None = None
    for entry in pipeline:
        if not isinstance(entry, dict) or entry.get("kind") not in _EPOCH_STAGE_KINDS:
            continue
        out = entry.get("output") or {}
        if out.get("diffs"):
            last = str(out["diffs"])
    return last or "hp_d"


def _stage_epoch_products(stage: dict[str, Any], *, primary_diff: str) -> list[tuple[str, str]]:
    """Return ``(workspace_label, kind)`` pairs for one subtract/hotpants stage."""
    kind = stage.get("kind")
    if kind not in _EPOCH_STAGE_KINDS:
        return []

    out = stage.get("output") or {}
    products: list[tuple[str, str]] = []

    write_convolved = _coerce_bool(stage.get("write_convolved"), default=True)
    write_bkg = _coerce_bool(stage.get("write_bkg"), default=True)

    convolved = out.get("convolved")
    if convolved and write_convolved:
        products.append((str(convolved), "convolved"))

    diffs = out.get("diffs")
    if diffs and str(diffs) != primary_diff:
        products.append((str(diffs), "diff"))

    bkg = out.get("phot_bkg") or out.get("bkg")
    if bkg and (write_bkg if kind == "hotpants" else True):
        products.append((str(bkg), "bkg"))

    return products


def list_epoch_products(labels: PipelineLabels) -> list[EpochProduct]:
    """Return ordered DS9 products for the Selected FFI sidebar."""
    primary = labels.diff_label
    products: list[EpochProduct] = [
        EpochProduct(
            key=primary,
            button_label=primary,
            kind="diff",
            workspace_label=primary,
        ),
        EpochProduct(key="sci", button_label="FFI", kind="sci"),
        EpochProduct(key="template", button_label="Template", kind="template"),
    ]

    for stage in reversed(labels.pipeline):
        if not isinstance(stage, dict) or stage.get("kind") not in _EPOCH_STAGE_KINDS:
            continue
        for workspace_label, kind in _stage_epoch_products(stage, primary_diff=primary):
            products.append(
                EpochProduct(
                    key=workspace_label,
                    button_label=workspace_label,
                    kind=kind,
                    workspace_label=workspace_label,
                )
            )

    if labels.conv_template_label:
        products.append(
            EpochProduct(
                key="conv_template",
                button_label="Conv Template",
                kind="conv_template",
                workspace_label=labels.conv_template_label,
            )
        )

    products.append(
        EpochProduct(
            key="mask",
            button_label="Mask",
            kind="mask",
            needs_epoch=False,
        )
    )
    return products


def _last_stage_output(pipeline: list[Any], stage_kind: str, output_key: str) -> str | None:
    for entry in reversed(pipeline):
        if not isinstance(entry, dict) or entry.get("kind") != stage_kind:
            continue
        out = entry.get("output") or {}
        value = out.get(output_key)
        if value:
            return str(value)
    return None


def parse_diff_config(path: str | Path) -> PipelineLabels:
    """Load ``ws/diff_config.yaml`` and return review-relevant workspace labels."""
    cfg_path = Path(path)
    with cfg_path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    pipeline = raw.get("pipeline") or []
    hotpants_stages_raw = _stages_by_kind(pipeline, "hotpants")
    hotpants = hotpants_stages_raw[0] if hotpants_stages_raw else {}
    phot = _stage_by_kind(pipeline, "forced_photometry") or {}
    kernel_sub = _stage_by_kind(pipeline, "kernel_subtract") or {}
    conv_tmpl = _stage_by_kind(pipeline, "convolved_templates") or {}
    kernel_fit = _stage_by_kind(pipeline, "kernel_fit") or {}

    ks_out = kernel_sub.get("output") or {}

    diff_label = primary_diff_label(pipeline, phot)
    bkg_label = (
        _last_stage_output(pipeline, "hotpants", "bkg")
        or ks_out.get("phot_bkg")
        or ks_out.get("bkg")
    )
    conv_label = _last_stage_output(pipeline, "hotpants", "convolved")
    conv_template_label = conv_tmpl.get("output")
    if conv_template_label is not None:
        conv_template_label = str(conv_template_label)

    kernel_fit_dir = kernel_fit.get("output")
    if kernel_fit_dir is not None:
        kernel_fit_dir = str(kernel_fit_dir)

    hotpants_stages = [_hotpants_labels(stage) for stage in hotpants_stages_raw]
    while len(hotpants_stages) < 2:
        hotpants_stages.append(_fallback_hotpants_stage(len(hotpants_stages)))

    lc_label = str(phot.get("output") or "lc_prf_on_diffs")

    additional: list[str] = []
    for item in raw.get("additional_forced_targets") or []:
        if isinstance(item, dict) and item.get("name"):
            additional.append(str(item["name"]))

    template_paths = raw.get("template_paths") or {}
    template_dir = raw.get("template_dir")
    if template_dir is not None:
        template_dir = str(template_dir)

    write_bkg = bool(bkg_label)
    if hotpants:
        write_bkg = bool(hotpants.get("write_bkg", write_bkg))
    write_convolved = bool(conv_label or conv_template_label)
    if hotpants:
        write_convolved = bool(hotpants.get("write_convolved", write_convolved))

    pipeline_entries = [entry for entry in pipeline if isinstance(entry, dict)]
    labels = PipelineLabels(
        diff_label=diff_label,
        conv_label=str(conv_label) if conv_label else None,
        bkg_label=str(bkg_label) if bkg_label else None,
        write_bkg=write_bkg,
        write_convolved=write_convolved,
        lc_label=lc_label,
        lc_dir=lc_label,
        conv_template_label=conv_template_label,
        kernel_fit_dir=kernel_fit_dir,
        hotpants_stages=hotpants_stages,
        additional_targets=additional,
        template_paths=dict(template_paths),
        template_dir=template_dir,
        pipeline=pipeline_entries,
    )
    return replace(labels, epoch_products=list_epoch_products(labels))


def list_lightcurve_options(labels: PipelineLabels) -> list[tuple[str, str]]:
    """Return ``[(display_name, filename), ...]`` for the LC selector dropdown."""
    options = [("primary", "lightcurve.csv")]
    for name in labels.additional_targets:
        options.append((name, f"lightcurve_{name}.csv"))
    return options
