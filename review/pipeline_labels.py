"""Parse frozen ``ws/diff_config.yaml`` into workspace label maps."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


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
    hp_out = hotpants.get("output") or {}

    diff_label = str(
        ks_out.get("diffs")
        or hp_out.get("diffs")
        or phot.get("inputs", {}).get("diffs")
        or "hp_d"
    )
    bkg_label = ks_out.get("phot_bkg") or ks_out.get("bkg") or hp_out.get("bkg")
    conv_label = hp_out.get("convolved")
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

    return PipelineLabels(
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
    )


def list_lightcurve_options(labels: PipelineLabels) -> list[tuple[str, str]]:
    """Return ``[(display_name, filename), ...]`` for the LC selector dropdown."""
    options = [("primary", "lightcurve.csv")]
    for name in labels.additional_targets:
        options.append((name, f"lightcurve_{name}.csv"))
    return options
