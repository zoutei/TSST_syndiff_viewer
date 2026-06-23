"""Parse frozen ``ws/diff_config.yaml`` into workspace label maps."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

LEGACY_METHOD = "default"

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
    photometry_methods: list[str] = field(default_factory=list)
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


def _parse_photometry_methods(phot: dict[str, Any]) -> list[str]:
    methods_raw = phot.get("methods") or []
    names: list[str] = []
    for item in methods_raw:
        if isinstance(item, dict) and item.get("name"):
            names.append(str(item["name"]))
    return names


def lightcurve_csv_basename(method: str, target: str = "primary") -> str:
    """Return the CSV filename for a (method, target) pair in new-style naming."""
    if target == "primary":
        return f"lightcurve_{method}.csv"
    return f"lightcurve_{method}_{target}.csv"


def _legacy_lightcurve_filename(target: str) -> str:
    if target == "primary":
        return "lightcurve.csv"
    return f"lightcurve_{target}.csv"


def _infer_photometry_methods(lc_dir: Path, additional_targets: list[str]) -> list[str]:
    methods: set[str] = set()
    for path in sorted(lc_dir.glob("lightcurve_*.csv")):
        suffix = path.stem.removeprefix("lightcurve_")
        if not suffix:
            continue
        matched = False
        for target in additional_targets:
            token = f"_{target}"
            if suffix.endswith(token) and len(suffix) > len(token):
                methods.add(suffix[: -len(token)])
                matched = True
                break
        if not matched:
            methods.add(suffix)
    return sorted(methods)


def _photometry_mode(labels: PipelineLabels, lc_dir: Path | None) -> str:
    if labels.photometry_methods:
        return "new"
    if lc_dir is not None and (lc_dir / "lightcurve.csv").is_file():
        return "legacy"
    return "inferred"


def list_photometry_methods(labels: PipelineLabels, lc_dir: Path | None = None) -> list[str]:
    """Return configured or inferred photometry method names for the Method dropdown."""
    if labels.photometry_methods:
        return list(labels.photometry_methods)
    if lc_dir is not None and (lc_dir / "lightcurve.csv").is_file():
        return [LEGACY_METHOD]
    if lc_dir is not None:
        inferred = _infer_photometry_methods(lc_dir, labels.additional_targets)
        if inferred:
            return inferred
    return []


def list_forced_targets(labels: PipelineLabels) -> list[str]:
    """Return forced-photometry position targets."""
    return ["primary", *labels.additional_targets]


def lightcurve_selection_key(method: str, target: str) -> str:
    """Return the combined Target dropdown value for *(method, target)*."""
    if method == LEGACY_METHOD:
        return target
    return f"{method}_{target}"


def parse_lightcurve_selection(
    key: str,
    labels: PipelineLabels,
    lc_dir: Path | None = None,
) -> tuple[str, str]:
    """Parse a Target dropdown value into *(method, target)*."""
    if _photometry_mode(labels, lc_dir) == "legacy":
        if key in list_forced_targets(labels):
            return LEGACY_METHOD, key
        raise ValueError(f"Unknown light curve selection: {key!r}")

    for target in sorted(list_forced_targets(labels), key=len, reverse=True):
        if target == "primary":
            continue
        suffix = f"_{target}"
        if key.endswith(suffix):
            method = key[: -len(suffix)]
            if method:
                return method, target

    primary_suffix = "_primary"
    if key.endswith(primary_suffix):
        method = key[: -len(primary_suffix)]
        if method:
            return method, "primary"

    methods = list_photometry_methods(labels, lc_dir)
    if key in methods:
        return key, "primary"

    raise ValueError(f"Unknown light curve selection: {key!r}")


def list_lightcurve_selections(labels: PipelineLabels, lc_dir: Path | None = None) -> list[str]:
    """Return combined method/target keys for the Target dropdown."""
    mode = _photometry_mode(labels, lc_dir)
    if mode == "legacy":
        keys = list_forced_targets(labels)
    else:
        methods = list_photometry_methods(labels, lc_dir)
        targets = list_forced_targets(labels)
        keys = [lightcurve_selection_key(method, target) for method in methods for target in targets]

    if lc_dir is None:
        return keys

    existing: list[str] = []
    for key in keys:
        method, target = parse_lightcurve_selection(key, labels, lc_dir)
        filename = resolve_lightcurve_filename(method, target, labels, lc_dir)
        if (lc_dir / filename).is_file():
            existing.append(key)
    return existing


def resolve_lightcurve_filename(
    method: str,
    target: str,
    labels: PipelineLabels,
    lc_dir: Path,
) -> str:
    """Resolve the light-curve CSV basename for *(method, target)*."""
    mode = _photometry_mode(labels, lc_dir)
    if mode == "legacy":
        return _legacy_lightcurve_filename(target)
    if mode == "new":
        return lightcurve_csv_basename(method, target)
    # inferred: new-style filenames on disk without methods in frozen config
    return lightcurve_csv_basename(method, target)


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
    photometry_methods = _parse_photometry_methods(phot)

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
        photometry_methods=photometry_methods,
        additional_targets=additional,
        template_paths=dict(template_paths),
        template_dir=template_dir,
        pipeline=pipeline_entries,
    )
    return replace(labels, epoch_products=list_epoch_products(labels))
