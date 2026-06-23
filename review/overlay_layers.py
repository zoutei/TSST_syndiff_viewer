"""Compare-layer state helpers for multi light-curve overlay."""

from __future__ import annotations

import uuid
from typing import Any

from .config import ReviewConfig
from .event_index import EventIndex

MAX_OVERLAY_LAYERS = 5
PRIMARY_LAYER_KEY = "primary"

LayerDict = dict[str, Any]
LayerIdentity = tuple[str, str, str]
SelectedPoint = dict[str, Any]
LayerIndexStore = dict[str, dict[str, Any]]


def layer_identity(layer: LayerDict) -> LayerIdentity:
    return (layer["workspace"], layer["lc_dir"], layer["lc_name"])


def primary_identity_from_store(store: dict[str, Any] | None) -> LayerIdentity:
    if not store:
        return ("", "", "")
    return (
        store.get("workspace", ""),
        store.get("lc_dir", ""),
        store.get("lc_name", ""),
    )


def layer_label(layer: LayerDict) -> str:
    return f"{layer['workspace']} / {layer['lc_name']}"


def active_layer_display_label(active_layer: str, primary_store: dict[str, Any] | None, layers: list[LayerDict] | None) -> str:
    if active_layer == PRIMARY_LAYER_KEY:
        store = primary_store or {}
        ws = store.get("workspace", "?")
        name = store.get("lc_name", "?")
        return f"{ws} / {name}"
    for layer in layers or []:
        if layer["id"] == active_layer:
            return layer_label(layer)
    return active_layer


def make_layer(
    *,
    workspace: str,
    lc_dir: str,
    lc_name: str,
    flux_offset: float = 0.0,
) -> LayerDict:
    return {
        "id": f"layer-{uuid.uuid4().hex[:8]}",
        "workspace": workspace,
        "lc_dir": lc_dir,
        "lc_name": lc_name,
        "flux_offset": float(flux_offset),
    }


def layer_is_visible_for_plot(layer: LayerDict, primary: LayerIdentity) -> bool:
    return layer_identity(layer) != primary


def can_add_layer(layers: list[LayerDict] | None) -> bool:
    return len(layers or []) < MAX_OVERLAY_LAYERS


def has_layer_identity(layers: list[LayerDict] | None, identity: LayerIdentity) -> bool:
    return any(layer_identity(layer) == identity for layer in layers or [])


def append_layer_if_new(
    layers: list[LayerDict] | None,
    *,
    workspace: str,
    lc_dir: str,
    lc_name: str,
) -> list[LayerDict]:
    current = list(layers or [])
    if not can_add_layer(current):
        return current
    identity = (workspace, lc_dir, lc_name)
    if has_layer_identity(current, identity):
        return current
    return [*current, make_layer(workspace=workspace, lc_dir=lc_dir, lc_name=lc_name)]


def remove_layer(layers: list[LayerDict] | None, layer_id: str) -> list[LayerDict]:
    return [layer for layer in layers or [] if layer["id"] != layer_id]


def set_layer_offset(layers: list[LayerDict] | None, layer_id: str, flux_offset: float) -> list[LayerDict]:
    updated: list[LayerDict] = []
    for layer in layers or []:
        if layer["id"] == layer_id:
            new_layer = dict(layer)
            new_layer["flux_offset"] = float(flux_offset)
            updated.append(new_layer)
        else:
            updated.append(layer)
    return updated


def build_layer_store(
    idx: EventIndex,
    *,
    event: str,
    workspace: str,
    lc_dir: str,
    lc_name: str,
    fits_event_dir: str,
) -> dict[str, Any]:
    return {
        "event": event,
        "workspace": workspace,
        "lc_dir": lc_dir,
        "lc_name": lc_name,
        "epochs": idx.epochs.to_dict(orient="list"),
        "regions_path": str(idx.regions_path) if idx.regions_path.is_file() else None,
        "fits_event_dir": str(fits_event_dir),
        "crop_bounds": idx.crop_bounds,
        **idx.kernel_workspace_paths(),
    }


def load_layer_index_payload(
    cfg: ReviewConfig,
    event: str,
    layers: list[LayerDict] | None,
) -> LayerIndexStore:
    if not event or not layers:
        return {}
    fits_event = cfg.source_event_dir(event)
    event_dir = cfg.event_dir(event)
    payload: LayerIndexStore = {}
    for layer in layers:
        layer_id = layer["id"]
        try:
            idx = EventIndex.load(
                event_dir,
                workspace_subdir=layer["workspace"],
                lc_dir=layer["lc_dir"],
                lc_name=layer["lc_name"],
                fits_event_dir=fits_event,
            )
            payload[layer_id] = build_layer_store(
                idx,
                event=event,
                workspace=layer["workspace"],
                lc_dir=layer["lc_dir"],
                lc_name=layer["lc_name"],
                fits_event_dir=str(fits_event),
            )
        except Exception as exc:
            payload[layer_id] = {"error": str(exc)}
    return payload


def resolve_active_context(
    active_layer: str | None,
    primary_store: dict[str, Any] | None,
    layer_index_store: LayerIndexStore | None,
    selected_point: SelectedPoint | None,
) -> tuple[dict[str, Any] | None, int | None]:
    layer_key = active_layer or PRIMARY_LAYER_KEY
    if layer_key == PRIMARY_LAYER_KEY:
        store = primary_store
    else:
        store = (layer_index_store or {}).get(layer_key)
        if store and store.get("error"):
            store = None
    epoch_idx = None
    if selected_point and selected_point.get("layer") == layer_key:
        raw_epoch = selected_point.get("epoch_idx")
        if raw_epoch is not None:
            epoch_idx = int(raw_epoch)
    return store, epoch_idx


def point_from_plot_click(click_data: dict | None) -> SelectedPoint | None:
    """Return ``{layer, epoch_idx}`` when a Syndiff marker is clicked."""
    if not click_data or not click_data.get("points"):
        return None
    custom = click_data["points"][0].get("customdata")
    if custom is None:
        return None
    if isinstance(custom, (list, tuple)) and len(custom) >= 2:
        return {"layer": str(custom[0]), "epoch_idx": int(custom[1])}
    if isinstance(custom, (int, float)):
        return {"layer": PRIMARY_LAYER_KEY, "epoch_idx": int(custom)}
    return None
