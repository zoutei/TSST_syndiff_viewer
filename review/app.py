"""Plotly Dash light-curve review application."""

from __future__ import annotations

import logging
import re
import sys
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, callback, ctx, dcc, html, no_update

from .config import ReviewConfig
from .crop_cache import ensure_cropped_fits
from .ds9 import Ds9Controller
from .event_index import EventIndex, clear_index_cache, master_index_is_cached
from .mount import is_healthy, list_events, list_photometry_dirs, list_workspaces
from .pipeline_labels import list_lightcurve_options, parse_diff_config
from .smoothing import SmoothingMode, apply_smoothing
from .sync_cache import sync_event_metadata, sync_workspace_metadata
from .tessreduce import clear_tessreduce_cache, load_tessreduce_for_event, tessreduce_store_payload

log = logging.getLogger(__name__)

_StorePayload = tuple[dict[str, Any], str, dict[str, Any]]
_store_payload_cache: dict[tuple[str, str, str, str], _StorePayload] = {}


def clear_store_payload_cache() -> None:
    """Drop cached Dash store payloads (call after metadata refresh)."""
    _store_payload_cache.clear()


def _store_payload_key(event: str, workspace: str, lc_dir: str, target: str) -> tuple[str, str, str, str]:
    return (event, workspace, lc_dir, target)


def _smoothing_mode_from_ui(mode: str) -> SmoothingMode:
    if mode == "Binned σ-clip":
        return "binned"
    if mode == "Savitzky-Golay":
        return "savgol"
    return "none"


def _pick_dropdown_value(current: str | None, options: list[dict[str, str]], default: str) -> str:
    values = {o["value"] for o in options}
    if current in values:
        return current
    return options[0]["value"] if options else default


def _epoch_from_plot_click(click_data: dict | None) -> int | None:
    """Return epoch_idx when a flux marker is clicked; otherwise deselect."""
    if not click_data or not click_data.get("points"):
        return None
    custom = click_data["points"][0].get("customdata")
    if custom is not None:
        return int(custom)
    return None


def _parse_flux_offset(value: float | int | str | None) -> float:
    if value is None or value == "":
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def find_epoch_idx_by_product_id(epochs: dict[str, list] | pd.DataFrame, query: str | None) -> tuple[int | None, str]:
    """Return ``(epoch_idx, error_message)`` for a product_id search query."""
    if not query or not str(query).strip():
        return None, "Enter a product ID."
    df = pd.DataFrame(epochs) if isinstance(epochs, dict) else epochs
    if df.empty or "product_id" not in df.columns:
        return None, "No light-curve data loaded."
    q = str(query).strip().lower()
    pids = df["product_id"].fillna("").astype(str).str.lower()
    exact = df.loc[pids == q]
    if not exact.empty:
        return int(exact.iloc[0]["epoch_idx"]), ""
    contains = df.loc[pids.str.contains(re.escape(q), na=False)]
    if len(contains) == 1:
        return int(contains.iloc[0]["epoch_idx"]), ""
    if len(contains) > 1:
        return int(contains.iloc[0]["epoch_idx"]), f"{len(contains)} matches; selected first."
    return None, f"No epoch with product_id matching {query!r}."


def _legend_only_scatter(
    name: str,
    *,
    mode: str = "markers",
    marker: dict | None = None,
    line: dict | None = None,
) -> go.Scatter:
    return go.Scatter(
        x=[None],
        y=[None],
        mode=mode,
        name=name,
        marker=marker,
        line=line,
        visible="legendonly",
        showlegend=True,
    )


_LC_POINT_SIZE = 7
_SYNDIFF_MARKER = dict(size=_LC_POINT_SIZE, color="steelblue")
_SELECTED_MARKER = dict(size=12, color="black", symbol="circle-open", line_width=2)
_TESS_RAW_MARKER = dict(size=_LC_POINT_SIZE, color="#6a1b9a", opacity=0.65)
_TESS_BINNED_MARKER = dict(size=10, symbol="diamond", color="#26a69a")
_TESS_SG_LINE = dict(color="#ec407a", width=2)


def _ds9_button_grid(buttons: list[tuple[str, str]]) -> html.Div:
    return html.Div(
        [html.Button(label, id=btn_id, n_clicks=0, style={"margin": "2px 4px 2px 0"}) for btn_id, label in buttons],
        style={"display": "flex", "flexWrap": "wrap"},
    )


def _ds9_open_mode_options() -> list[dict[str, str]]:
    options: list[dict[str, str]] = [
        {"label": "XPA (scale + regions)", "value": "xpa"},
        {"label": "ds9 command line", "value": "cli"},
    ]
    if sys.platform == "darwin":
        options.insert(1, {"label": "macOS open -a", "value": "open"})
    return options


def _default_ds9_open_mode(cfg: ReviewConfig) -> str:
    mode = cfg.ds9_open_mode
    allowed = {o["value"] for o in _ds9_open_mode_options()}
    return mode if mode in allowed else "xpa"


def _list_event_labels(cfg: ReviewConfig) -> list[str]:
    events = list_events(cfg.source_mount_expanded) or list_events(cfg.data_mount_expanded)
    if not events:
        events = [cfg.default_event]
    if cfg.default_event not in events:
        events = [cfg.default_event, *events]
    return events


def _workspace_options(cfg: ReviewConfig, event: str) -> tuple[list[dict[str, str]], list[str]]:
    workspaces = list_workspaces(cfg.event_dir(event))
    options = [{"label": w, "value": w} for w in workspaces]
    return options, workspaces


def _photometry_options(
    cfg: ReviewConfig, event: str, workspace: str | None
) -> tuple[list[dict[str, str]], str | None]:
    if not workspace:
        return [], None
    ws_dir = cfg.event_dir(event) / workspace
    phot_dirs = list_photometry_dirs(ws_dir)
    options = [{"label": d, "value": d} for d in phot_dirs]
    default = parse_diff_config(ws_dir / "diff_config.yaml").lc_dir if phot_dirs else None
    return options, default


def _target_options(
    cfg: ReviewConfig, event: str, workspace: str | None
) -> tuple[list[dict[str, str]], str]:
    if not workspace:
        return [], cfg.default_lc
    labels = parse_diff_config(cfg.event_dir(event) / workspace / "diff_config.yaml")
    options = [{"label": name, "value": name} for name, _fname in list_lightcurve_options(labels)]
    return options, cfg.default_lc


def create_app(cfg: ReviewConfig) -> Dash:
    app = Dash(__name__, suppress_callback_exceptions=True)
    ds9 = Ds9Controller(
        ds9_path=cfg.ds9_path,
        ds9_xpa_dir=cfg.ds9_xpa_dir,
        diff_scale=cfg.ds9_diff_scale,
        percentile_scale=cfg.ds9_percentile_scale,
        open_mode=_default_ds9_open_mode(cfg),
    )
    mount_ok, mount_msg = is_healthy(
        cfg.data_mount_expanded,
        cfg.default_event,
        metadata_root=cfg.data_mount_expanded,
        fits_root=cfg.source_mount_expanded,
    )
    events = _list_event_labels(cfg)

    app.layout = html.Div(
        [
            html.Div(
                [
                    html.H3("SynDiff LC Review", style={"margin": 0}),
                    dcc.Loading(
                        id="lists-loading",
                        type="circle",
                        color="#1976d2",
                        children=html.Div(
                            [
                                dcc.Dropdown(
                                    id="event-select",
                                    options=[{"label": e, "value": e} for e in events],
                                    value=cfg.default_event,
                                    clearable=False,
                                    style={"width": "280px"},
                                ),
                                dcc.Dropdown(
                                    id="workspace-select",
                                    options=[],
                                    value=cfg.default_workspace,
                                    clearable=False,
                                    placeholder="Workspace",
                                    style={"width": "200px"},
                                ),
                                dcc.Dropdown(
                                    id="photometry-select",
                                    options=[],
                                    value=None,
                                    clearable=False,
                                    placeholder="Photometry",
                                    style={"width": "200px"},
                                ),
                                dcc.Dropdown(
                                    id="target-select",
                                    options=[],
                                    value=cfg.default_lc,
                                    clearable=False,
                                    placeholder="Target",
                                    style={"width": "180px"},
                                ),
                            ],
                            style={"display": "flex", "gap": "12px", "alignItems": "center", "flexWrap": "wrap"},
                        ),
                    ),
                    html.Button(
                        "Refresh lists",
                        id="reload-btn",
                        n_clicks=0,
                        title="Re-scan NFS for new events and workspaces. Does not reload the plot.",
                    ),
                    html.Button(
                        "Show TESSreduce",
                        id="tessreduce-toggle",
                        n_clicks=0,
                        disabled=False,
                        title="Overlay TESSreduce light curve",
                    ),
                    dcc.Store(id="tessreduce-visible", data=False),
                    html.Span(
                        mount_msg,
                        id="mount-status",
                        style={
                            "color": "#2e7d32" if mount_ok else "#c62828",
                            "fontWeight": "bold",
                        },
                    ),
                ],
                style={
                    "display": "flex",
                    "gap": "12px",
                    "alignItems": "center",
                    "padding": "10px",
                    "borderBottom": "1px solid #ddd",
                    "flexWrap": "wrap",
                },
            ),
            html.Div(
                mount_msg if not mount_ok else "",
                id="mount-banner",
                style={
                    "display": "block" if not mount_ok else "none",
                    "background": "#fff3cd",
                    "padding": "8px 12px",
                    "borderBottom": "1px solid #ffeeba",
                },
            )
            if not mount_ok
            else html.Div(),
            html.Div(
                [
                    html.Div(
                        [dcc.Graph(id="lc-plot", style={"height": "72vh"})],
                        style={"flex": "7", "minWidth": "420px"},
                    ),
                    html.Div(
                        [
                            html.H4("Epoch"),
                            html.Div(
                                [
                                    html.Label("Product ID", style={"fontSize": "13px"}),
                                    html.Div(
                                        [
                                            dcc.Input(
                                                id="product-id-search",
                                                type="text",
                                                placeholder="tess2020…",
                                                style={"flex": "1", "minWidth": "0"},
                                            ),
                                            html.Button("Select", id="product-id-select-btn", n_clicks=0),
                                        ],
                                        style={"display": "flex", "gap": "6px", "marginTop": "4px"},
                                    ),
                                    html.Span(
                                        id="product-id-search-status",
                                        style={"fontSize": "12px", "color": "#616161"},
                                    ),
                                ],
                                style={"marginBottom": "10px"},
                            ),
                            html.Div(id="epoch-meta"),
                            html.Hr(),
                            html.H5("Selected FFI"),
                            _ds9_button_grid(
                                [
                                    ("btn-diff", "Open Diff"),
                                    ("btn-sci", "Open FFI"),
                                    ("btn-template", "Open Template"),
                                    ("btn-conv-template", "Open Conv Template"),
                                    ("btn-bkg", "Open Background"),
                                    ("btn-mask", "Open Mask"),
                                ]
                            ),
                            html.Div(
                                id="kernel-ds9-section",
                                children=[
                                    html.Hr(),
                                    html.H5("Kernel Determination"),
                                    _ds9_button_grid(
                                        [
                                            ("btn-kernel-ref", "Kernel Reference"),
                                            ("btn-kernel-template", "Open Template"),
                                            ("btn-hp1-diff", "hp1 diff"),
                                            ("btn-hp1-bkg", "hp1 bkg"),
                                            ("btn-hp1-phot-bkg", "bkg from hp1 diff"),
                                            ("btn-sci1-clean", "cleaned FFI"),
                                            ("btn-hp2-diff", "hp2 diff"),
                                            ("btn-hp2-bkg", "hp2 bkg (0th order)"),
                                            ("btn-kernel-mask", "Open Mask"),
                                        ]
                                    ),
                                ],
                                style={"display": "none"},
                            ),
                            html.Div(id="ds9-status", style={"marginTop": "10px"}),
                        ],
                        style={
                            "flex": "3",
                            "minWidth": "260px",
                            "padding": "12px",
                            "borderLeft": "1px solid #ddd",
                        },
                    ),
                ],
                style={"display": "flex", "flexWrap": "wrap"},
            ),
            html.Div(
                [
                    html.Label("Smoothing"),
                    dcc.Dropdown(
                        id="smooth-mode",
                        options=[
                            {"label": "None", "value": "None"},
                            {"label": "Binned σ-clip", "value": "Binned σ-clip"},
                            {"label": "Savitzky-Golay", "value": "Savitzky-Golay"},
                        ],
                        value="Binned σ-clip",
                        clearable=False,
                        style={"width": "180px"},
                    ),
                    html.Div(
                        id="binned-controls",
                        children=[
                            html.Small("Binning"),
                            html.Label("Bin hours"),
                            dcc.Slider(
                                id="bin-hours",
                                min=1,
                                max=24,
                                step=1,
                                value=int(cfg.bin_width_hours),
                                marks={1: "1", 6: "6", 12: "12", 24: "24"},
                            ),
                            html.Label("σ"),
                            dcc.Slider(
                                id="bin-sigma",
                                min=1,
                                max=5,
                                step=0.5,
                                value=float(cfg.bin_sigma),
                            ),
                        ],
                    ),
                    html.Div(
                        id="gap-controls",
                        children=[
                            html.Small("Gap segmentation"),
                            html.Label("Gap threshold (days)"),
                            dcc.Slider(
                                id="gap-threshold",
                                min=0.25,
                                max=5,
                                step=0.25,
                                value=float(cfg.gap_threshold_days),
                            ),
                            dcc.Checklist(
                                id="gap-auto",
                                options=[{"label": "Auto-split largest gap", "value": "auto"}],
                                value=["auto"] if cfg.gap_auto else [],
                            ),
                        ],
                    ),
                    html.Div(
                        id="savgol-controls",
                        children=[
                            html.Small("Savitzky-Golay"),
                            html.Label("SG window"),
                            dcc.Slider(
                                id="sg-window",
                                min=3,
                                max=51,
                                step=2,
                                value=int(cfg.savgol_window),
                            ),
                            html.Label("SG polyorder"),
                            dcc.Slider(
                                id="sg-poly",
                                min=1,
                                max=5,
                                step=1,
                                value=int(cfg.savgol_polyorder),
                            ),
                        ],
                    ),
                    html.Div(
                        id="tessreduce-offset-controls",
                        children=[
                            html.Small("TESSreduce display offset"),
                            html.Label("Flux offset"),
                            dcc.Input(
                                id="tessreduce-flux-offset",
                                type="number",
                                value=0,
                                step=0.01,
                                style={"width": "100px"},
                            ),
                        ],
                        style={"display": "none"},
                    ),
                    html.Div(
                        id="plot-display-controls",
                        children=[
                            html.Small("Plot display"),
                            dcc.Checklist(
                                id="show-errorbars",
                                options=[{"label": "Show error bars", "value": "show"}],
                                value=["show"],
                            ),
                        ],
                    ),
                    html.Div(
                        [
                            html.Label("DS9 open"),
                            dcc.Dropdown(
                                id="ds9-open-mode",
                                options=_ds9_open_mode_options(),
                                value=_default_ds9_open_mode(cfg),
                                clearable=False,
                                style={"width": "220px"},
                            ),
                        ],
                        style={"marginLeft": "auto", "alignSelf": "end"},
                    ),
                ],
                style={
                    "display": "flex",
                    "flexWrap": "wrap",
                    "gap": "10px",
                    "padding": "12px",
                    "borderTop": "1px solid #ddd",
                    "alignItems": "end",
                },
            ),
            dcc.Store(id="event-index-store"),
            dcc.Store(id="selected-epoch", data=None),
            dcc.Store(id="plot-click-listener", data=None),
            dcc.Store(id="plot-load-token", data=0),
        ],
        style={"fontFamily": "system-ui, sans-serif"},
    )

    @callback(
        Output("workspace-select", "options"),
        Output("workspace-select", "value"),
        Input("event-select", "value"),
        State("workspace-select", "value"),
    )
    def update_workspace_options(event: str, current_ws: str | None):
        try:
            options, _workspaces = _workspace_options(cfg, event)
            value = _pick_dropdown_value(current_ws, options, cfg.default_workspace)
            return options, value
        except Exception:
            log.exception("Failed to list workspaces")
            return [], cfg.default_workspace

    @callback(
        Output("photometry-select", "options"),
        Output("photometry-select", "value"),
        Input("event-select", "value"),
        Input("workspace-select", "value"),
        State("photometry-select", "value"),
    )
    def update_photometry_options(event: str, workspace: str | None, current_lc_dir: str | None):
        if not workspace:
            return [], None
        try:
            options, default = _photometry_options(cfg, event, workspace)
            value = _pick_dropdown_value(current_lc_dir, options, default or "")
            return options, value
        except Exception:
            log.exception("Failed to list photometry dirs")
            return [], None

    @callback(
        Output("target-select", "options"),
        Output("target-select", "value"),
        Output("plot-load-token", "data"),
        Input("event-select", "value"),
        Input("workspace-select", "value"),
        Input("photometry-select", "value"),
        Input("target-select", "value"),
        State("plot-load-token", "data"),
    )
    def update_target_options(
        event: str,
        workspace: str | None,
        _lc_dir: str | None,
        target: str | None,
        token: int | None,
    ):
        if not workspace:
            return [], cfg.default_lc, token or 0
        try:
            options, default = _target_options(cfg, event, workspace)
            value = _pick_dropdown_value(target, options, default)
            next_token = (token or 0) + 1
            return options, value, next_token
        except Exception:
            log.exception("Failed to list targets")
            return [], cfg.default_lc, token or 0

    @callback(
        Output("event-select", "options"),
        Output("event-select", "value"),
        Output("workspace-select", "options", allow_duplicate=True),
        Output("workspace-select", "value", allow_duplicate=True),
        Output("photometry-select", "options", allow_duplicate=True),
        Output("photometry-select", "value", allow_duplicate=True),
        Output("target-select", "options", allow_duplicate=True),
        Output("target-select", "value", allow_duplicate=True),
        Input("reload-btn", "n_clicks"),
        State("event-select", "value"),
        State("workspace-select", "value"),
        State("photometry-select", "value"),
        State("target-select", "value"),
        prevent_initial_call=True,
    )
    def refresh_lists(
        _n: int,
        cur_event: str | None,
        cur_ws: str | None,
        cur_lc: str | None,
        cur_target: str | None,
    ):
        clear_index_cache()
        clear_tessreduce_cache()
        clear_store_payload_cache()
        try:
            if cfg.sync_on_start and cur_event:
                sync_event_metadata(cfg.source_mount_expanded, cfg.cache_root_expanded, cur_event)
            event_labels = _list_event_labels(cfg)
            event_options = [{"label": e, "value": e} for e in event_labels]
            event = _pick_dropdown_value(cur_event, event_options, cfg.default_event)

            ws_options, _ = _workspace_options(cfg, event)
            workspace = _pick_dropdown_value(cur_ws, ws_options, cfg.default_workspace)

            phot_options, phot_default = _photometry_options(cfg, event, workspace)
            lc_dir = _pick_dropdown_value(cur_lc, phot_options, phot_default or "")

            tgt_options, tgt_default = _target_options(cfg, event, workspace)
            target = _pick_dropdown_value(cur_target, tgt_options, tgt_default)

            return (
                event_options,
                event,
                ws_options,
                workspace,
                phot_options,
                lc_dir if phot_options else None,
                tgt_options,
                target,
            )
        except Exception:
            log.exception("Failed to refresh lists")
            return (no_update,) * 8

    @callback(
        Output("event-index-store", "data"),
        Output("mount-status", "children"),
        Output("mount-status", "style"),
        Input("plot-load-token", "data"),
        State("event-select", "value"),
        State("workspace-select", "value"),
        State("photometry-select", "value"),
        State("target-select", "value"),
    )
    def load_index(
        _token: int,
        event: str,
        workspace: str | None,
        lc_dir: str | None,
        target: str,
    ):
        if not workspace or not lc_dir:
            return None, "Select event, workspace, and photometry", {"color": "#c62828", "fontWeight": "bold"}
        cache_key = _store_payload_key(event, workspace, lc_dir, target)
        cached_payload = _store_payload_cache.get(cache_key)
        if cached_payload is not None:
            return cached_payload
        fits_event = cfg.source_event_dir(event)
        if master_index_is_cached(fits_event, workspace):
            ok, msg = True, f"OK: {event}/{workspace}/{lc_dir}"
        else:
            ok, msg = is_healthy(
                cfg.data_mount_expanded,
                event,
                workspace_subdir=workspace,
                lc_dir=lc_dir,
                metadata_root=cfg.data_mount_expanded,
                fits_root=cfg.source_mount_expanded,
            )
        style = {"color": "#2e7d32" if ok else "#c62828", "fontWeight": "bold"}
        try:
            idx = EventIndex.load(
                cfg.event_dir(event),
                workspace_subdir=workspace,
                lc_dir=lc_dir,
                lc_name=target,
                fits_event_dir=fits_event,
            )
            tess = load_tessreduce_for_event(event, cfg.tessreduce_root_expanded)
            store = {
                "event": event,
                "workspace": workspace,
                "lc_dir": lc_dir,
                "lc_name": target,
                "epochs": idx.epochs.to_dict(orient="list"),
                "regions_path": str(idx.regions_path) if idx.regions_path.is_file() else None,
                "fits_event_dir": str(fits_event),
                "crop_bounds": idx.crop_bounds,
                "tessreduce": tessreduce_store_payload(tess),
                **idx.kernel_workspace_paths(),
            }
            result: _StorePayload = (store, msg, style)
            _store_payload_cache[cache_key] = result
            return result
        except Exception as exc:
            return None, f"Load failed: {exc}", style

    @callback(
        Output("selected-epoch", "data"),
        Input("event-index-store", "data"),
        prevent_initial_call=True,
    )
    def clear_epoch_on_store_change(_store: dict | None):
        return None

    @callback(
        Output("product-id-search-status", "children", allow_duplicate=True),
        Input("event-index-store", "data"),
        prevent_initial_call=True,
    )
    def clear_product_id_search_status(_store: dict | None):
        return ""

    app.clientside_callback(
        """
        function(_figure) {
            const attach = () => {
                const host = document.getElementById("lc-plot");
                if (!host) {
                    return;
                }
                const plot = host.querySelector(".js-plotly-plot");
                if (!plot) {
                    return;
                }
                plot.removeAllListeners("plotly_click");
                plot.on("plotly_click", (evt) => {
                    const points = evt.points || [];
                    let epoch = null;
                    if (points.length > 0) {
                        const custom = points[0].customdata;
                        if (custom !== undefined && custom !== null) {
                            epoch = custom;
                        }
                    }
                    window.dash_clientside.set_props("selected-epoch", {data: epoch});
                    window.dash_clientside.set_props("lc-plot", {selectedData: null});
                });
            };
            setTimeout(attach, 0);
            return Date.now();
        }
        """,
        Output("plot-click-listener", "data"),
        Input("lc-plot", "figure"),
    )

    @callback(
        Output("tessreduce-visible", "data"),
        Output("tessreduce-toggle", "children"),
        Output("tessreduce-toggle", "style"),
        Output("tessreduce-toggle", "disabled"),
        Input("tessreduce-toggle", "n_clicks"),
        Input("event-index-store", "data"),
        State("tessreduce-visible", "data"),
    )
    def update_tessreduce_ui(
        _n_clicks: int, store: dict | None, visible: bool
    ):
        tess = (store or {}).get("tessreduce") or {}
        unavailable_label = "TESSreduce (unavailable)"
        unavailable_style = {"opacity": "0.5"}

        if ctx.triggered_id != "tessreduce-toggle":
            if not tess.get("available"):
                return False, unavailable_label, unavailable_style, True
            return False, "Show TESSreduce", {}, False

        if not tess.get("available"):
            return False, unavailable_label, {**unavailable_style, "cursor": "not-allowed"}, True

        new_visible = not bool(visible)
        label = "Hide TESSreduce" if new_visible else "Show TESSreduce"
        style = {
            "background": "#6a1b9a" if new_visible else "",
            "color": "white" if new_visible else "",
        }
        return new_visible, label, style, False

    @callback(
        Output("tessreduce-offset-controls", "style"),
        Input("tessreduce-visible", "data"),
    )
    def toggle_tessreduce_offset_controls(visible: bool):
        if visible:
            return {"display": "block"}
        return {"display": "none"}

    @callback(
        Output("tessreduce-flux-offset", "value"),
        Input("event-index-store", "data"),
    )
    def reset_tessreduce_offset_on_event(_store: dict | None):
        return 0.0

    @callback(
        Output("lc-plot", "figure"),
        Input("event-index-store", "data"),
        Input("smooth-mode", "value"),
        Input("bin-hours", "value"),
        Input("bin-sigma", "value"),
        Input("gap-threshold", "value"),
        Input("gap-auto", "value"),
        Input("sg-window", "value"),
        Input("sg-poly", "value"),
        Input("selected-epoch", "data"),
        Input("tessreduce-visible", "data"),
        Input("tessreduce-flux-offset", "value"),
        Input("show-errorbars", "value"),
    )
    def update_plot(
        store: dict | None,
        smooth_mode: str,
        bin_hours: float,
        bin_sigma: float,
        gap_threshold: float,
        gap_auto_vals: list[str],
        sg_window: int,
        sg_poly: int,
        selected_epoch: int | None,
        show_tessreduce: bool,
        tessreduce_flux_offset: float | int | str | None,
        show_errorbars_vals: list[str],
    ):
        if not store:
            return go.Figure()

        df = pd.DataFrame(store["epochs"])
        if df.empty:
            return go.Figure()

        mode = _smoothing_mode_from_ui(smooth_mode)
        gap_auto = "auto" in (gap_auto_vals or [])
        smooth = apply_smoothing(
            df["btjd"],
            df["flux"],
            mode=mode,
            bin_width_hours=float(bin_hours),
            bin_sigma=float(bin_sigma),
            gap_threshold_days=float(gap_threshold),
            gap_auto=gap_auto,
            savgol_window=int(sg_window),
            savgol_polyorder=int(sg_poly),
        )

        fig = go.Figure()
        show_errorbars = "show" in (show_errorbars_vals or [])
        syndiff_yerr = df["eflux"].where(df["eflux"].notna(), None) if show_errorbars else None
        fig.add_trace(
            go.Scatter(
                x=df["btjd"],
                y=df["flux"],
                error_y=dict(type="data", array=syndiff_yerr, visible=show_errorbars),
                mode="markers",
                name="Syndiff",
                marker=_SYNDIFF_MARKER,
                customdata=df["epoch_idx"].tolist(),
            )
        )

        if mode == "binned" and smooth.binned_t.size:
            fig.add_trace(
                go.Scatter(
                    x=smooth.binned_t,
                    y=smooth.binned_flux,
                    mode="markers",
                    name="binned σ-clip",
                    marker=dict(size=10, symbol="diamond", color="darkorange"),
                )
            )
            rejected = df.loc[~smooth.clip_keep_mask]
            if not rejected.empty:
                fig.add_trace(
                    go.Scatter(
                        x=rejected["btjd"],
                        y=rejected["flux"],
                        mode="markers",
                        name="rejected",
                        marker=dict(size=7, symbol="x", color="crimson"),
                        customdata=rejected["epoch_idx"].tolist(),
                    )
                )
        elif mode == "savgol":
            fig.add_trace(
                go.Scatter(
                    x=df["btjd"],
                    y=smooth.savgol_flux,
                    mode="lines",
                    name="Savitzky-Golay",
                    line=dict(color="darkorange", width=2),
                )
            )

        for start in smooth.segment_starts[1:]:
            if 0 <= start < len(df):
                fig.add_vline(x=float(df.iloc[start]["btjd"]), line_dash="dot", line_color="#888")

        if selected_epoch is not None:
            sel = df.loc[df["epoch_idx"] == selected_epoch]
            if not sel.empty:
                fig.add_trace(
                    go.Scatter(
                        x=sel["btjd"],
                        y=sel["flux"],
                        mode="markers",
                        name="selected",
                        marker=_SELECTED_MARKER,
                    )
                )
            else:
                fig.add_trace(_legend_only_scatter("selected", marker=_SELECTED_MARKER))
        else:
            fig.add_trace(_legend_only_scatter("selected", marker=_SELECTED_MARKER))

        tess = store.get("tessreduce") or {}
        tess_available = bool(tess.get("available") and tess.get("btjd"))
        if tess_available and show_tessreduce:
            tr_btjd = tess["btjd"]
            offset = _parse_flux_offset(tessreduce_flux_offset)
            tr_flux = [float(f) + offset for f in tess["flux"]]
            tr_eflux = tess.get("eflux") or []
            tr_smooth = apply_smoothing(
                tr_btjd,
                tr_flux,
                mode=mode,
                bin_width_hours=float(bin_hours),
                bin_sigma=float(bin_sigma),
                gap_threshold_days=float(gap_threshold),
                gap_auto=gap_auto,
                savgol_window=int(sg_window),
                savgol_polyorder=int(sg_poly),
            )
            tess_yerr = tr_eflux if show_errorbars and tr_eflux else None
            fig.add_trace(
                go.Scatter(
                    x=tr_btjd,
                    y=tr_flux,
                    error_y=dict(
                        type="data",
                        array=tess_yerr,
                        visible=show_errorbars and bool(tr_eflux),
                    ),
                    mode="markers",
                    name="TESSreduce",
                    marker=_TESS_RAW_MARKER,
                )
            )
            if mode == "binned" and tr_smooth.binned_t.size:
                fig.add_trace(
                    go.Scatter(
                        x=tr_smooth.binned_t,
                        y=tr_smooth.binned_flux,
                        mode="markers",
                        name="TESSreduce binned",
                        marker=_TESS_BINNED_MARKER,
                    )
                )
            elif mode == "savgol":
                fig.add_trace(
                    go.Scatter(
                        x=tr_btjd,
                        y=tr_smooth.savgol_flux,
                        mode="lines",
                        name="TESSreduce SG",
                        line=_TESS_SG_LINE,
                    )
                )
        elif tess_available:
            fig.add_trace(
                _legend_only_scatter(
                    "TESSreduce",
                    marker=_TESS_RAW_MARKER,
                )
            )
            if mode == "binned":
                fig.add_trace(
                    _legend_only_scatter(
                        "TESSreduce binned",
                        marker=_TESS_BINNED_MARKER,
                    )
                )
            elif mode == "savgol":
                fig.add_trace(
                    _legend_only_scatter(
                        "TESSreduce SG",
                        mode="lines",
                        line=_TESS_SG_LINE,
                    )
                )

        fig.update_layout(
            margin=dict(l=40, r=20, t=50, b=40),
            xaxis_title="BTJD",
            yaxis_title="Flux",
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            uirevision=store.get("event"),
            clickmode="event",
            clickanywhere=True,
        )
        return fig

    @callback(
        Output("selected-epoch", "data", allow_duplicate=True),
        Output("product-id-search-status", "children"),
        Output("product-id-search-status", "style"),
        Input("product-id-select-btn", "n_clicks"),
        Input("product-id-search", "n_submit"),
        State("product-id-search", "value"),
        State("event-index-store", "data"),
        prevent_initial_call=True,
    )
    def select_epoch_by_product_id(
        _n_clicks: int, _n_submit: int, query: str | None, store: dict | None
    ):
        epoch_idx, message = find_epoch_idx_by_product_id((store or {}).get("epochs") or {}, query)
        if epoch_idx is None:
            return no_update, message, {"fontSize": "12px", "color": "#c62828"}
        note = message or "Epoch selected."
        return epoch_idx, note, {"fontSize": "12px", "color": "#2e7d32"}

    @callback(
        Output("epoch-meta", "children"),
        Input("event-index-store", "data"),
        Input("selected-epoch", "data"),
    )
    def update_sidebar(store: dict | None, epoch_idx: int | None):
        if not store or epoch_idx is None:
            return "Click a Syndiff point to select an epoch."
        df = pd.DataFrame(store["epochs"])
        row = df.loc[df["epoch_idx"] == epoch_idx]
        if row.empty:
            return "Epoch not found."
        r = row.iloc[0]
        product_id = r.get("product_id")
        product_id_text = str(product_id) if product_id else "—"
        product_id_row: list[Any] = [html.Span("product_id: ")]
        if product_id:
            product_id_row.extend(
                [
                    html.Code(
                        product_id_text,
                        id="epoch-product-id",
                        style={"fontSize": "12px"},
                    ),
                    dcc.Clipboard(
                        target_id="epoch-product-id",
                        title="Copy product ID",
                        style={
                            "display": "inline-block",
                            "marginLeft": "6px",
                            "cursor": "pointer",
                            "fontSize": "12px",
                        },
                    ),
                ]
            )
        else:
            product_id_row.append(html.Span("—"))
        return html.Div(
            [
                html.Div(f"epoch_idx: {int(r['epoch_idx'])}"),
                html.Div(f"BTJD: {r['btjd']:.6f}"),
                html.Div(f"flux: {r['flux']:.6g} ± {r['eflux']:.6g}"),
                html.Div(f"SNR: {r['snr']:.2f}" if pd.notna(r["snr"]) else "SNR: —"),
                html.Div(
                    product_id_row,
                    style={"display": "flex", "alignItems": "center", "flexWrap": "wrap", "gap": "4px"},
                ),
                html.Div(f"group_id: {r.get('group_id') if pd.notna(r.get('group_id')) else '—'}"),
                html.Div(f"hotpants_ok: {r.get('hotpants_ok')}"),
            ]
        )

    def _row_from_store(store: dict | None, epoch_idx: int | None) -> pd.Series | None:
        if not store or epoch_idx is None:
            return None
        df = pd.DataFrame(store["epochs"])
        row = df.loc[df["epoch_idx"] == epoch_idx]
        return None if row.empty else row.iloc[0]

    @callback(
        Output("kernel-ds9-section", "style"),
        Input("event-index-store", "data"),
    )
    def toggle_kernel_section(store: dict | None):
        if store and store.get("has_kernel_fit"):
            return {"display": "block"}
        return {"display": "none"}

    @callback(
        Output("binned-controls", "style"),
        Output("gap-controls", "style"),
        Output("savgol-controls", "style"),
        Input("smooth-mode", "value"),
    )
    def toggle_smoothing_controls(mode: str):
        show = {"display": "block"}
        hide = {"display": "none"}
        if mode == "Binned σ-clip":
            return show, show, hide
        if mode == "Savitzky-Golay":
            return hide, show, show
        return hide, show, hide

    _DS9_BTN_IDS = (
        "btn-diff",
        "btn-sci",
        "btn-template",
        "btn-conv-template",
        "btn-bkg",
        "btn-mask",
        "btn-kernel-ref",
        "btn-kernel-template",
        "btn-hp1-diff",
        "btn-hp1-bkg",
        "btn-hp1-phot-bkg",
        "btn-sci1-clean",
        "btn-hp2-diff",
        "btn-hp2-bkg",
        "btn-kernel-mask",
    )

    @callback(
        Output("ds9-status", "children"),
        [Input(btn_id, "n_clicks") for btn_id in _DS9_BTN_IDS],
        State("event-index-store", "data"),
        State("selected-epoch", "data"),
        State("ds9-open-mode", "value"),
        prevent_initial_call=True,
    )
    def ds9_buttons(*args):
        n_clicks = args[: len(_DS9_BTN_IDS)]
        store = args[len(_DS9_BTN_IDS)]
        epoch_idx = args[len(_DS9_BTN_IDS) + 1]
        open_mode = args[len(_DS9_BTN_IDS) + 2]

        if not ctx.triggered_id:
            return no_update

        ds9.open_mode = open_mode if open_mode in ("xpa", "open", "cli") else "xpa"

        btn = ctx.triggered_id
        regions = store.get("regions_path") if store else None
        mask_path = store.get("mask_path") if store else None
        crop_bounds = store.get("crop_bounds") if store else None
        event_key = store.get("event") if store else "event"
        workspace = store.get("workspace") if store else "ws"

        def _resolve_cropped_path(
            path: str | None,
            *,
            kind: str,
            label: str,
        ) -> tuple[str | None, str | None]:
            """Return (display_path, warning_message)."""
            if not path:
                return None, None
            if crop_bounds is None:
                return path, f"No targets.reg ROI; showing full frame for {label}."
            try:
                cropped = ensure_cropped_fits(
                    path,
                    kind=kind,  # type: ignore[arg-type]
                    crop_bounds=crop_bounds,
                    cache_root=cfg.cache_root_expanded,
                    event_key=str(event_key),
                    workspace=str(workspace),
                )
                return str(cropped), None
            except Exception as exc:
                log.exception("Crop failed for %s", label)
                return path, f"Crop failed for {label}; showing full frame ({exc})."

        def _enqueue(path: str | None, *, is_diff: bool, label: str, needs_epoch: bool = True):
            if needs_epoch and epoch_idx is None:
                return "Select an epoch first."
            if not path:
                return f"No path for {label}."
            res = ds9.enqueue_load(path, regions=regions, is_diff=is_diff, label=label)
            color = "#2e7d32" if res.ok else "#c62828"
            return html.Span(res.message, style={"color": color})

        def _enqueue_cropped(
            path: str | None,
            *,
            kind: str,
            label: str,
            needs_epoch: bool = True,
        ):
            if needs_epoch and epoch_idx is None:
                return "Select an epoch first."
            if not path:
                return f"No path for {label}."
            display_path, warning = _resolve_cropped_path(path, kind=kind, label=label)
            if not display_path:
                return f"No path for {label}."
            res = ds9.enqueue_load(display_path, regions=regions, is_diff=False, label=label)
            color = "#2e7d32" if res.ok else "#c62828"
            msg = res.message
            if warning and res.ok:
                msg = f"{msg} ({warning})"
            return html.Span(msg, style={"color": color})

        row = _row_from_store(store, epoch_idx)

        if btn == "btn-diff":
            return _enqueue(row["diff_path"] if row is not None else None, is_diff=True, label="diff")
        if btn == "btn-sci":
            return _enqueue_cropped(
                row["sci_path"] if row is not None else None, kind="ffi", label="FFI"
            )
        if btn == "btn-template":
            return _enqueue_cropped(
                row["template_path"] if row is not None else None, kind="template", label="template"
            )
        if btn == "btn-conv-template":
            path = row.get("conv_template_path") or row.get("conv_path") if row is not None else None
            return _enqueue(path, is_diff=False, label="conv template")
        if btn == "btn-bkg":
            return _enqueue(row["bkg_path"] if row is not None else None, is_diff=False, label="background")
        if btn == "btn-mask":
            return _enqueue(mask_path, is_diff=False, label="mask", needs_epoch=False)

        if btn == "btn-kernel-ref":
            return _enqueue(
                store.get("kernel_reference_path"), is_diff=False, label="kernel reference", needs_epoch=False
            )
        if btn == "btn-kernel-template":
            return _enqueue(
                store.get("kernel_template_path"), is_diff=False, label="template", needs_epoch=False
            )
        if btn == "btn-kernel-mask":
            return _enqueue(mask_path, is_diff=False, label="mask", needs_epoch=False)
        if btn == "btn-hp1-phot-bkg":
            return _enqueue(
                store.get("kernel_phot_bkg_fine_path"),
                is_diff=False,
                label="phot bkg fine",
                needs_epoch=False,
            )
        if btn == "btn-sci1-clean":
            return _enqueue(
                store.get("kernel_sci1_clean_path"), is_diff=False, label="sci1 clean", needs_epoch=False
            )
        if btn == "btn-hp1-diff":
            return _enqueue(store.get("kernel_hp1_diff_path"), is_diff=True, label="hp1 diff", needs_epoch=False)
        if btn == "btn-hp1-bkg":
            return _enqueue(store.get("kernel_hp1_bkg_path"), is_diff=False, label="hp1 bkg", needs_epoch=False)
        if btn == "btn-hp2-diff":
            return _enqueue(store.get("kernel_hp2_diff_path"), is_diff=True, label="hp2 diff", needs_epoch=False)
        if btn == "btn-hp2-bkg":
            return _enqueue(store.get("kernel_hp2_bkg_path"), is_diff=False, label="hp2 bkg", needs_epoch=False)

        return no_update

    return app


def run_app(cfg: ReviewConfig) -> None:
    if cfg.sync_on_start:
        result = sync_workspace_metadata(cfg.source_mount_expanded, cfg.cache_root_expanded)
        log.info("Cache sync: %d copied, %d skipped", result.copied, result.skipped)
    app = create_app(cfg)
    app.run(host=cfg.host, port=cfg.port, debug=False)
