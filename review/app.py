"""Plotly Dash light-curve review application."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, callback, ctx, dcc, html, no_update

from .config import ReviewConfig
from .crop_cache import ensure_cropped_fits
from .ds9 import Ds9Controller
from .event_index import EventIndex
from .mount import is_healthy, list_events, list_photometry_dirs, list_workspaces
from .pipeline_labels import list_lightcurve_options, parse_diff_config
from .smoothing import SmoothingMode, apply_smoothing
from .sync_cache import sync_workspace_metadata
from .tessreduce import load_tessreduce_for_event, tessreduce_store_payload

log = logging.getLogger(__name__)


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


def create_app(cfg: ReviewConfig) -> Dash:
    app = Dash(__name__, suppress_callback_exceptions=True)
    ds9 = Ds9Controller(
        ds9_path=cfg.ds9_path,
        ds9_xpa_dir=cfg.ds9_xpa_dir,
        diff_scale=cfg.ds9_diff_scale,
        percentile_scale=cfg.ds9_percentile_scale,
    )
    mount_ok, mount_msg = is_healthy(
        cfg.data_mount_expanded,
        cfg.default_event,
        metadata_root=cfg.data_mount_expanded,
        fits_root=cfg.source_mount_expanded,
    )
    events = list_events(cfg.data_mount_expanded) or [cfg.default_event]
    if cfg.default_event not in events:
        events = [cfg.default_event] + events

    app.layout = html.Div(
        [
            html.Div(
                [
                    html.H3("SynDiff LC Review", style={"margin": 0}),
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
                    html.Button("Reload", id="reload-btn", n_clicks=0),
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
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(auto-fit, minmax(180px, 1fr))",
                    "gap": "10px",
                    "padding": "12px",
                    "borderTop": "1px solid #ddd",
                },
            ),
            dcc.Store(id="event-index-store"),
            dcc.Store(id="selected-epoch", data=None),
            dcc.Store(id="plot-click-listener", data=None),
        ],
        style={"fontFamily": "system-ui, sans-serif"},
    )

    @callback(
        Output("workspace-select", "options"),
        Output("workspace-select", "value"),
        Input("event-select", "value"),
        Input("reload-btn", "n_clicks"),
        State("workspace-select", "value"),
    )
    def update_workspace_options(event: str, _n: int, current_ws: str | None):
        try:
            workspaces = list_workspaces(cfg.event_dir(event))
            options = [{"label": w, "value": w} for w in workspaces]
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
        Input("reload-btn", "n_clicks"),
        State("photometry-select", "value"),
    )
    def update_photometry_options(event: str, workspace: str | None, _n: int, current_lc_dir: str | None):
        if not workspace:
            return [], None
        try:
            ws_dir = cfg.event_dir(event) / workspace
            phot_dirs = list_photometry_dirs(ws_dir)
            options = [{"label": d, "value": d} for d in phot_dirs]
            default = parse_diff_config(ws_dir / "diff_config.yaml").lc_dir
            value = _pick_dropdown_value(current_lc_dir, options, default)
            return options, value
        except Exception:
            log.exception("Failed to list photometry dirs")
            return [], None

    @callback(
        Output("target-select", "options"),
        Output("target-select", "value"),
        Input("event-select", "value"),
        Input("workspace-select", "value"),
        Input("photometry-select", "value"),
        Input("reload-btn", "n_clicks"),
        State("target-select", "value"),
    )
    def update_target_options(
        event: str, workspace: str | None, _lc_dir: str | None, _n: int, current_target: str | None
    ):
        if not workspace:
            return [], cfg.default_lc
        try:
            labels = parse_diff_config(cfg.event_dir(event) / workspace / "diff_config.yaml")
            options = [
                {"label": name, "value": name}
                for name, _fname in list_lightcurve_options(labels)
            ]
            value = _pick_dropdown_value(current_target, options, cfg.default_lc)
            return options, value
        except Exception:
            log.exception("Failed to list targets")
            return [], cfg.default_lc

    @callback(
        Output("event-index-store", "data"),
        Output("mount-status", "children"),
        Output("mount-status", "style"),
        Input("event-select", "value"),
        Input("workspace-select", "value"),
        Input("photometry-select", "value"),
        Input("target-select", "value"),
        Input("reload-btn", "n_clicks"),
    )
    def load_index(event: str, workspace: str | None, lc_dir: str | None, target: str, _n: int):
        if not workspace or not lc_dir:
            return None, "Select event, workspace, and photometry", {"color": "#c62828", "fontWeight": "bold"}
        if cfg.sync_on_start and ctx.triggered_id == "reload-btn":
            sync_workspace_metadata(cfg.source_mount_expanded, cfg.cache_root_expanded)
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
                fits_event_dir=cfg.source_event_dir(event),
            )
            tess = load_tessreduce_for_event(event, cfg.tessreduce_root_expanded)
            store = {
                "event": event,
                "workspace": workspace,
                "lc_dir": lc_dir,
                "lc_name": target,
                "epochs": idx.epochs.to_dict(orient="list"),
                "regions_path": str(idx.regions_path) if idx.regions_path.is_file() else None,
                "fits_event_dir": str(cfg.source_event_dir(event)),
                "crop_bounds": idx.crop_bounds,
                "tessreduce": tessreduce_store_payload(tess),
                **idx.kernel_workspace_paths(),
            }
            return store, msg, style
        except Exception as exc:
            return None, f"Load failed: {exc}", style

    @callback(
        Output("selected-epoch", "data"),
        Input("event-index-store", "data"),
        prevent_initial_call=True,
    )
    def clear_epoch_on_reload(_store: dict | None):
        return None

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
        return html.Div(
            [
                html.Div(f"epoch_idx: {int(r['epoch_idx'])}"),
                html.Div(f"BTJD: {r['btjd']:.6f}"),
                html.Div(f"flux: {r['flux']:.6g} ± {r['eflux']:.6g}"),
                html.Div(f"SNR: {r['snr']:.2f}" if pd.notna(r["snr"]) else "SNR: —"),
                html.Div(f"product_id: {r.get('product_id') or '—'}"),
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
        prevent_initial_call=True,
    )
    def ds9_buttons(*args):
        n_clicks = args[: len(_DS9_BTN_IDS)]
        store = args[len(_DS9_BTN_IDS)]
        epoch_idx = args[len(_DS9_BTN_IDS) + 1]

        if not ctx.triggered_id:
            return no_update

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
