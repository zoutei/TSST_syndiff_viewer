"""Plotly trace helpers for SynDiff light-curve review."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go

from .smoothing import SmoothingMode, SmoothingResult

LC_POINT_SIZE = 7
ACTIVE_LC_POINT_SIZE = 9
PRIMARY_MARKER = dict(size=LC_POINT_SIZE, color="steelblue")
ACTIVE_PRIMARY_MARKER = dict(size=ACTIVE_LC_POINT_SIZE, color="steelblue")
SELECTED_MARKER = dict(size=12, color="black", symbol="circle-open", line_width=2)

OVERLAY_COLORS = ("#e65100", "#2e7d32", "#6a1b9a", "#00838f", "#c62828")

TESS_RAW_MARKER = dict(size=LC_POINT_SIZE, color="#6a1b9a", opacity=0.65)
TESS_BINNED_MARKER = dict(size=10, symbol="diamond", color="#26a69a")
TESS_SG_LINE = dict(color="#ec407a", width=2)


def overlay_marker(color_index: int, *, active: bool = False) -> dict[str, Any]:
    size = ACTIVE_LC_POINT_SIZE if active else LC_POINT_SIZE
    return dict(size=size, color=OVERLAY_COLORS[color_index % len(OVERLAY_COLORS)])


def overlay_smoothing_marker(color_index: int) -> dict[str, Any]:
    color = OVERLAY_COLORS[color_index % len(OVERLAY_COLORS)]
    return dict(size=10, symbol="diamond", color=color)


def overlay_smoothing_line(color_index: int) -> dict[str, Any]:
    color = OVERLAY_COLORS[color_index % len(OVERLAY_COLORS)]
    return dict(color=color, width=2)


def layer_trace_name(layer_label_text: str) -> str:
    return layer_label_text


def legend_only_scatter(
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


def _layer_customdata(layer_key: str, epoch_indices: pd.Series) -> list[list[str | int]]:
    return [[layer_key, int(epoch_idx)] for epoch_idx in epoch_indices]


def add_syndiff_traces(
    fig: go.Figure,
    df: pd.DataFrame,
    *,
    layer_key: str,
    name: str,
    marker: dict[str, Any],
    show_errorbars: bool,
    smooth: SmoothingResult,
    mode: SmoothingMode,
    flux_offset: float = 0.0,
    selected_epoch: int | None = None,
    show_diagnostics: bool = False,
    color_index: int | None = None,
) -> None:
    """Add raw flux markers; diagnostics only when ``show_diagnostics`` is True."""
    if df.empty:
        return

    offset = float(flux_offset)
    flux = df["flux"] + offset
    yerr = None
    if show_errorbars and "eflux" in df.columns:
        yerr = df["eflux"].where(df["eflux"].notna(), None)

    customdata = _layer_customdata(layer_key, df["epoch_idx"])

    fig.add_trace(
        go.Scatter(
            x=df["btjd"],
            y=flux,
            error_y=dict(type="data", array=yerr, visible=show_errorbars),
            mode="markers",
            name=name,
            marker=marker,
            customdata=customdata,
        )
    )

    if not show_diagnostics:
        return

    smooth_marker = (
        overlay_smoothing_marker(color_index)
        if color_index is not None
        else dict(size=10, symbol="diamond", color="darkorange")
    )
    smooth_line = (
        overlay_smoothing_line(color_index)
        if color_index is not None
        else dict(color="darkorange", width=2)
    )
    rejected_marker = dict(size=7, symbol="x", color="crimson")
    binned_name = f"{name} binned" if color_index is not None else "binned σ-clip"
    sg_name = f"{name} SG" if color_index is not None else "Savitzky-Golay"
    rejected_name = f"{name} rejected" if color_index is not None else "rejected"

    if mode == "binned" and smooth.binned_t.size:
        fig.add_trace(
            go.Scatter(
                x=smooth.binned_t,
                y=smooth.binned_flux + offset,
                mode="markers",
                name=binned_name,
                marker=smooth_marker,
            )
        )
        rejected = df.loc[~smooth.clip_keep_mask]
        if not rejected.empty:
            fig.add_trace(
                go.Scatter(
                    x=rejected["btjd"],
                    y=rejected["flux"] + offset,
                    mode="markers",
                    name=rejected_name,
                    marker=rejected_marker,
                    customdata=_layer_customdata(layer_key, rejected["epoch_idx"]),
                )
            )
    elif mode == "savgol":
        fig.add_trace(
            go.Scatter(
                x=df["btjd"],
                y=smooth.savgol_flux + offset,
                mode="lines",
                name=sg_name,
                line=smooth_line,
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
                    y=sel["flux"] + offset,
                    mode="markers",
                    name="selected",
                    marker=SELECTED_MARKER,
                )
            )
        else:
            fig.add_trace(legend_only_scatter("selected", marker=SELECTED_MARKER))
    else:
        fig.add_trace(legend_only_scatter("selected", marker=SELECTED_MARKER))


def add_tessreduce_traces(
    fig: go.Figure,
    *,
    btjd: list[float],
    flux: list[float],
    eflux: list[float] | None,
    show_errorbars: bool,
    smooth: SmoothingResult,
    mode: SmoothingMode,
    visible: bool,
) -> None:
    tess_yerr = eflux if show_errorbars and eflux else None
    if visible:
        fig.add_trace(
            go.Scatter(
                x=btjd,
                y=flux,
                error_y=dict(
                    type="data",
                    array=tess_yerr,
                    visible=show_errorbars and bool(eflux),
                ),
                mode="markers",
                name="TESSreduce",
                marker=TESS_RAW_MARKER,
            )
        )
        if mode == "binned" and smooth.binned_t.size:
            fig.add_trace(
                go.Scatter(
                    x=smooth.binned_t,
                    y=smooth.binned_flux,
                    mode="markers",
                    name="TESSreduce binned",
                    marker=TESS_BINNED_MARKER,
                )
            )
        elif mode == "savgol":
            fig.add_trace(
                go.Scatter(
                    x=btjd,
                    y=smooth.savgol_flux,
                    mode="lines",
                    name="TESSreduce SG",
                    line=TESS_SG_LINE,
                )
            )
    else:
        fig.add_trace(legend_only_scatter("TESSreduce", marker=TESS_RAW_MARKER))
        if mode == "binned":
            fig.add_trace(legend_only_scatter("TESSreduce binned", marker=TESS_BINNED_MARKER))
        elif mode == "savgol":
            fig.add_trace(legend_only_scatter("TESSreduce SG", mode="lines", line=TESS_SG_LINE))
