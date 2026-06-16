"""Gap-aware binning, sigma-clipping, and Savitzky-Golay smoothing for review plots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
from scipy.signal import savgol_filter

SmoothingMode = Literal["none", "binned", "savgol"]


@dataclass(frozen=True)
class SmoothingResult:
    clip_keep_mask: np.ndarray
    binned_t: np.ndarray
    binned_flux: np.ndarray
    savgol_t: np.ndarray
    savgol_flux: np.ndarray
    segment_starts: list[int]


def sigma_clipped_mean(values: np.ndarray, *, n_sigma: float) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float(np.nan)
    if v.size == 1:
        return float(v[0])
    mu = float(np.nanmean(v))
    sig = float(np.nanstd(v))
    if not np.isfinite(sig) or sig <= 0.0:
        return mu
    lo, hi = mu - n_sigma * sig, mu + n_sigma * sig
    clipped = v[(v >= lo) & (v <= hi)]
    if clipped.size == 0:
        return float(np.nanmean(v))
    return float(np.nanmean(clipped))


def robust_std_1d(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    return float(1.4826 * max(mad, 1e-12 * (1.0 + abs(med))))


def binned_sigma_clip_btjd(
    btjd: np.ndarray,
    flux: np.ndarray,
    *,
    bin_width_days: float,
    sigma: float,
    t_anchor: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    btjd = np.asarray(btjd, dtype=float)
    flux = np.asarray(flux, dtype=float)
    mask = np.zeros_like(flux, dtype=bool)
    if btjd.size == 0:
        return mask, np.array([]), np.array([])

    tmin = float(t_anchor if t_anchor is not None else np.nanmin(btjd))
    tmax = float(np.nanmax(btjd))
    if not np.isfinite(tmin) or not np.isfinite(tmax):
        return mask, np.array([]), np.array([])

    bins = np.arange(tmin, tmax + bin_width_days, bin_width_days)
    inds = np.digitize(btjd, bins)

    binned_avg_flux_list: list[float] = []
    binned_avg_time_list: list[float] = []

    for b in np.unique(inds):
        in_bin = inds == b
        if np.sum(in_bin) < 1:
            continue
        f = flux[in_bin]
        t = btjd[in_bin]
        if np.sum(in_bin) < 3:
            mask[in_bin] = True
            binned_avg_flux_list.append(float(np.nanmean(f)))
            binned_avg_time_list.append(float(np.nanmean(t)))
            continue
        med = float(np.median(f))
        rob_std = robust_std_1d(f)
        keep = np.abs(f - med) <= sigma * rob_std
        mask[in_bin] = keep
        if np.any(keep):
            binned_avg_flux_list.append(float(np.nanmean(f[keep])))
            binned_avg_time_list.append(float(np.nanmean(t[keep])))
        else:
            binned_avg_flux_list.append(float("nan"))
            binned_avg_time_list.append(float("nan"))

    return mask, np.asarray(binned_avg_time_list, dtype=float), np.asarray(
        binned_avg_flux_list, dtype=float
    )


def split_segments(
    btjd: np.ndarray,
    *,
    gap_threshold_days: float = 1.0,
    gap_auto: bool = True,
) -> list[tuple[int, int]]:
    btjd = np.asarray(btjd, dtype=float)
    n = btjd.size
    if n == 0:
        return []
    if n == 1:
        return [(0, 1)]

    order = np.argsort(btjd)
    sorted_t = btjd[order]
    gaps = np.diff(sorted_t)
    split_after: set[int] = set()

    if gap_threshold_days > 0:
        for i, gap in enumerate(gaps):
            if gap > gap_threshold_days:
                split_after.add(i)

    if gap_auto and gaps.size > 0:
        imax = int(np.argmax(gaps))
        if gaps[imax] > 0:
            split_after.add(imax)

    segment_bounds: list[tuple[int, int]] = []
    start = 0
    for i in range(n - 1):
        if i in split_after:
            segment_bounds.append((start, i + 1))
            start = i + 1
    segment_bounds.append((start, n))

    segments: list[tuple[int, int]] = []
    for s, e in segment_bounds:
        orig_idx = order[s:e]
        if orig_idx.size == 0:
            continue
        segments.append((int(orig_idx.min()), int(orig_idx.max()) + 1))
    return segments


def _savgol_segment(
    btjd: np.ndarray,
    flux: np.ndarray,
    *,
    window: int,
    polyorder: int,
) -> np.ndarray:
    y = np.asarray(flux, dtype=float)
    n = y.size
    if n < 3:
        return y.copy()
    w = int(window)
    if w % 2 == 0:
        w += 1
    w = max(3, min(w, n if n % 2 == 1 else n - 1))
    p = min(int(polyorder), w - 1)
    if n < w:
        w = n if n % 2 == 1 else n - 1
        p = min(p, w - 1)
    if w < 3 or p < 1:
        return y.copy()
    finite = np.isfinite(y)
    if finite.sum() < w:
        return y.copy()
    out = y.copy()
    out[finite] = savgol_filter(y[finite], window_length=w, polyorder=p, mode="interp")
    return out


def apply_smoothing(
    btjd: Sequence[float],
    flux: Sequence[float],
    *,
    mode: SmoothingMode = "binned",
    bin_width_hours: float = 6.0,
    bin_sigma: float = 3.0,
    gap_threshold_days: float = 1.0,
    gap_auto: bool = True,
    savgol_window: int = 11,
    savgol_polyorder: int = 2,
) -> SmoothingResult:
    btjd_arr = np.asarray(btjd, dtype=float)
    flux_arr = np.asarray(flux, dtype=float)
    n = btjd_arr.size
    clip_mask = np.ones(n, dtype=bool)
    binned_t_parts: list[np.ndarray] = []
    binned_f_parts: list[np.ndarray] = []
    savgol_out = np.full(n, np.nan, dtype=float)
    segment_starts: list[int] = []

    segments = split_segments(
        btjd_arr, gap_threshold_days=gap_threshold_days, gap_auto=gap_auto
    )
    if not segments:
        segments = [(0, n)]

    for start, end in segments:
        segment_starts.append(start)
        seg_btjd = btjd_arr[start:end]
        seg_flux = flux_arr[start:end]
        if mode == "binned":
            seg_mask, seg_bt, seg_bf = binned_sigma_clip_btjd(
                seg_btjd,
                seg_flux,
                bin_width_days=bin_width_hours / 24.0,
                sigma=bin_sigma,
                t_anchor=float(np.nanmin(seg_btjd)),
            )
            clip_mask[start:end] = seg_mask
            if seg_bt.size:
                binned_t_parts.append(seg_bt)
                binned_f_parts.append(seg_bf)
        elif mode == "savgol":
            savgol_out[start:end] = _savgol_segment(
                seg_btjd,
                seg_flux,
                window=savgol_window,
                polyorder=savgol_polyorder,
            )

    binned_t = np.concatenate(binned_t_parts) if binned_t_parts else np.array([])
    binned_flux = np.concatenate(binned_f_parts) if binned_f_parts else np.array([])
    savgol_t = btjd_arr if mode == "savgol" else np.array([])
    savgol_flux = savgol_out if mode == "savgol" else np.array([])

    return SmoothingResult(
        clip_keep_mask=clip_mask,
        binned_t=binned_t,
        binned_flux=binned_flux,
        savgol_t=savgol_t,
        savgol_flux=savgol_flux,
        segment_starts=segment_starts,
    )
