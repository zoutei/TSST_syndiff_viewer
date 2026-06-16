import numpy as np
import pytest

from review.smoothing import apply_smoothing, binned_sigma_clip_btjd, split_segments


def test_six_hours_is_quarter_day():
    btjd = np.array([100.0, 100.1, 100.2])
    flux = np.array([1.0, 2.0, 3.0])
    _, bt, _ = binned_sigma_clip_btjd(btjd, flux, bin_width_days=6.0 / 24.0, sigma=3.0)
    assert bt.size >= 1


def test_two_segment_bins_do_not_span_gap():
    btjd = np.array([0.0, 0.1, 0.2, 5.0, 5.1, 5.2])
    flux = np.array([1.0, 1.1, 1.0, 2.0, 2.1, 2.0])
    segs = split_segments(btjd, gap_threshold_days=1.0, gap_auto=False)
    assert len(segs) == 2
    result = apply_smoothing(
        btjd, flux, mode="binned", bin_width_hours=6, gap_threshold_days=1.0, gap_auto=False
    )
    assert result.binned_t.size >= 2


def test_outlier_rejected_in_bin():
    btjd = np.array([0.0, 0.05, 0.1, 0.15])
    flux = np.array([1.0, 1.0, 100.0, 1.0])
    mask, _, _ = binned_sigma_clip_btjd(btjd, flux, bin_width_days=1.0, sigma=3.0)
    assert not mask[2]


def test_savgol_does_not_cross_gap():
    btjd = np.array([0.0, 0.1, 0.2, 5.0, 5.1, 5.2])
    flux = np.array([1.0, 1.2, 1.1, 3.0, 3.2, 3.1])
    result = apply_smoothing(
        btjd, flux, mode="savgol", gap_threshold_days=1.0, gap_auto=False, savgol_window=3
    )
    assert len(result.segment_starts) == 2
