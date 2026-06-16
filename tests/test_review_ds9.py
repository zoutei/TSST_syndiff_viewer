import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from review.ds9 import (
    Ds9Controller,
    format_fits_display_path,
    launch_ds9,
    _build_xpa_cmd,
    _xpa_accepts_process_flag,
)


def _fits_stub(tmp_path: Path) -> Path:
    p = tmp_path / "test.fits"
    p.write_bytes(b"SIMPLE  =                    T / stub")
    return p


def test_format_fits_display_path_strips_workspace():
    p = "/data/events/foo/ws_single_hp_kernel/kernel_fit/ffi.fits"
    assert format_fits_display_path(p) == "kernel_fit/ffi.fits"


def test_build_xpa_cmd_without_process_flag():
    assert _build_xpa_cmd("xpaset", "ds9", "version", use_process_flag=False) == [
        "xpaset",
        "ds9",
        "version",
    ]


def test_build_xpa_cmd_with_process_flag():
    assert _build_xpa_cmd("xpaset", "ds9", "version", use_process_flag=True) == [
        "xpaset",
        "-p",
        "ds9",
        "version",
    ]


def test_xpa_accepts_process_flag_detects_illegal_option():
    with patch("review.ds9.subprocess.run") as run:
        run.return_value = MagicMock(stderr="illegal option -- p", returncode=1, stdout="")
        assert _xpa_accepts_process_flag("/usr/local/bin/xpaget") is False


def test_xpa_accepts_process_flag_when_version_returned_despite_stderr():
    with patch("review.ds9.subprocess.run") as run:
        run.return_value = MagicMock(stderr="illegal option -- p", returncode=0, stdout="ds9 8.7")
        assert _xpa_accepts_process_flag("/usr/local/bin/xpaget") is True


def test_is_running_checks_xpaget():
    ctrl = Ds9Controller()
    with patch.object(ctrl, "_refresh_xpa_target", return_value=True) as refresh:
        assert ctrl.is_running() is True
    refresh.assert_called_once()


def test_cold_start_opens_ds9_when_not_running():
    ctrl = Ds9Controller()

    with patch.object(sys, "platform", "darwin"):
        with patch("review.ds9.subprocess.Popen") as popen:
            with patch.object(ctrl, "_refresh_xpa_target", side_effect=[False, False, True]):
                ctrl.ensure_running()

    popen.assert_called_once()
    assert popen.call_args[0][0][:3] == ["open", "-a", "SAOImageDS9"]


def test_diff_uses_scale_limits(tmp_path):
    fits = _fits_stub(tmp_path)
    ctrl = Ds9Controller(diff_scale=(-10.0, 10.0))
    ctrl._use_process_flag = False
    captured: list[list[str]] = []

    def fake_run(*args, **_kwargs):
        captured.append(list(args))
        return MagicMock(returncode=0, stdout="", stderr="")

    with patch.object(ctrl, "_xpa_set", side_effect=fake_run):
        ctrl._load_via_xpa(
            __import__("review.ds9", fromlist=["_LoadJob"])._LoadJob(
                fits_path=str(fits),
                display_path="test.fits",
                regions=None,
                is_diff=True,
                label="diff",
            )
        )

    scale_cmd = next(a for a in captured if "scale" in a)
    assert scale_cmd[-4:] == ["scale", "limits", "-10.0", "10.0"]


def test_nondiff_uses_percentile_scale(tmp_path):
    fits = _fits_stub(tmp_path)
    ctrl = Ds9Controller(percentile_scale=90)
    ctrl._use_process_flag = False
    captured: list[list[str]] = []

    def fake_run(*args, **_kwargs):
        captured.append(list(args))
        return MagicMock(returncode=0, stdout="", stderr="")

    with patch.object(ctrl, "_xpa_set", side_effect=fake_run):
        ctrl._load_via_xpa(
            __import__("review.ds9", fromlist=["_LoadJob"])._LoadJob(
                fits_path=str(fits),
                display_path="test.fits",
                regions=None,
                is_diff=False,
                label="sci",
            )
        )

    scale_cmd = next(a for a in captured if "scale" in a)
    assert scale_cmd[-3:] == ["scale", "mode", "90"]


def test_load_via_xpa_uses_fits_not_file(tmp_path):
    fits = _fits_stub(tmp_path)
    ctrl = Ds9Controller()
    captured: list[list[str]] = []

    def fake_run(*args, **_kwargs):
        captured.append(list(args))
        return MagicMock(returncode=0, stdout="", stderr="")

    with patch.object(ctrl, "_xpa_set", side_effect=fake_run):
        ctrl._load_via_xpa(
            __import__("review.ds9", fromlist=["_LoadJob"])._LoadJob(
                fits_path=str(fits),
                display_path="test.fits",
                regions="/tmp/mask.reg",
                is_diff=False,
                label="sci",
            )
        )

    assert captured[0] == ["frame", "new"]
    assert captured[1][:2] == ["fits", str(fits)]
    assert captured[-1] == ["region", "load", "/tmp/mask.reg"]


def test_execute_load_uses_xpa_on_darwin(tmp_path):
    fits = _fits_stub(tmp_path)
    ctrl = Ds9Controller()

    with patch.object(sys, "platform", "darwin"):
        with patch.object(ctrl, "is_running", return_value=True):
            with patch.object(ctrl, "_load_via_xpa") as load:
                ctrl._execute_load(
                    __import__("review.ds9", fromlist=["_LoadJob"])._LoadJob(
                        fits_path=str(fits),
                        display_path="test.fits",
                        regions=None,
                        is_diff=False,
                        label="sci",
                    )
                )
    load.assert_called_once()


def test_queue_processes_two_jobs_in_order(tmp_path):
    fits_a = _fits_stub(tmp_path)
    fits_b = tmp_path / "b.fits"
    fits_b.write_bytes(b"SIMPLE  =                    T / stub b")
    ctrl = Ds9Controller()
    order: list[str] = []

    def fake_execute(job):
        order.append(job.label)

    with patch.object(ctrl, "_execute_load", side_effect=fake_execute):
        ctrl.enqueue_load(fits_a, label="first")
        ctrl.enqueue_load(fits_b, label="second")
        ctrl._queue.join()

    assert order == ["first", "second"]


def test_enqueue_message_includes_display_path(tmp_path):
    fits = tmp_path / "ws_single_hp_kernel" / "kernel_fit" / "ffi.fits"
    fits.parent.mkdir(parents=True)
    fits.write_bytes(b"SIMPLE  =                    T / stub")
    ctrl = Ds9Controller()
    with patch.object(ctrl, "_execute_load"):
        res = ctrl.enqueue_load(fits, label="ref")
    assert res.ok
    assert "Queued kernel_fit/ffi.fits" in res.message


def test_launch_ds9_rejects_missing_path():
    res = launch_ds9([Path("/nonexistent/file.fits")])
    assert not res.ok
