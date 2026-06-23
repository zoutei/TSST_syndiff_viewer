from pathlib import Path

import yaml

from review.mount import (
    is_photometry_dir,
    is_workspace_dir,
    list_events,
    list_photometry_dirs,
    list_workspaces,
    resolve_mount_root,
)


def _write_event_with_workspaces(tmp: Path, *, new_style: bool = False) -> Path:
    event = tmp / "s0040_test"
    for ws_name in ("ws", "ws_single_hp_kernel"):
        ws = event / ws_name
        lc = ws / "lc_prf_on_diffs"
        master = ws / "master"
        lc.mkdir(parents=True)
        master.mkdir(parents=True)
        phot_stage: dict = {"kind": "forced_photometry", "output": "lc_prf_on_diffs"}
        if new_style:
            phot_stage["methods"] = [{"name": "prf", "type": "psf", "psf_type": "prf"}]
        (ws / "diff_config.yaml").write_text(
            yaml.dump({"pipeline": [phot_stage]})
        )
        if new_style:
            (lc / "lightcurve_prf.csv").write_text("btjd,flux,eflux\n1,1,0.1\n")
        else:
            (lc / "lightcurve.csv").write_text("btjd,flux,eflux\n1,1,0.1\n")
    return event


def test_list_workspaces(tmp_path):
    event = _write_event_with_workspaces(tmp_path)
    assert list_workspaces(event) == ["ws", "ws_single_hp_kernel"]


def test_list_photometry_dirs_legacy(tmp_path):
    event = _write_event_with_workspaces(tmp_path, new_style=False)
    assert list_photometry_dirs(event / "ws") == ["lc_prf_on_diffs"]


def test_list_photometry_dirs_new_style(tmp_path):
    event = _write_event_with_workspaces(tmp_path, new_style=True)
    assert list_photometry_dirs(event / "ws") == ["lc_prf_on_diffs"]


def test_resolve_mount_root_prefers_working_preferred(tmp_path):
    mount = tmp_path / "mount"
    (mount / "events").mkdir(parents=True)
    assert resolve_mount_root(mount, allow_fallback=False) == mount.resolve()


def test_resolve_mount_root_strict_keeps_broken_preferred(tmp_path):
    broken = tmp_path / "broken"
    broken.mkdir()
    assert resolve_mount_root(broken, allow_fallback=False) == broken.resolve()


def test_resolve_mount_root_tries_fallbacks(tmp_path, monkeypatch):
    good = tmp_path / "nfs"
    (good / "events").mkdir(parents=True)
    broken = tmp_path / "broken_mount"
    broken.mkdir()
    monkeypatch.setattr("review.mount.DEFAULT_MOUNT_ROOT", str(tmp_path / "missing_default"))
    monkeypatch.setattr("review.mount.FALLBACK_MOUNT_ROOTS", (str(good),))
    assert resolve_mount_root(broken) == good.resolve()


def test_list_events(tmp_path):
    events_root = tmp_path / "events"
    _write_event_with_workspaces(events_root)
    assert list_events(tmp_path) == ["s0040_test"]


def test_workspace_and_photometry_helpers(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "diff_config.yaml").write_text("pipeline: []\n")
    assert is_workspace_dir(ws)
    assert not is_workspace_dir(tmp_path / "ws_dbg")

    lc = tmp_path / "lc_prf_on_diffs"
    lc.mkdir()
    (lc / "lightcurve_prf.csv").write_text("btjd\n")
    assert is_photometry_dir(lc)
    assert not is_photometry_dir(tmp_path / "hp_d")

    legacy = tmp_path / "lc_legacy"
    legacy.mkdir()
    (legacy / "lightcurve.csv").write_text("btjd\n")
    assert is_photometry_dir(legacy)
