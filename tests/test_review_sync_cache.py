import shutil
from pathlib import Path

import pandas as pd
import yaml

from review.event_index import EventIndex, epoch_file_exists
from review.sync_cache import discover_metadata_files, needs_update, sync_workspace_metadata


def _write_event(tmp: Path, label: str = "s0023_test") -> Path:
    event = tmp / label
    ws = event / "ws" / "hp_d"
    master = event / "ws" / "master"
    lc_dir = event / "ws" / "lc_prf_on_diffs"
    ws.mkdir(parents=True)
    master.mkdir(parents=True)
    lc_dir.mkdir(parents=True)

    diff_cfg = {
        "pipeline": [
            {
                "kind": "hotpants",
                "output": {"diffs": "hp_d"},
                "write_bkg": False,
                "write_convolved": False,
            },
            {"kind": "forced_photometry", "inputs": {"diffs": "hp_d"}, "output": "lc_prf_on_diffs"},
        ],
    }
    (event / "ws" / "diff_config.yaml").write_text(yaml.dump(diff_cfg))
    (event / "ws" / "targets.reg").write_text("# Region file\n")

    manifest = pd.DataFrame(
        [
            {
                "filename": "tess2020019142923-s0023-1-3-0165-s_ffic.fits",
                "path": "/data/tess2020019142923-s0023-1-3-0165-s_ffic.fits",
                "group_id": 0,
                "hotpants_hp_d_ok": True,
            }
        ]
    )
    manifest.to_csv(event / "syndiff_ffi_frames.csv", index=False)

    lc = pd.DataFrame(
        [
            {
                "btjd": 1928.94,
                "flux": 10.0,
                "eflux": 1.0,
                "filename": str(ws / "tess2020019142923_hp_d.fits"),
                "group_id": 0,
            }
        ]
    )
    lc.to_csv(lc_dir / "lightcurve.csv", index=False)

    diff_fits = ws / "tess2020019142923_hp_d.fits"
    diff_fits.write_bytes(b"SIMPLE  =                    T / syn diff test")
    (master / "tess2020019142923_hp_d.fits").symlink_to(diff_fits.resolve())
    (master / "tess2020019142923-s0023-1-3-0165-s_ffic.fits").symlink_to(diff_fits.resolve())

    return event


def _source_layout(tmp: Path) -> Path:
    source = tmp / "source"
    events = source / "events"
    events.mkdir(parents=True)
    _write_event(events, "event_a")
    _write_event(events, "event_b")
    return source


def test_needs_update_skips_matching_mtime(tmp_path):
    src = tmp_path / "a.csv"
    dst = tmp_path / "b.csv"
    src.write_text("x\n")
    shutil.copy2(src, dst)
    assert not needs_update(src, dst)


def test_needs_update_recopies_changed_mtime(tmp_path):
    src = tmp_path / "a.csv"
    dst = tmp_path / "b.csv"
    src.write_text("v1\n")
    dst.write_text("v0\n")
    assert needs_update(src, dst)


def test_discover_metadata_files_excludes_fits(tmp_path):
    source = _source_layout(tmp_path)
    discovered = discover_metadata_files(source)
    suffixes = {p.suffix for p in discovered}
    assert ".csv" in suffixes
    assert ".yaml" in suffixes
    assert ".reg" in suffixes
    assert ".fits" not in suffixes


def test_sync_copies_metadata_not_fits(tmp_path):
    source = _source_layout(tmp_path)
    cache = tmp_path / "cache"
    result = sync_workspace_metadata(source, cache)
    assert result.copied > 0
    assert (cache / "events" / "event_a" / "syndiff_ffi_frames.csv").is_file()
    assert (cache / "events" / "event_a" / "ws" / "diff_config.yaml").is_file()
    assert not (cache / "events" / "event_a" / "ws" / "master").exists()


def test_sync_all_events(tmp_path):
    source = _source_layout(tmp_path)
    cache = tmp_path / "cache"
    result = sync_workspace_metadata(source, cache)
    assert (cache / "events" / "event_a" / "syndiff_ffi_frames.csv").is_file()
    assert (cache / "events" / "event_b" / "syndiff_ffi_frames.csv").is_file()
    assert result.copied == len(discover_metadata_files(source))


def test_sync_skips_unchanged_on_second_run(tmp_path):
    source = _source_layout(tmp_path)
    cache = tmp_path / "cache"
    first = sync_workspace_metadata(source, cache)
    second = sync_workspace_metadata(source, cache)
    assert first.copied > 0
    assert second.copied == 0
    assert second.skipped == first.copied


def test_event_index_dual_root(tmp_path):
    source = tmp_path / "source"
    cache = tmp_path / "cache"
    events = source / "events"
    events.mkdir(parents=True)
    event = _write_event(events)

    sync_workspace_metadata(source, cache)
    cache_event = cache / "events" / event.name
    assert not (cache_event / "ws" / "master").exists()

    idx = EventIndex.load(cache_event, fits_event_dir=event)
    assert len(idx.epochs) == 1
    assert epoch_file_exists(idx.epochs.iloc[0])["diff_exists"]
    assert (cache_event / "ws" / "targets.reg").is_file()
    assert "tess2020019142923_hp_d.fits" in idx.epochs.iloc[0]["diff_path"]
