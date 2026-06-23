import os

import pytest

from review.event_index import EventIndex, epoch_file_exists
from review.mount import DEFAULT_MOUNT_ROOT

MOUNT = os.environ.get("SYNDIFF_VIEWER_MOUNT", DEFAULT_MOUNT_ROOT)
EVENT = "s0023_c1_k3_2020ftl"

pytestmark = pytest.mark.skipif(
    not os.path.isdir(f"{MOUNT}/events/{EVENT}"),
    reason="Workspace not available on NFS",
)


def test_load_real_lightcurve():
    idx = EventIndex.load(f"{MOUNT}/events/{EVENT}", lc_name="primary")
    assert len(idx.epochs) > 1000
    assert idx.epochs["diff_path"].notna().sum() > 0


def test_master_diff_exists():
    idx = EventIndex.load(f"{MOUNT}/events/{EVENT}", lc_name="primary")
    ok = sum(epoch_file_exists(row)["diff_exists"] for _, row in idx.epochs.iterrows())
    assert ok > 100
