from pathlib import Path

import pandas as pd
import pytest

from review.tessreduce import (
    MJD_TO_BTJD_OFFSET,
    TessreduceLightCurve,
    clear_tessreduce_cache,
    load_tessreduce_for_event,
    parse_event_label,
    tessreduce_csv_path,
    tessreduce_store_payload,
)


def test_parse_event_label():
    assert parse_event_label("s0023_c1_k3_2020ftl") == (23, "2020ftl")
    assert parse_event_label("s0041_c1_k2_2021udg") == (41, "2021udg")
    assert parse_event_label("not_an_event") is None


def test_tessreduce_csv_path(tmp_path: Path):
    (tmp_path / "0004_SN2020ftl_s23_tessreduce.csv").write_text("time,flux,flux_err\n")
    (tmp_path / "0010_SN2021udg_s41_tessreduce.csv").write_text("time,flux,flux_err\n")

    hit = tessreduce_csv_path("s0023_c1_k3_2020ftl", tmp_path)
    assert hit is not None
    assert hit.name == "0004_SN2020ftl_s23_tessreduce.csv"

    hit_udg = tessreduce_csv_path("s0041_c1_k2_2021udg", tmp_path)
    assert hit_udg is not None
    assert hit_udg.name == "0010_SN2021udg_s41_tessreduce.csv"

    assert tessreduce_csv_path("s0099_c1_k1_missing", tmp_path) is None


def test_read_tessreduce_converts_mjd_to_btjd(tmp_path: Path):
    csv = tmp_path / "0001_SN2020ut_s20_tessreduce.csv"
    csv.write_text(
        "time,flux,flux_err\n"
        "58927.620544433594,10.0,1.0\n"
        "58927.641357421875,nan,nan\n"
        "58927.662170410156,20.0,2.0\n"
    )
    lc = load_tessreduce_for_event("s0020_c3_k3_2020ut", tmp_path)
    assert lc.path == csv
    assert len(lc.data) == 2
    assert lc.data.iloc[0]["btjd"] == pytest.approx(58927.620544433594 - MJD_TO_BTJD_OFFSET)
    assert lc.data.iloc[0]["flux"] == 10.0
    assert lc.data.iloc[0]["eflux"] == 1.0


def test_tessreduce_store_payload_empty():
    payload = tessreduce_store_payload(
        TessreduceLightCurve(path=None, data=pd.DataFrame())
    )
    assert payload["available"] is False
    assert payload["btjd"] == []


def test_tessreduce_load_is_memoized(tmp_path: Path):
    clear_tessreduce_cache()
    csv = tmp_path / "0004_SN2020ftl_s23_tessreduce.csv"
    csv.write_text("time,flux,flux_err\n58927.62,10.0,1.0\n")
    first = load_tessreduce_for_event("s0023_c1_k3_2020ftl", tmp_path)
    second = load_tessreduce_for_event("s0023_c1_k3_2020ftl", tmp_path)
    assert first is second
    clear_tessreduce_cache()


@pytest.mark.integration
def test_tessreduce_nfs_lookup():
    root = Path("/System/Volumes/Data/astro/armin/koji/tessreduce_data")
    if not root.is_dir():
        pytest.skip("TESSreduce NFS data not mounted")
    path = tessreduce_csv_path("s0023_c1_k3_2020ftl", root)
    assert path is not None
    assert path.name == "0004_SN2020ftl_s23_tessreduce.csv"
    lc = load_tessreduce_for_event("s0023_c1_k3_2020ftl", root)
    assert len(lc.data) > 1000
