"""Resolve and load TESSreduce comparison light curves."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# BTJD = JD - 2457000; MJD = JD - 2400000.5  =>  BTJD = MJD - 56999.5
MJD_TO_BTJD_OFFSET = 56999.5

_EVENT_LABEL_RE = re.compile(
    r"^s(?P<sector>\d+)_c\d+_k\d+_(?P<sn>.+)$",
    re.IGNORECASE,
)
_TESSREDUCE_FILENAME_RE = re.compile(
    r"^\d+_SN(?P<sn>.+)_(?P<sector>s\d+)_tessreduce\.csv$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TessreduceLightCurve:
    path: Path | None
    data: pd.DataFrame


_tessreduce_cache: dict[tuple[str, str], TessreduceLightCurve] = {}


def clear_tessreduce_cache() -> None:
    _tessreduce_cache.clear()


def parse_event_label(event_label: str) -> tuple[int, str] | None:
    """Return ``(sector, sn_name)`` from ``s0023_c1_k3_2020ftl``."""
    match = _EVENT_LABEL_RE.match(str(event_label).strip())
    if not match:
        return None
    return int(match.group("sector")), match.group("sn").lower()


def tessreduce_csv_path(event_label: str, tessreduce_root: str | Path) -> Path | None:
    """Find ``*_SN{sn}_s{sector}_tessreduce.csv`` for a SynDiff event label."""
    parsed = parse_event_label(event_label)
    if parsed is None:
        return None
    sector, sn = parsed
    root = Path(tessreduce_root)
    if not root.is_dir():
        return None

    pattern = f"*SN{sn}_s{sector}_tessreduce.csv"
    hits = sorted(root.glob(pattern))
    if len(hits) == 1:
        return hits[0]
    if hits:
        return hits[0]

    needle = f"SN{sn}_s{sector}_tessreduce.csv".lower()
    for path in root.glob("*_tessreduce.csv"):
        if path.name.lower().endswith(needle):
            return path
    return None


def _read_tessreduce_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "time" not in df.columns:
        raise ValueError(f"TESSreduce CSV missing time column: {path}")
    if "flux" not in df.columns:
        raise ValueError(f"TESSreduce CSV missing flux column: {path}")

    out = pd.DataFrame(
        {
            "btjd": pd.to_numeric(df["time"], errors="coerce") - MJD_TO_BTJD_OFFSET,
            "flux": pd.to_numeric(df["flux"], errors="coerce"),
        }
    )
    if "flux_err" in df.columns:
        out["eflux"] = pd.to_numeric(df["flux_err"], errors="coerce")
    else:
        out["eflux"] = np.nan

    ok = out["btjd"].notna() & np.isfinite(out["btjd"]) & out["flux"].notna()
    return out.loc[ok].reset_index(drop=True)


def load_tessreduce_for_event(event_label: str, tessreduce_root: str | Path) -> TessreduceLightCurve:
    key = (str(event_label), str(Path(tessreduce_root).expanduser().resolve()))
    if key in _tessreduce_cache:
        return _tessreduce_cache[key]
    path = tessreduce_csv_path(event_label, tessreduce_root)
    if path is None:
        lc = TessreduceLightCurve(path=None, data=pd.DataFrame(columns=["btjd", "flux", "eflux"]))
    else:
        try:
            data = _read_tessreduce_csv(path)
            lc = TessreduceLightCurve(path=path, data=data)
        except (OSError, ValueError, pd.errors.ParserError):
            lc = TessreduceLightCurve(path=path, data=pd.DataFrame(columns=["btjd", "flux", "eflux"]))
    _tessreduce_cache[key] = lc
    return lc


def tessreduce_store_payload(lc: TessreduceLightCurve) -> dict[str, object]:
    """Compact JSON-serializable payload for the Dash store."""
    if lc.path is None:
        return {"path": None, "available": False, "btjd": [], "flux": [], "eflux": []}
    df = lc.data
    return {
        "path": str(lc.path),
        "available": not df.empty,
        "btjd": df["btjd"].astype(float).tolist(),
        "flux": df["flux"].astype(float).tolist(),
        "eflux": df["eflux"].astype(float).tolist() if "eflux" in df.columns else [],
    }
