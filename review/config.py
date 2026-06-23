"""Review tool configuration."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

from review.mount import resolve_mount_root
from review.sync_cache import resolve_cache_root

DEFAULT_CONFIG_PATH = Path.home() / ".config" / "syndiff" / "review.yaml"


@dataclass
class ReviewConfig:
    mount_root: str = "/System/Volumes/Data/astro/armin/koji/syndiff/workspace"
    tessreduce_root: str = "/System/Volumes/Data/astro/armin/koji/tessreduce_data"
    cache_root: str | None = None
    sync_on_start: bool = True
    ds9_path: str = "SAOImageDS9"
    ds9_xpa_dir: str | None = None
    ds9_open_mode: str = "xpa"
    ds9_diff_scale_min: float = -10.0
    ds9_diff_scale_max: float = 10.0
    ds9_percentile_scale: float = 90.0
    host: str = "127.0.0.1"
    port: int = 8050
    default_event: str = "s0023_c1_k3_2020ftl"
    default_workspace: str = "ws"
    default_lc: str = "prf_primary"
    gap_threshold_days: float = 1.0
    gap_auto: bool = True
    bin_width_hours: float = 6.0
    bin_sigma: float = 3.0
    savgol_window: int = 11
    savgol_polyorder: int = 2
    mount_root_strict: bool = False

    @property
    def ds9_diff_scale(self) -> tuple[float, float]:
        return (self.ds9_diff_scale_min, self.ds9_diff_scale_max)

    @classmethod
    def load(cls, path: str | Path | None = None) -> ReviewConfig:
        cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
        data: dict[str, Any] = {}
        if cfg_path.is_file():
            with cfg_path.open(encoding="utf-8") as fh:
                loaded = yaml.safe_load(fh) or {}
                if isinstance(loaded, dict):
                    data = loaded
        allowed = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in allowed}
        return cls(**kwargs)

    @property
    def mount_root_expanded(self) -> Path:
        return resolve_mount_root(self.mount_root, allow_fallback=not self.mount_root_strict)

    @property
    def source_mount_expanded(self) -> Path:
        return self.mount_root_expanded

    @property
    def cache_root_expanded(self) -> Path:
        return resolve_cache_root(self.cache_root)

    @property
    def data_mount_expanded(self) -> Path:
        return self.cache_root_expanded

    @property
    def tessreduce_root_expanded(self) -> Path:
        return Path(self.tessreduce_root).expanduser()

    def events_root(self) -> Path:
        return self.data_mount_expanded / "events"

    def event_dir(self, target_label: str) -> Path:
        return self.events_root() / target_label

    def source_event_dir(self, target_label: str) -> Path:
        return self.source_mount_expanded / "events" / target_label
