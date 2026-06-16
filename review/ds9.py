"""DS9 controller via XPA (local Unix sockets) with a background command queue."""

from __future__ import annotations

import logging
import queue
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

from review.paths_resolve import resolve_fits_path

log = logging.getLogger(__name__)

Ds9OpenMode = Literal["xpa", "open", "cli"]
DS9_OPEN_MODE_LABELS: dict[Ds9OpenMode, str] = {
    "xpa": "XPA",
    "open": "macOS open",
    "cli": "ds9 CLI",
}

DEFAULT_DS9_APP = "SAOImageDS9"
DEFAULT_XPA_DIR = Path("/Applications/SAOImageDS9.app/Contents/MacOS")
DEFAULT_DIFF_SCALE = (-10.0, 10.0)
DEFAULT_PERCENTILE_SCALE = 90
XPA_ACCESS_POINTS = ("ds9", "DS9")
XPA_METHODS = ("inet", "local", "unix")


@dataclass(frozen=True)
class Ds9LaunchResult:
    ok: bool
    message: str
    command: list[str]


@dataclass(frozen=True)
class _LoadJob:
    fits_path: str
    display_path: str
    regions: str | None
    is_diff: bool
    label: str
    open_mode: Ds9OpenMode


def format_fits_display_path(path: str | Path) -> str:
    """Return a workspace-relative FITS path (strip ``ws`` / ``ws_*`` prefix)."""
    p = Path(path)
    parts = list(p.parts)
    for i, part in enumerate(parts):
        if part == "ws" or part.startswith("ws_"):
            rel = Path(*parts[i + 1 :])
            return str(rel) if rel.parts else p.name
    return p.name


def _resolve_xpa_bin(name: str, *, ds9_path: str, ds9_xpa_dir: str | None) -> str:
    if ds9_xpa_dir:
        candidate = Path(ds9_xpa_dir) / name
        if candidate.is_file():
            return str(candidate)
    if ds9_path not in (DEFAULT_DS9_APP, "ds9", "open"):
        parent = Path(ds9_path).parent
        candidate = parent / name
        if candidate.is_file():
            return str(candidate)
    candidate = DEFAULT_XPA_DIR / name
    if candidate.is_file():
        return str(candidate)
    found = shutil.which(name)
    if found:
        return found
    return name


def _probe_xpa_env(
    xpaget: str,
    *,
    use_process_flag: bool,
) -> tuple[dict[str, str], str] | None:
    """Return ``(env, access_point)`` when DS9 responds; else ``None``."""
    base_env = dict(__import__("os").environ)
    for method in XPA_METHODS:
        env = {**base_env, "XPA_METHOD": method}
        for name in XPA_ACCESS_POINTS:
            cmd = _build_xpa_cmd(xpaget, name, "version", use_process_flag=use_process_flag)
            try:
                result = subprocess.run(  # noqa: S603
                    cmd,
                    capture_output=True,
                    text=True,
                    env=env,
                )
            except OSError:
                continue
            if result.returncode == 0:
                return env, name
    return None


def _xpa_accepts_process_flag(xpa_bin: str) -> bool:
    """Return True when ``-p`` selects the XPA access point (SAO bundle tools)."""
    try:
        result = subprocess.run(  # noqa: S603
            [xpa_bin, "-p", "ds9", "version"],
            capture_output=True,
            text=True,
            env={**__import__("os").environ, "XPA_METHOD": "inet"},
        )
    except OSError:
        return False
    out = (result.stdout or "").lower()
    return result.returncode == 0 and "ds9" in out


def _build_xpa_cmd(
    tool: str,
    access_point: str,
    *args: str,
    use_process_flag: bool,
) -> list[str]:
    if use_process_flag:
        return [tool, "-p", access_point, *args]
    return [tool, access_point, *args]


def _build_xpa_set_cmd(tool: str, access_point: str, *args: str) -> list[str]:
    """Write commands use ``-p`` per DS9 docs; required on macOS to avoid xpaset hangs."""
    return [tool, "-p", access_point, *args]


def _normalize_open_mode(mode: str | Ds9OpenMode) -> Ds9OpenMode:
    if mode in ("xpa", "open", "cli"):
        return mode  # type: ignore[return-value]
    return "xpa"


class Ds9Controller:
    """Singleton-style DS9 launcher: one app instance, queued XPA loads."""

    def __init__(
        self,
        *,
        ds9_path: str = DEFAULT_DS9_APP,
        ds9_xpa_dir: str | None = None,
        diff_scale: tuple[float, float] = DEFAULT_DIFF_SCALE,
        percentile_scale: float = DEFAULT_PERCENTILE_SCALE,
        launch_poll_s: float = 0.5,
        launch_retries: int = 40,
        open_mode: Ds9OpenMode | str = "xpa",
    ) -> None:
        self.ds9_path = ds9_path
        self.ds9_xpa_dir = ds9_xpa_dir
        self.open_mode: Ds9OpenMode = _normalize_open_mode(open_mode)
        self.diff_scale = diff_scale
        self.percentile_scale = percentile_scale
        self.launch_poll_s = launch_poll_s
        self.launch_retries = launch_retries
        self._xpaset = _resolve_xpa_bin("xpaset", ds9_path=ds9_path, ds9_xpa_dir=ds9_xpa_dir)
        self._xpaget = _resolve_xpa_bin("xpaget", ds9_path=ds9_path, ds9_xpa_dir=ds9_xpa_dir)
        self._use_process_flag = _xpa_accepts_process_flag(self._xpaget)
        base_env = dict(__import__("os").environ)
        self._xpa_env: dict[str, str] = {**base_env, "XPA_METHOD": "inet"}
        self._access_point = XPA_ACCESS_POINTS[0]
        probe = _probe_xpa_env(self._xpaget, use_process_flag=self._use_process_flag)
        if probe:
            self._xpa_env, self._access_point = probe
        self._queue: queue.Queue[_LoadJob | None] = queue.Queue()
        self._pending = 0
        self._lock = threading.Lock()
        self._last_message = ""
        self._worker = threading.Thread(target=self._worker_loop, name="ds9-xpa-worker", daemon=True)
        self._worker.start()
        log.info(
            "DS9 XPA: xpaset=%s method=%s use_process_flag=%s access_point=%s",
            self._xpaset,
            self._xpa_env.get("XPA_METHOD"),
            self._use_process_flag,
            self._access_point,
        )

    def _refresh_xpa_target(self) -> bool:
        probe = _probe_xpa_env(self._xpaget, use_process_flag=self._use_process_flag)
        if probe is None:
            return False
        self._xpa_env, self._access_point = probe
        return True

    def is_running(self) -> bool:
        return self._refresh_xpa_target()

    def _resolve_ds9_app_name(self) -> str:
        app = self.ds9_path
        if app in ("ds9", DEFAULT_DS9_APP, "open"):
            return DEFAULT_DS9_APP
        return app

    def _resolve_ds9_cli_exe(self) -> str:
        exe = self.ds9_path
        if exe in (DEFAULT_DS9_APP, "open"):
            exe = shutil.which("ds9") or str(DEFAULT_XPA_DIR / "ds9")
        elif exe == "ds9":
            exe = shutil.which("ds9") or str(DEFAULT_XPA_DIR / "ds9")
        return exe

    def _launch_ds9_app(self) -> None:
        if sys.platform == "darwin":
            app = self.ds9_path
            if app in ("ds9", DEFAULT_DS9_APP, "open"):
                app = DEFAULT_DS9_APP
            subprocess.Popen(  # noqa: S603
                ["open", "-a", app],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        else:
            exe = self.ds9_path
            if exe in (DEFAULT_DS9_APP, "open"):
                exe = shutil.which("ds9") or str(DEFAULT_XPA_DIR / "ds9")
            subprocess.Popen(  # noqa: S603
                [exe],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )

    def ensure_running(self) -> None:
        if self.is_running():
            return
        self._launch_ds9_app()
        for _ in range(self.launch_retries):
            time.sleep(self.launch_poll_s)
            if self.is_running():
                return
        raise RuntimeError(
            f"DS9 did not become available via XPA after launch "
            f"(xpaget={self._xpaget!r}, tried methods={XPA_METHODS})"
        )

    def enqueue_load(
        self,
        path: str | Path | None,
        *,
        regions: str | Path | None = None,
        is_diff: bool = False,
        label: str = "image",
    ) -> Ds9LaunchResult:
        resolved = resolve_fits_path(path)
        if resolved is None:
            return Ds9LaunchResult(False, f"No valid FITS path for {label}", [])

        reg_path: str | None = None
        if regions:
            reg = resolve_fits_path(regions) or Path(regions).expanduser()
            if reg.is_file():
                reg_path = str(reg)

        display = format_fits_display_path(resolved)
        with self._lock:
            self._pending += 1
            pending = self._pending

        mode = self.open_mode
        self._queue.put(
            _LoadJob(
                fits_path=str(resolved),
                display_path=display,
                regions=reg_path,
                is_diff=is_diff,
                label=label,
                open_mode=mode,
            )
        )
        msg = f"Queued {display}"
        if mode != "xpa":
            msg += f" ({DS9_OPEN_MODE_LABELS[mode]})"
        if pending > 1:
            msg += f" ({pending} pending)"
        return Ds9LaunchResult(True, msg, ["enqueue", label, str(resolved)])

    def _worker_loop(self) -> None:
        while True:
            job = self._queue.get()
            try:
                if job is None:
                    return
                self._execute_load(job)
            except Exception as exc:
                log.exception("DS9 load failed for %s (%s)", job.label if job else "?", job.display_path if job else "")
                with self._lock:
                    disp = job.display_path if job else ""
                    self._last_message = f"Failed {disp}: {exc}" if disp else str(exc)
            finally:
                if job is not None:
                    with self._lock:
                        self._pending = max(0, self._pending - 1)
                self._queue.task_done()

    def _xpa_set(self, *args: str, timeout: float = 30.0) -> subprocess.CompletedProcess[str]:
        cmd = _build_xpa_set_cmd(self._xpaset, self._access_point, *args)
        return self._xpa_run(cmd, timeout=timeout)

    def _load_via_xpa(self, job: _LoadJob) -> None:
        steps: list[tuple[list[str], str, float]] = [
            (["frame", "new"], "frame new", 30.0),
            (["fits", job.fits_path], f"fits {job.display_path}", 180.0),
        ]
        if job.is_diff:
            lo, hi = self.diff_scale
            steps.append((["scale", "limits", str(lo), str(hi)], f"scale limits {lo} {hi}", 30.0))
        else:
            steps.append(
                (["scale", "mode", str(self.percentile_scale)], f"scale mode {self.percentile_scale}", 30.0)
            )
        if job.regions:
            steps.append((["region", "load", job.regions], "region load", 30.0))

        for xpa_args, desc, timeout in steps:
            result = self._xpa_set(*xpa_args, timeout=timeout)
            if result.returncode != 0:
                err = (result.stderr or result.stdout or "").strip()
                raise RuntimeError(f"xpaset {desc} failed: {err or 'unknown error'}")

    def _load_via_open(self, job: _LoadJob) -> list[str]:
        if sys.platform != "darwin":
            raise RuntimeError("macOS open mode requires darwin")
        app = self._resolve_ds9_app_name()
        cmd = ["open", "-a", app, job.fits_path]
        subprocess.Popen(  # noqa: S603
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return cmd

    def _load_via_cli(self, job: _LoadJob) -> list[str]:
        exe = self._resolve_ds9_cli_exe()
        cmd: list[str] = [exe]
        if job.is_diff:
            lo, hi = self.diff_scale
            cmd.extend(["-scale", "limits", str(lo), str(hi)])
        else:
            cmd.extend(["-scale", "mode", str(self.percentile_scale)])
        if job.regions:
            cmd.extend(["-regions", "load", job.regions])
        cmd.append(job.fits_path)
        subprocess.Popen(  # noqa: S603
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return cmd

    def _execute_load(self, job: _LoadJob) -> None:
        if job.open_mode == "open":
            self._load_via_open(job)
        elif job.open_mode == "cli":
            self._load_via_cli(job)
        else:
            if not self.is_running():
                self.ensure_running()
            self._load_via_xpa(job)
        with self._lock:
            self._last_message = f"Loaded {job.display_path}"

    def _xpa_run(self, cmd: list[str], *, timeout: float = 30.0) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(  # noqa: S603
                cmd,
                capture_output=True,
                text=True,
                env=self._xpa_env,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"XPA command timed out after {timeout}s: {' '.join(cmd)}"
            ) from exc


def launch_ds9(
    paths: Sequence[str | Path],
    *,
    ds9_path: str = DEFAULT_DS9_APP,
    regions: str | Path | None = None,
    is_diff: bool = False,
    label: str = "image",
    controller: Ds9Controller | None = None,
    percentile_scale: float = DEFAULT_PERCENTILE_SCALE,
    diff_scale: tuple[float, float] = DEFAULT_DIFF_SCALE,
    ds9_xpa_dir: str | None = None,
) -> Ds9LaunchResult:
    """Enqueue a FITS load (first valid path). Kept for backward-compatible call sites."""
    if not paths:
        return Ds9LaunchResult(False, "No valid FITS paths to open", [])
    ctrl = controller or Ds9Controller(
        ds9_path=ds9_path,
        ds9_xpa_dir=ds9_xpa_dir,
        diff_scale=diff_scale,
        percentile_scale=percentile_scale,
    )
    return ctrl.enqueue_load(paths[0], regions=regions, is_diff=is_diff, label=label)


def open_diff(
    diff_path: str | Path,
    *,
    ds9_path: str = DEFAULT_DS9_APP,
    regions: str | Path | None = None,
    controller: Ds9Controller | None = None,
    diff_scale: tuple[float, float] = DEFAULT_DIFF_SCALE,
    ds9_xpa_dir: str | None = None,
) -> Ds9LaunchResult:
    return launch_ds9(
        [diff_path],
        ds9_path=ds9_path,
        regions=regions,
        is_diff=True,
        controller=controller,
        diff_scale=diff_scale,
        ds9_xpa_dir=ds9_xpa_dir,
        label="diff",
    )
