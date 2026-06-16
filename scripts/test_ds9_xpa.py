#!/usr/bin/env python3
"""Diagnose DS9 XPA connectivity (run with DS9 open or closed)."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

XPA_METHODS = ("inet", "local", "unix")
ACCESS_POINTS = ("ds9", "DS9")


def _run(cmd: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    print(f"$ XPA_METHOD={env.get('XPA_METHOD', '')} {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)  # noqa: S603
    out = (result.stdout or "").strip()
    err = (result.stderr or "").strip()
    if out:
        print(out)
    if err:
        print(err, file=sys.stderr)
    print(f"exit code: {result.returncode}\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    xpaget = shutil.which("xpaget") or "xpaget"
    xpaset = shutil.which("xpaset") or "xpaset"
    print(f"xpaget: {xpaget}")
    print(f"xpaset: {xpaset}\n")

    dash_p = _run([xpaget, "-p", "ds9", "version"], {**os.environ, "XPA_METHOD": "inet"})
    uses_dash_p = dash_p.returncode == 0 and "illegal option" not in (dash_p.stderr or "").lower()
    print(f"-p process flag supported: {uses_dash_p}\n")

    for method in XPA_METHODS:
        env = {**os.environ, "XPA_METHOD": method}
        for name in ACCESS_POINTS:
            if uses_dash_p:
                _run([xpaget, "-p", name, "version"], env)
            else:
                _run([xpaget, name, "version"], env)

    print("If DS9 is running, look for exit code 0 and a version string above.")
    print("The review app probes inet/local/unix and uses the first method that works.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
