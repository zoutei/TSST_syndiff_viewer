#!/usr/bin/env python3
"""Load a FITS file into DS9 through the review Ds9Controller (end-to-end test)."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow running from repo root without install
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from review.ds9 import Ds9Controller, format_fits_display_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fits", type=Path, help="FITS file to load")
    parser.add_argument("--diff", action="store_true", help="Use diff scale limits (-10, 10)")
    parser.add_argument("--regions", type=Path, default=None, help="Optional DS9 regions file")
    parser.add_argument("--wait", type=float, default=5.0, help="Seconds to wait for queue drain")
    args = parser.parse_args()

    if not args.fits.is_file():
        print(f"FITS not found: {args.fits}", file=sys.stderr)
        return 1

    ctrl = Ds9Controller()
    print(f"XPA tools: xpaset={ctrl._xpaset!r} xpaget={ctrl._xpaget!r}")
    print(f"XPA method: {ctrl._xpa_env.get('XPA_METHOD')}")
    print(f"Display path: {format_fits_display_path(args.fits)}")

    res = ctrl.enqueue_load(
        args.fits,
        regions=args.regions,
        is_diff=args.diff,
        label="test",
    )
    print(res.message)
    if not res.ok:
        return 1

    time.sleep(args.wait)
    ctrl._queue.join()
    print(f"Last status: {ctrl._last_message!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
