"""CLI for the SynDiff light-curve review tool."""

from __future__ import annotations

import argparse
import logging

from review.app import run_app
from review.config import DEFAULT_CONFIG_PATH, ReviewConfig


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SynDiff light-curve review (Dash + DS9)")
    p.add_argument("--config", default=None, help=f"YAML config (default: {DEFAULT_CONFIG_PATH})")
    p.add_argument("--mount", dest="mount_root", default=None, help="Workspace mount root")
    p.add_argument("--cache-dir", dest="cache_root", default=None, help="Local metadata cache directory")
    p.add_argument("--no-sync", dest="sync_on_start", action="store_false", help="Skip metadata sync on startup")
    p.add_argument("--event", dest="default_event", default=None, help="Default event label")
    p.add_argument("--workspace", dest="default_workspace", default=None, help="Default workspace subdir")
    p.add_argument("--host", default=None)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--ds9-path", dest="ds9_path", default=None)
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    cfg = ReviewConfig.load(args.config)
    if args.mount_root:
        cfg.mount_root = args.mount_root
        cfg.mount_root_strict = True
    if args.cache_root:
        cfg.cache_root = args.cache_root
    if not args.sync_on_start:
        cfg.sync_on_start = False
    if args.default_event:
        cfg.default_event = args.default_event
    if args.default_workspace:
        cfg.default_workspace = args.default_workspace
    if args.host:
        cfg.host = args.host
    if args.port is not None:
        cfg.port = args.port
    if args.ds9_path:
        cfg.ds9_path = args.ds9_path

    run_app(cfg)


if __name__ == "__main__":
    main()
