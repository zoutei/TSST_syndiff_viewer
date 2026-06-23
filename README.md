# syndiff-viewer

Laptop-only Plotly Dash app for reviewing SynDiff light curves and opening FITS in DS9.

## Setup

```bash
mamba activate syndiff
pip install -e ".[dev]"
syndiff-viewer --event s0023_c1_k3_2020ftl
```

Workspace data is read from STScI NFS at `/System/Volumes/Data/astro/armin/koji/syndiff/workspace` (on the STScI network). See [docs/lc_review.md](docs/lc_review.md) for details.
