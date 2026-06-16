# Light-curve review tool

Laptop-only Dash app for SynDiff pipeline outputs. Reads event data from the STScI NFS workspace and opens FITS in local DS9 via XPA.

## Prerequisites

- `mamba activate syndiff`
- Editable install of the pipeline is **not** required; minimal FFI/manifest helpers are vendored in `review/support/`.
- On the STScI network, workspace data at `/System/Volumes/Data/astro/armin/koji/syndiff/workspace`
- SAOImageDS9 installed (macOS: `open -a SAOImageDS9`). XPA tools (`xpaset`, `xpaget`) ship inside the app bundle.

## Install and run

```bash
cd syndiff_viewer
pip install -e ".[dev]"
syndiff-review --event s0023_c1_k3_2020ftl
```

Open http://127.0.0.1:8050

Copy `config/review_config.example.yaml` to `~/.config/syndiff/review.yaml` to persist settings.

## DS9 integration

The review tool uses a **single DS9 instance** on macOS:

1. **Detect running DS9** via XPA (`xpaget ds9 version`), probing transports `inet`, `local`, then `unix`.
2. **Launch** with `open -a SAOImageDS9` when needed.
3. **Load FITS** via XPA: `xpaset -p ds9 frame new`, `xpaset -p ds9 fits <path>`, then scale and `region load` as needed.
4. Commands are **queued** so rapid clicks load sequentially. Status shows the workspace-relative path, e.g. `Queued kernel_fit/ffi.fits`.

### Sidebar buttons

**Selected FFI** (per epoch): Diff, FFI, Template, Conv Template, Background, Mask.

**Kernel Determination** (when `kernel_fit/` exists): kernel reference (`kernel_fit/ffi.fits`), kernel template (`kernel_fit/template.fits`), static hp1/hp2 diff and bkg under `kernel_fit/`, `phot_bkg_fine_on_hp1_diff.fits`, `sci1_clean.fits`, mask. hp2 bkg is the 0th-order background.

### DS9 test scripts

```bash
python scripts/test_ds9_xpa.py          # which XPA_METHOD reaches DS9
python scripts/test_ds9_load.py /path/to/image.fits --wait 10
```

## Metadata cache

On startup (and when you click Reload), the app copies non-FITS workspace files from the NFS source into `.cache/workspace/` under the project root:

- Per-event manifest (`syndiff_ffi_frames.csv`) and `cluster_template_job.json`
- Per-workspace `diff_config.yaml`, `targets.reg`, light-curve CSVs
- `convolved_templates.csv` manifests when present under a workspace

Files whose modification time already matches the cache are skipped. FITS images are read from NFS for DS9; cropped FFI/template previews are written under `.cache/crops/`.

Use `--no-sync` to skip syncing, or `--cache-dir` to override the cache location.

## FITS cropping for DS9

Diff images and pipeline products are already cropped to the diff ROI. Full-chip FFIs and syndiff templates are cropped on demand when you click **Open FFI** or **Open Template**:

1. Crop bounds are parsed from `targets.reg` in the selected workspace.
2. A cropped FITS is written to `.cache/crops/{event}/{workspace}/`.
3. DS9 loads the cached crop (reused until the source FITS mtime changes).

If `targets.reg` has no ROI comment, DS9 opens the full frame and the status line notes the fallback.

Template paths are resolved from `{workspace}/templates` (symlink) using manifest `group_dx`/`group_dy` offsets. Convolved templates are looked up via `convolved_templates.csv` in the conv-template output directory.

## Tests

```bash
pytest tests/test_review_smoothing.py tests/test_review_event_index.py tests/test_review_ds9.py tests/test_review_pipeline_labels.py -v
pytest tests/test_review_integration.py -v   # requires NFS workspace
```

## Layout

```
syndiff_viewer/
  review/          # Python package
  tests/
  config/
  docs/
```

Pipeline code stays in `../syndiff_pipeline` (separate repo); this project vendors only the small path/naming helpers it needs.
