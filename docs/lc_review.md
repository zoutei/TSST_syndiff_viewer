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
syndiff-viewer --event s0023_c1_k3_2020ftl
```

Open http://127.0.0.1:8050

Copy `config/review_config.example.yaml` to `~/.config/syndiff/review.yaml` to persist settings.

## DS9 integration

The review tool uses a **single DS9 instance** on macOS:

1. **Detect running DS9** via XPA (`xpaget ds9 version`), probing transports `inet`, `local`, then `unix`.
2. **Launch** with `open -a SAOImageDS9` when needed.
3. **Load FITS** via XPA: `xpaset -p ds9 frame new`, `xpaset -p ds9 fits <path>`, then scale and `region load` as needed.
4. Commands are **queued** so rapid clicks load sequentially. Status shows the workspace-relative path, e.g. `Queued kernel_fit/ffi.fits`.

### Open mode (bottom bar)

Use the **DS9 open** dropdown in the bottom control bar to choose how FITS files are sent to DS9:

| Mode | Behavior |
|------|----------|
| **XPA** (default) | Full control via `xpaset`: new frame, scale limits/mode, and `targets.reg` overlay |
| **macOS open -a** | `open -a SAOImageDS9 <fits>` — opens the file in DS9 without scale or region control |
| **ds9 command line** | `ds9 -scale … [-regions load …] <fits>` — scale and regions via CLI flags |

Set the default in `review.yaml` with `ds9_open_mode: xpa` (or `open`, `cli`).

### Sidebar buttons

**Selected FFI** (per epoch): buttons are built from the workspace `diff_config.yaml` pipeline. Order is always:

1. Primary diff (`forced_photometry.inputs.diffs`)
2. FFI, Template
3. Other `kernel_subtract` / `hotpants` outputs, last stage first (labels taken from each stage's `output` map; `write_bkg` / `write_convolved` respected)
4. Conv Template (`convolved_templates.output`) when present
5. Mask

Example with both `kernel_subtract` (`ks_d`, `ks_b`) and `hotpants` (`mk_d`, `mk_b`): `mk_d`, FFI, Template, `mk_b`, `ks_d`, `ks_b`, Conv Template, Mask.

**Kernel Determination** (when `kernel_fit/` exists): kernel reference (`kernel_fit/ffi.fits`), kernel template (`kernel_fit/template.fits`), static hp1/hp2 diff and bkg under `kernel_fit/`, `phot_bkg_fine_on_hp1_diff.fits`, `sci1_clean.fits`, mask. hp2 bkg is the 0th-order background.

### DS9 test scripts

```bash
python scripts/test_ds9_xpa.py          # which XPA_METHOD reaches DS9
python scripts/test_ds9_load.py /path/to/image.fits --wait 10
```

## Metadata cache

On startup the app copies non-FITS workspace files from the NFS source into `.cache/workspace/` under the project root:

- Per-event manifest (`syndiff_ffi_frames.csv`) and `cluster_template_job.json`
- Per-workspace `diff_config.yaml`, `targets.reg`, light-curve CSVs
- `convolved_templates.csv` manifests when present under a workspace

Files whose modification time already matches the cache are skipped. FITS images are read from NFS for DS9; cropped FFI/template previews are written under `.cache/crops/`.

Use `--no-sync` to skip syncing, or `--cache-dir` to override the cache location.

### Refresh lists vs plot loading

**Refresh lists** re-scans NFS for new events and workspaces and updates the dropdown options. It shows a spinner while syncing metadata for the current event. It does **not** reload the plot.

The plot loads when you pick an event, workspace, photometry dir, or target from the dropdowns. Subsequent target changes within the same workspace reuse a cached NFS `master/` index so switching light curves stays fast.

### Photometry dropdowns

| Dropdown | Meaning |
|----------|---------|
| **Photometry** | `lc_*` output directory from `forced_photometry` |
| **Target** | Combined method and position: `prf_primary`, `ap3_primary`, `prf_offset_top`, … |

For legacy workspaces with `lightcurve.csv`, the Target dropdown uses `primary`, `offset_top`, etc. (no method prefix).

CSV files on disk follow the pipeline naming: `lightcurve_{method}.csv` / `lightcurve_{method}_{target}.csv`. Aperture method CSVs include `flux_wo_sky`; the plot uses that sky-subtracted column.

Set the default target in `review.yaml`: `default_lc: prf_primary` (or `primary` for legacy workspaces).

### Backward compatibility

The viewer detects the layout at read time:

- **New:** `forced_photometry.methods` in frozen `diff_config.yaml` → Target lists `{method}_{target}` combinations that exist on disk.
- **Legacy:** no `methods`, but `lightcurve.csv` exists → Target lists `primary`, `offset_top`, …
- **Inferred:** no `methods` and no `lightcurve.csv`, but `lightcurve_{method}.csv` files exist → methods parsed from filenames.

## Compare layers

Overlay multiple SynDiff light curves on one plot (same event, different workspace or target):

1. Use the top dropdowns to pick a curve, then click **Add to compare** to snapshot it as a compare chip.
2. Change the top dropdowns to another workspace or target; the plot updates the primary curve. Add more chips as needed.
3. Click a compare chip to make it the **active layer** (highlighted). The sidebar, DS9 buttons, binned/rejected markers, and product-ID search use the active layer's workspace and FITS paths.
4. Click any Syndiff point on the plot to activate that layer and select an epoch.
5. When a compare chip is active, adjust **Offset** in the compare strip to vertically align curves.
6. Up to five compare layers. Layers identical to the current primary are not drawn twice (but remain in the chip list).

TESSreduce remains a separate external overlay toggle.

## FITS cropping for DS9

Diff images and pipeline products are already cropped to the diff ROI. Full-chip FFIs and syndiff templates are cropped on demand when you click **Open FFI** or **Open Template**:

1. Crop bounds are parsed from `targets.reg` in the selected workspace.
2. A cropped FITS is written to `.cache/crops/{event}/{workspace}/`.
3. DS9 loads the cached crop (reused until the source FITS mtime changes).

If `targets.reg` has no ROI comment, DS9 opens the full frame and the status line notes the fallback.

Template paths are resolved from `{workspace}/templates` (symlink) using manifest `group_dx`/`group_dy` offsets. Convolved templates are looked up via `convolved_templates.csv` in the conv-template output directory.

## Tests

```bash
pytest tests/test_review_smoothing.py tests/test_review_event_index.py tests/test_review_ds9.py tests/test_review_pipeline_labels.py tests/test_review_overlay.py -v
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
