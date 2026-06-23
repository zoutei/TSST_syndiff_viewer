import textwrap
from pathlib import Path

import yaml

from review.pipeline_labels import (
    LEGACY_METHOD,
    lightcurve_csv_basename,
    lightcurve_selection_key,
    list_epoch_products,
    list_forced_targets,
    list_lightcurve_selections,
    list_photometry_methods,
    parse_diff_config,
    parse_lightcurve_selection,
    resolve_lightcurve_filename,
)


KERNEL_CONFIG = textwrap.dedent(
    """
    pipeline:
      - kind: kernel_fit
        output: kernel_fit
      - kind: convolved_templates
        inputs:
          kernel_fit: kernel_fit
        output: tmpl_conv
      - kind: kernel_subtract
        output:
          diffs: kd_d
          phot_bkg: kd_b
      - kind: forced_photometry
        inputs:
          diffs: kd_d
        output: lc_prf_on_diffs
        methods:
          - name: prf
            type: psf
            psf_type: prf
          - name: ap3
            type: aperture
            tar_ap: 3
      - kind: hotpants
        output:
          diffs: hp1_d
          bkg: hp1_b
      - kind: hotpants
        output:
          diffs: hp2_d
          bkg: hp2_b
    additional_forced_targets:
      - name: offset_top
        position_mode: offset
        dx: 0
        dy: -7
    """
)


def test_parse_kernel_subtract_labels(tmp_path):
    path = tmp_path / "diff_config.yaml"
    path.write_text(KERNEL_CONFIG)
    labels = parse_diff_config(path)
    assert labels.diff_label == "kd_d"
    assert labels.bkg_label == "hp2_b"
    assert labels.conv_template_label == "tmpl_conv"
    assert labels.kernel_fit_dir == "kernel_fit"
    assert labels.photometry_methods == ["prf", "ap3"]
    assert labels.additional_targets == ["offset_top"]
    assert len(labels.hotpants_stages) >= 2
    assert labels.hotpants_stages[0].diffs == "hp1_d"
    assert labels.hotpants_stages[1].diffs == "hp2_d"


MULTI_HP_KERNEL_SUBTRACT_CONFIG = textwrap.dedent(
    """
    pipeline:
      - kind: kernel_fit
        output: kernel_fit
      - kind: convolved_templates
        output: tmpl_conv
      - kind: kernel_subtract
        output:
          diffs: ks_d
          phot_bkg: ks_b
      - kind: hotpants
        write_convolved: false
        write_bkg: true
        output:
          diffs: mk_d
          convolved: mk_c
          bkg: mk_b
      - kind: forced_photometry
        inputs:
          diffs: mk_d
        output: lc_prf_on_mk_diffs
    """
)


def test_epoch_products_kernel_subtract_then_hotpants(tmp_path):
    path = tmp_path / "diff_config.yaml"
    path.write_text(MULTI_HP_KERNEL_SUBTRACT_CONFIG)
    labels = parse_diff_config(path)
    assert labels.diff_label == "mk_d"
    assert labels.bkg_label == "mk_b"
    keys = [product.key for product in labels.epoch_products]
    assert keys == [
        "mk_d",
        "sci",
        "template",
        "mk_b",
        "ks_d",
        "ks_b",
        "conv_template",
        "mask",
    ]


def test_epoch_products_hotpants_only(tmp_path):
    path = tmp_path / "diff_config.yaml"
    path.write_text(
        yaml.dump(
            {
                "pipeline": [
                    {
                        "kind": "hotpants",
                        "write_bkg": True,
                        "write_convolved": True,
                        "output": {"diffs": "hp_d", "convolved": "hp_c", "bkg": "hp_b"},
                    },
                    {"kind": "forced_photometry", "inputs": {"diffs": "hp_d"}, "output": "lc"},
                ]
            }
        )
    )
    labels = parse_diff_config(path)
    assert [product.key for product in labels.epoch_products] == [
        "hp_d",
        "sci",
        "template",
        "hp_c",
        "hp_b",
        "mask",
    ]


def test_list_epoch_products_matches_parse_diff_config(tmp_path):
    path = tmp_path / "diff_config.yaml"
    path.write_text(KERNEL_CONFIG)
    labels = parse_diff_config(path)
    assert list_epoch_products(labels) == labels.epoch_products


def test_fallback_hotpants_when_no_stages(tmp_path):
    path = tmp_path / "diff_config.yaml"
    path.write_text(
        yaml.dump(
            {
                "pipeline": [
                    {"kind": "kernel_fit", "output": "kernel_fit"},
                    {"kind": "forced_photometry", "inputs": {"diffs": "kd_d"}, "output": "lc"},
                ]
            }
        )
    )
    labels = parse_diff_config(path)
    assert labels.hotpants_stages[0].diffs == "hp1_d"
    assert labels.hotpants_stages[1].diffs == "hp2_d"


def test_lightcurve_csv_basename():
    assert lightcurve_csv_basename("prf") == "lightcurve_prf.csv"
    assert lightcurve_csv_basename("ap3", "offset_top") == "lightcurve_ap3_offset_top.csv"


def test_resolve_new_style_filename(tmp_path):
    path = tmp_path / "diff_config.yaml"
    path.write_text(KERNEL_CONFIG)
    labels = parse_diff_config(path)
    lc_dir = tmp_path / "lc_prf_on_diffs"
    lc_dir.mkdir()
    assert resolve_lightcurve_filename("prf", "primary", labels, lc_dir) == "lightcurve_prf.csv"
    assert (
        resolve_lightcurve_filename("ap3", "offset_top", labels, lc_dir)
        == "lightcurve_ap3_offset_top.csv"
    )


def test_lightcurve_selection_key():
    assert lightcurve_selection_key("prf", "primary") == "prf_primary"
    assert lightcurve_selection_key("ap3", "offset_top") == "ap3_offset_top"
    assert lightcurve_selection_key(LEGACY_METHOD, "primary") == "primary"


def test_list_lightcurve_selections_new_style(tmp_path):
    path = tmp_path / "diff_config.yaml"
    path.write_text(KERNEL_CONFIG)
    labels = parse_diff_config(path)
    lc_dir = tmp_path / "lc_prf_on_diffs"
    lc_dir.mkdir()
    (lc_dir / "lightcurve_prf.csv").write_text("btjd,flux\n")
    (lc_dir / "lightcurve_ap3_offset_top.csv").write_text("btjd,flux\n")
    assert list_lightcurve_selections(labels, lc_dir) == ["prf_primary", "ap3_offset_top"]


def test_parse_lightcurve_selection_roundtrip(tmp_path):
    path = tmp_path / "diff_config.yaml"
    path.write_text(KERNEL_CONFIG)
    labels = parse_diff_config(path)
    lc_dir = tmp_path / "lc_prf_on_diffs"
    lc_dir.mkdir()
    assert parse_lightcurve_selection("prf_primary", labels, lc_dir) == ("prf", "primary")
    assert parse_lightcurve_selection("ap3_offset_top", labels, lc_dir) == ("ap3", "offset_top")


def test_resolve_legacy_filename(tmp_path):
    path = tmp_path / "diff_config.yaml"
    path.write_text(
        yaml.dump(
            {
                "pipeline": [
                    {"kind": "forced_photometry", "inputs": {"diffs": "hp_d"}, "output": "lc"},
                ],
                "additional_forced_targets": [{"name": "offset_top"}],
            }
        )
    )
    labels = parse_diff_config(path)
    lc_dir = tmp_path / "lc"
    lc_dir.mkdir()
    (lc_dir / "lightcurve.csv").write_text("btjd,flux\n")
    assert list_photometry_methods(labels, lc_dir) == [LEGACY_METHOD]
    assert resolve_lightcurve_filename(LEGACY_METHOD, "primary", labels, lc_dir) == "lightcurve.csv"
    assert (
        resolve_lightcurve_filename(LEGACY_METHOD, "offset_top", labels, lc_dir)
        == "lightcurve_offset_top.csv"
    )


def test_infer_methods_from_csv_glob(tmp_path):
    path = tmp_path / "diff_config.yaml"
    path.write_text(
        yaml.dump(
            {
                "pipeline": [
                    {"kind": "forced_photometry", "inputs": {"diffs": "hp_d"}, "output": "lc"},
                ],
                "additional_forced_targets": [{"name": "offset_top"}],
            }
        )
    )
    labels = parse_diff_config(path)
    lc_dir = tmp_path / "lc"
    lc_dir.mkdir()
    (lc_dir / "lightcurve_prf.csv").write_text("btjd,flux\n")
    (lc_dir / "lightcurve_prf_offset_top.csv").write_text("btjd,flux\n")
    assert list_photometry_methods(labels, lc_dir) == ["prf"]
    assert list_forced_targets(labels) == ["primary", "offset_top"]

