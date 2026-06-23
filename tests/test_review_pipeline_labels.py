import textwrap
from pathlib import Path

import yaml

from review.pipeline_labels import list_epoch_products, parse_diff_config


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
      - kind: hotpants
        output:
          diffs: hp1_d
          bkg: hp1_b
      - kind: hotpants
        output:
          diffs: hp2_d
          bkg: hp2_b
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
