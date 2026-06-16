import textwrap
from pathlib import Path

import yaml

from review.pipeline_labels import parse_diff_config


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
    assert labels.bkg_label == "kd_b"
    assert labels.conv_template_label == "tmpl_conv"
    assert labels.kernel_fit_dir == "kernel_fit"
    assert len(labels.hotpants_stages) >= 2
    assert labels.hotpants_stages[0].diffs == "hp1_d"
    assert labels.hotpants_stages[1].diffs == "hp2_d"


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
