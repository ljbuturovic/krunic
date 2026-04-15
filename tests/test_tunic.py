"""Unit and integration tests for tunic."""

import json
import tempfile
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# _detect_format
# ---------------------------------------------------------------------------

def test_detect_format_s3():
    from krunic.tunic import _detect_format
    assert _detect_format("s3://my-bucket/dataset") == "webdataset"


def test_detect_format_imagefolder(tmp_path):
    from krunic.tunic import _detect_format
    assert _detect_format(str(tmp_path)) == "imagefolder"


def test_detect_format_webdataset(tmp_path):
    from krunic.tunic import _detect_format
    (tmp_path / "wds").mkdir()
    (tmp_path / "wds" / "dataset_info.json").write_text("{}")
    assert _detect_format(str(tmp_path)) == "webdataset"


# ---------------------------------------------------------------------------
# load_search_space_overrides
# ---------------------------------------------------------------------------

def test_load_search_space_overrides(tmp_path):
    from krunic.tunic import load_search_space_overrides
    f = tmp_path / "ss.yaml"
    f.write_text("lr: [0.001, 0.01]\noptimizer: [AdamW]\n")
    result = load_search_space_overrides(str(f))
    assert result["lr"] == [0.001, 0.01]
    assert result["optimizer"] == ["AdamW"]


def test_load_search_space_overrides_empty(tmp_path):
    from krunic.tunic import load_search_space_overrides
    f = tmp_path / "empty.yaml"
    f.write_text("")
    assert load_search_space_overrides(str(f)) == {}


# ---------------------------------------------------------------------------
# smoke test (CPU, end-to-end)
# ---------------------------------------------------------------------------

def test_smoke(tmp_path):
    """End-to-end smoke test: tuning + final training on synthetic data."""
    import argparse
    import numpy as np
    from PIL import Image
    from krunic.tunic import run_smoke_test

    for split in ["train", "val"]:
        for cls in ["cat", "dog", "bird"]:
            d = tmp_path / split / cls
            d.mkdir(parents=True)
            for i in range(10):
                img = Image.fromarray(
                    np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
                )
                img.save(d / f"img_{i}.jpg")

    args = argparse.Namespace(device="cpu", smoke_test=True)
    run_smoke_test(args)
