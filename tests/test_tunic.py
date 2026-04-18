"""Unit and integration tests for tunic."""

import json
import tempfile
from pathlib import Path

import numpy as np
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
# _compute_auroc
# ---------------------------------------------------------------------------

def test_compute_auroc_binary_perfect():
    from krunic.tunic import _compute_auroc
    probs = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    labels = np.array([0, 0, 1, 1])
    assert _compute_auroc(probs, labels) == 1.0


def test_compute_auroc_binary_random():
    from krunic.tunic import _compute_auroc
    probs = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
    labels = np.array([0, 0, 1, 1])
    auroc = _compute_auroc(probs, labels)
    assert 0.0 <= auroc <= 1.0


def test_compute_auroc_multiclass():
    from krunic.tunic import _compute_auroc
    # 3-class, each class perfectly predicted
    probs = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ])
    labels = np.array([0, 1, 2, 0])
    auroc = _compute_auroc(probs, labels)
    assert auroc == pytest.approx(1.0)


def test_compute_auroc_single_class_returns_nan():
    from krunic.tunic import _compute_auroc
    # Only one class present — undefined AUROC
    probs = np.array([[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]])
    labels = np.array([0, 0, 0])
    assert np.isnan(_compute_auroc(probs, labels))


# ---------------------------------------------------------------------------
# build_transforms
# ---------------------------------------------------------------------------

def test_build_transforms_train_output_shape():
    from krunic.tunic import build_transforms
    from PIL import Image
    tf = build_transforms(img_size=64, randaug_magnitude=0, is_train=True)
    img = Image.fromarray(np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8))
    tensor = tf(img)
    assert tensor.shape == (3, 64, 64)


def test_build_transforms_val_output_shape():
    from krunic.tunic import build_transforms
    from PIL import Image
    tf = build_transforms(img_size=64, is_train=False)
    img = Image.fromarray(np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8))
    tensor = tf(img)
    assert tensor.shape == (3, 64, 64)


def test_build_transforms_normalized():
    from krunic.tunic import build_transforms
    from PIL import Image
    tf = build_transforms(img_size=64, is_train=False)
    img = Image.fromarray(np.full((128, 128, 3), 128, dtype=np.uint8))
    tensor = tf(img)
    # After ImageNet normalization values should not be in [0, 1]
    assert tensor.min().item() < 0.5


# ---------------------------------------------------------------------------
# make_stratified_split
# ---------------------------------------------------------------------------

def test_make_stratified_split_sizes(tmp_path):
    from torchvision import datasets
    from krunic.tunic import make_stratified_split
    # Build a tiny imagefolder with 3 classes, 10 images each
    for cls in ["a", "b", "c"]:
        d = tmp_path / cls
        d.mkdir()
        for i in range(10):
            from PIL import Image
            Image.fromarray(np.random.randint(0, 255, (8, 8, 3), dtype=np.uint8)).save(d / f"{i}.jpg")
    ds = datasets.ImageFolder(str(tmp_path))
    train, val = make_stratified_split(ds, val_fraction=0.2, seed=42)
    assert len(train) + len(val) == 30
    assert len(val) == pytest.approx(6, abs=3)  # ~20% of 30


def test_make_stratified_split_reproducible(tmp_path):
    from torchvision import datasets
    from krunic.tunic import make_stratified_split
    for cls in ["a", "b"]:
        d = tmp_path / cls
        d.mkdir()
        for i in range(10):
            from PIL import Image
            Image.fromarray(np.random.randint(0, 255, (8, 8, 3), dtype=np.uint8)).save(d / f"{i}.jpg")
    ds = datasets.ImageFolder(str(tmp_path))
    _, val1 = make_stratified_split(ds, val_fraction=0.2, seed=42)
    _, val2 = make_stratified_split(ds, val_fraction=0.2, seed=42)
    assert list(val1.indices) == list(val2.indices)


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def test_cli_dash_args(monkeypatch):
    """Verify that dashed args (--n-trials, --training-fraction) are accepted."""
    import sys
    from krunic.tunic import parse_args
    monkeypatch.setattr(sys, "argv", [
        "tunic",
        "--data", "/tmp",
        "--model", "resnet18",
        "--n-trials", "5",
        "--training-fraction", "0.1",
        "--epochs", "2",
    ])
    args = parse_args()
    assert args.n_trials == 5
    assert args.training_fraction == pytest.approx(0.1)


def test_cli_amp_flag(monkeypatch):
    import sys
    from krunic.tunic import parse_args
    monkeypatch.setattr(sys, "argv", [
        "tunic", "--data", "/tmp", "--model", "resnet18", "--amp",
    ])
    args = parse_args()
    assert args.amp is True


def test_cli_prefix(monkeypatch):
    import sys
    from krunic.tunic import parse_args
    monkeypatch.setattr(sys, "argv", [
        "tunic", "--data", "/tmp", "--model", "resnet18", "--prefix", "myrun",
    ])
    args = parse_args()
    assert args.prefix == "myrun"


# ---------------------------------------------------------------------------
# smoke test (CPU, end-to-end)
# ---------------------------------------------------------------------------

def test_smoke(tmp_path):
    """End-to-end smoke test: tuning + final training on synthetic data."""
    import argparse
    from PIL import Image
    from krunic.tunic import run_smoke_test

    for split in ["train", "val"]:
        for cls in ["cat", "dog", "bird"]:
            d = tmp_path / split / cls
            d.mkdir(parents=True)
            for i in range(10):
                Image.fromarray(
                    np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
                ).save(d / f"img_{i}.jpg")

    args = argparse.Namespace(device="cpu", smoke_test=True)
    run_smoke_test(args)
