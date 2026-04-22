"""Unit and smoke tests for cvic."""

import argparse
import sys

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# _compute_metric
# ---------------------------------------------------------------------------

def test_compute_metric_auroc_perfect():
    from krunic.cvic import _compute_metric
    probs = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    labels = np.array([0, 0, 1, 1])
    assert _compute_metric(probs, labels, "val_auroc") == 1.0


def test_compute_metric_acc_perfect():
    from krunic.cvic import _compute_metric
    probs = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3]])
    labels = np.array([0, 1, 0])
    assert _compute_metric(probs, labels, "val_acc") == 1.0


def test_compute_metric_acc_partial():
    from krunic.cvic import _compute_metric
    probs = np.array([[0.9, 0.1], [0.9, 0.1], [0.1, 0.9]])
    labels = np.array([0, 1, 0])  # second wrong, third wrong
    acc = _compute_metric(probs, labels, "val_acc")
    assert acc == pytest.approx(1 / 3)


def test_compute_metric_unknown_falls_back_to_auroc():
    from krunic.cvic import _compute_metric
    probs = np.array([[1.0, 0.0], [0.0, 1.0]])
    labels = np.array([0, 1])
    result = _compute_metric(probs, labels, "something_else")
    assert result == 1.0


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------

def test_cli_defaults(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["cvic", "--data", "/tmp", "--model", "resnet18"])
    from krunic.cvic import parse_args
    args = parse_args()
    assert args.folds == 5
    assert args.repeats == 1
    assert args.stratified is True
    assert args.pooling is False
    assert args.tune_metric == "val_auroc"
    assert args.n_trials == 30
    assert args.epochs == 30


def test_cli_folds_repeats(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "cvic", "--data", "/tmp", "--model", "resnet18",
        "--folds", "3", "--repeats", "2",
    ])
    from krunic.cvic import parse_args
    args = parse_args()
    assert args.folds == 3
    assert args.repeats == 2


def test_cli_pooling(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "cvic", "--data", "/tmp", "--model", "resnet18", "--pooling",
    ])
    from krunic.cvic import parse_args
    args = parse_args()
    assert args.pooling is True


def test_cli_no_stratified(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "cvic", "--data", "/tmp", "--model", "resnet18", "--no-stratified",
    ])
    from krunic.cvic import parse_args
    args = parse_args()
    assert args.stratified is False


def test_cli_tune_metric(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "cvic", "--data", "/tmp", "--model", "resnet18", "--tune-metric", "val_acc",
    ])
    from krunic.cvic import parse_args
    args = parse_args()
    assert args.tune_metric == "val_acc"


def test_cli_prefix(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "cvic", "--data", "/tmp", "--model", "resnet18", "--prefix", "myexp",
    ])
    from krunic.cvic import parse_args
    args = parse_args()
    assert args.prefix == "myexp"


# ---------------------------------------------------------------------------
# smoke test (CPU, end-to-end with Ray local cluster)
# ---------------------------------------------------------------------------

def test_smoke_cvic(tmp_path):
    """End-to-end smoke test: CV search on synthetic data, CPU, 1 trial, 2 folds, 1 epoch."""
    from PIL import Image
    from krunic.cvic import run_cv

    for cls in ["cat", "dog", "bird"]:
        d = tmp_path / "train" / cls
        d.mkdir(parents=True)
        for i in range(12):
            Image.fromarray(
                np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
            ).save(d / f"img_{i}.jpg")

    args = argparse.Namespace(
        data=str(tmp_path),
        model="resnet18",
        pretrained=False,
        n_trials=1,
        epochs=1,
        folds=2,
        repeats=1,
        stratified=True,
        pooling=False,
        tune_metric="val_acc",
        batch_size=4,
        prefix=str(tmp_path / "cvic_smoke"),
        seed=0,
        device="cpu",
        workers=0,
        img_size=32,
        freeze_backbone=0,
        amp=False,
        search_space=None,
        ray_address=None,
        ray_storage=str(tmp_path / "ray_results"),
    )

    run_cv(args)

    import json
    out_path = f"{args.prefix}.json"
    with open(out_path) as f:
        result = json.load(f)
    assert "best_val_acc" in result
    assert result["n_folds"] == 2
    assert result["completed_trials"] == 1
