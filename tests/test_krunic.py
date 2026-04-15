"""Unit tests for krunic YAML generation."""

import argparse

import pytest


def make_args(**kwargs):
    defaults = dict(
        cluster="test-cluster",
        cloud="aws",
        accelerator="T4:4",
        num_nodes=1,
        disk_size=200,
        spot=False,
        bucket="my-bucket",
        requirements=None,
        workdir="/tmp/workdir",
        model="resnet50",
        n_trials=30,
        n_epochs=30,
        s3_path="my-dataset",
        prefix="run1",
        training_fraction=1.0,
        idle_minutes=60,
        copy=False,
        no_autostop=False,
        tune_metric="val_auroc",
        batch_size=32,
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def test_build_yaml_keys():
    from krunic.krunic import build_yaml
    data = build_yaml(make_args())
    assert "resources" in data
    assert "envs" in data
    assert "setup" in data
    assert "run" in data


def test_build_yaml_envs():
    from krunic.krunic import build_yaml
    data = build_yaml(make_args(model="resnet18", n_trials=10, n_epochs=5))
    assert data["envs"]["MODEL"] == "resnet18"
    assert data["envs"]["N_TRIALS"] == "10"
    assert data["envs"]["EPOCHS"] == "5"


def test_build_yaml_spot():
    from krunic.krunic import build_yaml
    data = build_yaml(make_args(spot=True))
    assert data["resources"].get("use_spot") is True


def test_build_yaml_no_spot():
    from krunic.krunic import build_yaml
    data = build_yaml(make_args(spot=False))
    assert "use_spot" not in data["resources"]


def test_build_yaml_mount():
    from krunic.krunic import build_yaml
    data = build_yaml(make_args(copy=False))
    assert "file_mounts" in data


def test_build_yaml_copy_no_mount():
    from krunic.krunic import build_yaml
    data = build_yaml(make_args(copy=True))
    assert "file_mounts" not in data


def test_build_yaml_s3_path():
    from krunic.krunic import build_yaml
    data = build_yaml(make_args(bucket="my-bucket", prefix="exp1"))
    assert "my-bucket" in data["envs"]["RAY_RESULTS"]
    assert "exp1" in data["envs"]["RAY_RESULTS"]
