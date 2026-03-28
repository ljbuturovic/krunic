# tunic — Hyperparameter Tuning for Image Classifiers

**tunic** is an automated hyperparameter search system for image classifiers, built on [Ray Tune](https://docs.ray.io/en/latest/tune/index.html), [Optuna](https://optuna.org/), and [timm](https://github.com/huggingface/pytorch-image-models). **krunic** is its companion launcher that runs tunic on multi-node GPU clusters via [SkyPilot](https://skypilot.readthedocs.io/).

Together they let you go from a dataset on S3 to a fully tuned image classifier with a single command.

## Features

- **Any timm model** — ResNet, EfficientNet, ConvNeXt, ViT, and thousands more
- **Bayesian search** via Optuna TPE with ASHA early stopping
- **Multi-node, multi-GPU** — scales across any number of nodes; one GPU per trial
- **WebDataset and ImageFolder** — auto-detected from dataset layout
- **Configurable search metric** — optimize for AUROC, accuracy, or any reported metric
- **Data subsampling** — fast HP search on a fraction of training data; full val set always used
- **Resumable** — warm-start from a previous Ray Tune experiment directory
- **S3-native** — datasets mounted directly from S3; results uploaded automatically
- **Spot instance support** — retry-until-up for cost-efficient cloud runs

## Search Space

tunic searches over the following hyperparameters:

| Parameter | Range |
|---|---|
| Optimizer | AdamW, SGD |
| Learning rate | 1e-5 – 1e-1 (log) |
| Weight decay | 1e-6 – 1e-1 (log) |
| Label smoothing | 0 – 0.3 |
| Dropout rate | 0 – 0.5 |
| RandAugment magnitude | 1 – 15 |
| RandAugment num ops | 1 – 4 |
| Mixup alpha | 0 – 0.5 |
| CutMix alpha | 0 – 1.0 |

The search space can be overridden with a YAML file via `--search-space`.

## Local Usage (tunic.py)

```bash
python tunic.py \
  --data     /path/to/dataset \
  --model    resnet18 \
  --n_trials 30 \
  --epochs   50 \
  --output   results.json
```

**Key options:**

| Flag | Default | Description |
|---|---|---|
| `--data` | required | Dataset root (ImageFolder or WebDataset layout) |
| `--model` | `resnet50` | Any timm model name |
| `--n_trials` | 80 | Number of Optuna trials |
| `--epochs` | 30 | Training epochs per trial |
| `--tune-metric` | `val_auroc` | Metric used for trial selection and pruning |
| `--training_fraction` | 1.0 | Fraction of training data (val always uses 1.0) |
| `--ray-address` | local | Ray cluster address (e.g. `localhost:6385`) |
| `--ray-storage` | local | Ray Tune storage path (local dir or S3 URI) |
| `--resume` | — | Warm-start from a previous experiment directory |
| `--final` | — | Skip tuning; train final model from results JSON |
| `--device` | `auto` | `auto`, `cuda`, `mps`, or `cpu` |
| `--smoke-test` | — | Quick end-to-end test with synthetic data |

## Cloud Usage (krunic.py)

krunic generates a SkyPilot YAML and launches the job. The dataset is mounted directly from S3; results are uploaded to S3 when the job completes.

**Prerequisites:**
- [SkyPilot](https://skypilot.readthedocs.io/) installed and configured
- AWS credentials with S3 and EC2 access
- Dataset in S3 (ImageFolder or WebDataset format)

```bash
python krunic.py \
  --cluster     my-cluster \
  --s3-path     my-dataset \
  --model       resnet18 \
  --accelerator T4:4 \
  --num-nodes   4 \
  --n-trials    48 \
  --n-epochs    50 \
  --training-fraction 0.5 \
  --prefix      run1
```

Results are uploaded to `s3://<bucket>/ray-results/<prefix>/<prefix>_results.json`.

**Key options:**

| Flag | Default | Description |
|---|---|---|
| `--cluster` | required | SkyPilot cluster name |
| `--s3-path` | required | Dataset path within the S3 bucket |
| `--model` | `resnet50` | Any timm model name |
| `--accelerator` | `T4:4` | GPU spec (e.g. `T4:4`, `A10G:1`, `A100:8`) |
| `--num-nodes` | 1 | Number of cluster nodes |
| `--n-trials` | 30 | Number of Optuna trials |
| `--n-epochs` | 30 | Training epochs per trial |
| `--training-fraction` | 1.0 | Fraction of training data per trial |
| `--tune-metric` | `val_auroc` | Metric used for trial selection and pruning |
| `--bucket` | `image.data` | S3 bucket name |
| `--prefix` | `tunic` | Prefix for output files and S3 paths |
| `--spot` | — | Use spot instances (with retry-until-up) |
| `--copy` | — | Copy data from S3 to local disk instead of mounting |
| `--idle-minutes` | 60 | Auto-stop cluster after N idle minutes |
| `--no-autostop` | — | Disable auto-stop |

## Dataset Format

tunic auto-detects the dataset format by looking for `wds/dataset_info.json`:

- **WebDataset** — sharded TAR files, detected when `wds/dataset_info.json` exists
- **ImageFolder** — standard `train/class_name/image.png` layout

## Output

Results are saved as JSON:

```json
{
  "best_val_auroc": 0.747,
  "best_val_acc": 0.312,
  "best_params": {
    "optimizer": "AdamW",
    "lr": 0.0028,
    ...
  },
  "model": "resnet18",
  "n_trials": 48,
  "completed_trials": 48,
  "all_trials": [...]
}
```

## Scaling

Concurrent trials = total GPUs across all nodes. For example, `--num-nodes 4 --accelerator T4:4` gives 16 concurrent trials. Trial count is independent of concurrency — schedule as many trials as your search budget allows.

A rule of thumb: Optuna's TPE needs ~20 trials before it meaningfully outperforms random search. 32–64 trials is a practical range for most problems.

## Requirements

```
ray[tune]
optuna
timm
torch
torchvision
webdataset
PyYAML
```

For cloud launches, also install:
```
skypilot[aws]
```
