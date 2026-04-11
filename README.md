# krunic

Automated hyperparameter search for image classifiers — from dataset to tuned model with one command.

Built on [Ray Tune](https://docs.ray.io/en/latest/tune/index.html), [Optuna](https://optuna.org/), [timm](https://github.com/huggingface/pytorch-image-models), and [SkyPilot](https://skypilot.readthedocs.io/).

## Install (Mac and Linux)

```bash
$ pipx install krunic
```

This installs three commands: `tunic` (local training), `krunic` (cloud launcher), and `tunic-plotter` (results visualizer).

## Quick start

**Local:**
```bash
$ tunic --data /path/to/dataset --model resnet50 --n_trials 30 --epochs 30 --output results.json
```

**Cloud (AWS):**
```bash
$ krunic \
  --cluster my-cluster \
  --workdir ~/github/krunic \
  --s3-path my-dataset \
  --model resnet50 \
  --accelerator T4:4 \
  --num-nodes 4 \
  --n-trials 48 \
  --n-epochs 50 \
  --prefix run1
```

**Train final model from tuning results:**
```bash
$ tunic --final results.json --data /path/to/dataset --epochs 50 --amp
```

**Plot results:**
```bash
$ tunic-plotter results.json
```

## Results on standard benchmarks

| Dataset | Model | Metric | Val AUROC | Test AUROC | SOTA |
|---|---|---|---|---|---|
| PCam (patch camelyon) | ResNet18 | AUROC | 0.96 | 0.97 | 0.96 |
| TinyImageNet | ViT-Small | Acc | 0.87  | | 0.90 |
| ChestMNIST | ResNet18 | AUROC | 0.76 | 0.75 |  |
| TissueMNIST | ResNet18 | AUROC |  | 0.94 | |

All runs use generic off-the-shelf models with no domain-specific modifications.

## Search space

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

Override any part with a YAML file via `--search-space`.

## tunic — local hyperparameter search

```
tunic --data PATH --model MODEL [options]
```

| Flag | Default | Description |
|---|---|---|
| `--data` | required | Dataset root (ImageFolder or WebDataset) |
| `--model` | required | Any timm model name |
| `--n_trials` | 80 | Number of Optuna trials |
| `--epochs` | 30 | Training epochs per trial (also used for `--final`) |
| `--tune-metric` | `val_auroc` | Metric for trial selection and pruning |
| `--training_fraction` | 1.0 | Fraction of training data (val always uses 1.0) |
| `--batch-size` | 32 | Batch size per trial |
| `--amp` | — | Enable automatic mixed precision |
| `--ray-address` | local | Ray cluster address |
| `--ray-storage` | local | Ray Tune storage path (local or S3 URI) |
| `--resume` | — | Warm-start from a previous experiment directory |
| `--final` | — | Skip tuning; train final model from results JSON |
| `--combine` | — | Train final model on train+val combined |
| `--final-model` | `tunic_final.pt` | Output path for final model weights |
| `--final-stats` | — | Output path for final model stats (JSON) |
| `--device` | `auto` | `auto`, `cuda`, `mps`, or `cpu` |
| `--smoke-test` | — | Quick end-to-end test with synthetic data |

## krunic — cloud launcher

krunic generates a SkyPilot YAML and launches the job. The dataset is S3-mounted (or copied); results are uploaded to S3 when the job completes.

**Prerequisites:** SkyPilot configured with AWS credentials; dataset in S3.

`--workdir` defaults to the installed package directory (contains `tunic.py` and `requirements.txt`). Override it only if you are developing from a local source checkout and want to test unpublished changes.

```
krunic --cluster NAME --workdir DIR --s3-path PATH --model MODEL [options]
```

| Flag | Default | Description |
|---|---|---|
| `--cluster` | required | SkyPilot cluster name |
| `--workdir` | package dir | Local directory synced to the cluster |
| `--s3-path` | required | Dataset path within the S3 bucket |
| `--model` | required | Any timm model name |
| `--accelerator` | `T4:4` | GPU spec (e.g. `T4:4`, `A10G:1`, `A100:8`) |
| `--num-nodes` | 1 | Number of cluster nodes |
| `--n-trials` | 30 | Number of Optuna trials |
| `--n-epochs` | 30 | Training epochs per trial |
| `--batch-size` | 32 | Batch size per trial |
| `--training-fraction` | 1.0 | Fraction of training data per trial |
| `--tune-metric` | `val_auroc` | Metric for trial selection and pruning |
| `--bucket` | `image.data` | S3 bucket name |
| `--prefix` | `tunic` | Prefix for output files and S3 paths |
| `--spot` | — | Use spot instances (with retry-until-up) |
| `--copy` | — | Copy data from S3 to local disk instead of mounting |
| `--idle-minutes` | 60 | Auto-stop cluster after N idle minutes |
| `--no-autostop` | — | Disable auto-stop |

Results are uploaded to `s3://<bucket>/ray-results/<prefix>/<prefix>_results.json`.

## tunic-plotter — visualize results

```bash
tunic-plotter results.json                  # plots val_auroc and val_acc
tunic-plotter results.json --metric val_acc # single metric
tunic-plotter results.json --trial_sort     # keep original trial order, show running best
```

Saves PNG files alongside the results JSON.

## Dataset format

tunic auto-detects the dataset format:

- **ImageFolder** — standard `split/class/image.ext` layout
- **WebDataset** — sharded TAR files; detected when `wds/dataset_info.json` exists

## Scaling

Concurrent trials = total GPUs. `--num-nodes 4 --accelerator T4:4` = 16 concurrent trials.

Optuna's TPE needs ~20 trials before it outperforms random search. 32–64 trials is a practical range for most problems.

## Output format

```json
{
  "model": "resnet18",
  "best_val_auroc": 0.963,
  "best_val_acc": 0.891,
  "best_params": {
    "optimizer": "AdamW",
    "lr": 0.0028,
    "weight_decay": 3.6e-06,
    "label_smoothing": 0.058,
    "drop_rate": 0.183
  },
  "n_trials": 48,
  "completed_trials": 48,
  "epochs": 50,
  "all_trials": [...]
}
```
