#!/usr/bin/env python3
"""cvic.py — Cross-validation hyperparameter search for image classifiers using Ray Tune + timm."""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kwargs):
        return it

try:
    import ray
    from ray import tune
    from ray.tune.search.optuna import OptunaSearch
    from ray.tune import RunConfig

    @ray.remote
    class TrialCounter:
        def __init__(self): self._n = 0
        def next(self): self._n += 1; return self._n

    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

from krunic.tunic import (
    set_seed, get_device, get_amp_dtype, build_transforms, create_model,
    freeze_backbone, unfreeze_all, get_optimizer, build_scheduler,
    train_one_epoch, _compute_auroc, MixupCutmixCollator,
    load_search_space_overrides, validate_dataset_path,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("cvic")
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _compute_metric(probs: np.ndarray, labels: np.ndarray, metric: str) -> float:
    if "auroc" in metric:
        return _compute_auroc(probs, labels)
    if "acc" in metric:
        return float(np.mean(np.argmax(probs, axis=1) == labels))
    return _compute_auroc(probs, labels)


def _cvic_trial(config: dict):
    """CV trial: trains one model per fold, reports aggregated metric to Ray Tune."""
    from sklearn.model_selection import StratifiedKFold, KFold

    data_path          = Path(config["data"])
    device             = get_device(config["device"])
    n_folds            = config["n_folds"]
    n_repeats          = config["n_repeats"]
    stratified         = config["stratified"]
    pooling            = config["pooling"]
    tune_metric        = config["tune_metric"]
    epochs             = config["epochs"]
    batch_size         = config["batch_size"]
    img_size           = config["img_size"]
    workers            = config["dataloader_workers"]
    base_seed          = config["seed"]
    num_classes        = config["num_classes"]
    lr                 = config["lr"]
    weight_decay       = config["weight_decay"]
    label_smoothing    = config["label_smoothing"]
    drop_rate          = config["drop_rate"]
    randaug_magnitude  = config["randaugment_magnitude"]
    randaug_num_ops    = config["randaugment_num_ops"]
    mixup_alpha        = config["mixup_alpha"]
    cutmix_alpha       = config["cutmix_alpha"]
    optimizer_name     = config["optimizer"]
    use_amp            = config.get("use_amp", False)
    freeze_bb          = config["freeze_backbone"]

    _counter = ray.get_actor("trial_counter")
    _trial_num = ray.get(_counter.next.remote())
    trial_id = f"{_trial_num}/{config['n_trials']}"

    use_mixup_cutmix = mixup_alpha > 0 or cutmix_alpha > 0
    collate_fn = MixupCutmixCollator(mixup_alpha, cutmix_alpha, num_classes) if use_mixup_cutmix else None

    train_tf = build_transforms(img_size, randaug_magnitude, randaug_num_ops, is_train=True)
    val_tf   = build_transforms(img_size, is_train=False)

    # Two views of the same data: augmented for training, clean for evaluation
    train_dir = data_path / "train"
    base_ds_aug   = datasets.ImageFolder(str(train_dir), transform=train_tf)
    base_ds_clean = datasets.ImageFolder(str(train_dir), transform=val_tf)
    all_labels = [label for _, label in base_ds_aug.samples]
    N = len(base_ds_aug)

    # probs_accum[i] sums out-of-fold softmax probs for sample i across repeats
    probs_accum  = np.zeros((N, num_classes), dtype=np.float64)
    labels_arr   = np.array(all_labels)
    fold_metrics = []

    try:
        for repeat in range(n_repeats):
            repeat_seed = base_seed + repeat * 10000
            set_seed(repeat_seed)

            if stratified:
                kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=repeat_seed)
                splits = list(kf.split(range(N), all_labels))
            else:
                kf = KFold(n_splits=n_folds, shuffle=True, random_state=repeat_seed)
                splits = list(kf.split(range(N)))

            for fold_idx, (train_idx, val_idx) in enumerate(splits):
                fold_seed = repeat_seed + fold_idx
                set_seed(fold_seed)
                fold_label = f"{trial_id} repeat{repeat+1}/{n_repeats} fold{fold_idx+1}/{n_folds}"

                train_loader = DataLoader(
                    Subset(base_ds_aug, train_idx),
                    batch_size=batch_size, shuffle=True,
                    num_workers=workers, pin_memory=(device.type == "cuda"),
                    drop_last=True, collate_fn=collate_fn,
                )
                val_loader = DataLoader(
                    Subset(base_ds_clean, val_idx),
                    batch_size=batch_size, shuffle=False,
                    num_workers=workers, pin_memory=(device.type == "cuda"),
                )

                model = create_model(config["model"], num_classes, config["pretrained"], drop_rate)
                model = model.to(device)
                if freeze_bb > 0:
                    freeze_backbone(model)

                optimizer = get_optimizer(model, optimizer_name, lr, weight_decay)
                scheduler = build_scheduler(optimizer, epochs, len(train_loader))
                criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

                for epoch in range(epochs):
                    if freeze_bb > 0 and epoch == freeze_bb:
                        unfreeze_all(model)
                        optimizer = get_optimizer(model, optimizer_name, lr, weight_decay)
                    train_one_epoch(
                        model, train_loader, optimizer, scheduler, criterion, device,
                        use_soft_labels=use_mixup_cutmix,
                        trial_id=fold_label, epoch=epoch, epochs=epochs,
                        use_amp=use_amp, show_progress=True,
                    )

                # Collect out-of-fold predictions
                amp_dtype = get_amp_dtype()
                model.eval()
                fold_probs, fold_labels = [], []
                with torch.no_grad():
                    for images, labels in val_loader:
                        images = images.to(device)
                        labels = labels.to(device)
                        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                            enabled=use_amp and device.type == "cuda"):
                            outputs = model(images)
                        fold_probs.append(torch.softmax(outputs.float(), dim=1).cpu().numpy())
                        fold_labels.extend(labels.cpu().numpy())

                fold_probs_np  = np.concatenate(fold_probs)
                fold_labels_np = np.array(fold_labels)

                if pooling:
                    probs_accum[val_idx] += fold_probs_np
                else:
                    fold_metrics.append(
                        _compute_metric(fold_probs_np, fold_labels_np, tune_metric)
                    )

                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if pooling:
            probs_final = probs_accum / n_repeats
            metric_val  = _compute_metric(probs_final, labels_arr, tune_metric)
        else:
            metric_val = float(np.nanmean(fold_metrics))

    except torch.cuda.OutOfMemoryError:
        logger.warning("CUDA OOM — reporting 0.0")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        metric_val = 0.0

    # Report both metrics so tunic-plotter can read either
    auroc = metric_val if "auroc" in tune_metric else float("nan")
    acc   = metric_val if "acc"   in tune_metric else float("nan")
    tune.report({tune_metric: metric_val, "val_auroc": auroc, "val_acc": acc})


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run_cv(args):
    if not args.model:
        logger.error("--model is required. E.g. --model resnet18")
        sys.exit(1)

    if not RAY_AVAILABLE:
        logger.error("Ray is required. Install with: pip install ray[tune]")
        sys.exit(1)

    data_root = args.data
    data_path = Path(data_root)
    validate_dataset_path(data_path)

    train_dir = data_path / "train"
    tmp_ds = datasets.ImageFolder(str(train_dir))
    num_classes = len(tmp_ds.classes)
    N = len(tmp_ds)

    logger.info(
        f"Dataset: {data_root} | Samples: {N} | Classes: {num_classes} | Model: {args.model}"
    )
    logger.info(
        f"CV: {args.folds} folds x {args.repeats} repeats | "
        f"{'stratified' if args.stratified else 'random'} | "
        f"{'pooling' if args.pooling else 'averaging'}"
    )
    if args.amp:
        amp_dtype = get_amp_dtype()
        logger.info(f"AMP enabled: {'BF16' if amp_dtype == torch.bfloat16 else 'FP16'}")

    use_gpu = args.device != "cpu"

    ss = {}
    if args.search_space:
        ss = load_search_space_overrides(args.search_space)

    hp_keys = ["lr", "weight_decay", "label_smoothing", "drop_rate",
               "randaugment_magnitude", "randaugment_num_ops",
               "mixup_alpha", "cutmix_alpha", "optimizer"]

    search_space = {
        "data":              data_root,
        "model":             args.model,
        "pretrained":        args.pretrained,
        "epochs":            args.epochs,
        "batch_size":        args.batch_size,
        "img_size":          args.img_size,
        "freeze_backbone":   args.freeze_backbone,
        "seed":              args.seed,
        "dataloader_workers": args.workers,
        "num_classes":       num_classes,
        "device":            args.device,
        "use_amp":           args.amp,
        "n_trials":          args.n_trials,
        "n_folds":           args.folds,
        "n_repeats":         args.repeats,
        "stratified":        args.stratified,
        "pooling":           args.pooling,
        "tune_metric":       args.tune_metric,
        "lr":                    tune.loguniform(ss.get("lr_min", 1e-5),    ss.get("lr_max", 1e-1)),
        "weight_decay":          tune.loguniform(ss.get("wd_min", 1e-6),    ss.get("wd_max", 1e-2)),
        "label_smoothing":       tune.uniform(   ss.get("ls_min", 0.0),     ss.get("ls_max", 0.2)),
        "drop_rate":             tune.uniform(   ss.get("dr_min", 0.0),     ss.get("dr_max", 0.5)),
        "randaugment_magnitude": tune.randint(   ss.get("ra_mag_min", 0),   ss.get("ra_mag_max", 15) + 1),
        "randaugment_num_ops":   tune.randint(   ss.get("ra_ops_min", 1),   ss.get("ra_ops_max", 3) + 1),
        "mixup_alpha":           tune.uniform(   ss.get("mixup_min", 0.0),  ss.get("mixup_max", 0.4)),
        "cutmix_alpha":          tune.uniform(   ss.get("cutmix_min", 0.0), ss.get("cutmix_max", 1.0)),
        "optimizer":             tune.choice(    ss.get("optimizers",       ["AdamW", "SGD"])),
    }

    trainable = tune.with_resources(
        _cvic_trial,
        resources={"GPU": 1 if use_gpu else 0, "CPU": max(1, args.workers)},
    )

    search_alg = OptunaSearch(metric=args.tune_metric, mode="max", seed=args.seed)

    if args.ray_storage:
        storage_path = args.ray_storage
    else:
        storage_path = str(Path(f"{args.prefix}.json").parent.resolve() / "ray_results")

    class _IntermediateResultsCallback(tune.Callback):
        def __init__(self):
            self.completed = []

        def on_trial_complete(self, iteration, trials, trial, **kwargs):
            if not trial.last_result:
                return
            self.completed.append({
                "val_acc":   trial.last_result.get("val_acc"),
                "val_auroc": trial.last_result.get("val_auroc"),
                args.tune_metric: trial.last_result.get(args.tune_metric),
                "params":    {k: trial.config[k] for k in hp_keys if k in trial.config},
                "status":    trial.status,
            })
            valid = [t for t in self.completed if t.get(args.tune_metric) is not None]
            if not valid:
                return
            best = max(valid, key=lambda t: t.get(args.tune_metric, float("-inf")))
            snapshot = {
                "best_val_acc":     best.get("val_acc"),
                "best_val_auroc":   best.get("val_auroc"),
                "best_params":      best.get("params", {}),
                "completed_trials": len(self.completed),
                "model":            args.model,
                "epochs":           args.epochs,
                "n_folds":          args.folds,
                "n_repeats":        args.repeats,
                "all_trials":       self.completed,
                "status":           "in_progress",
            }
            with open(f"{args.prefix}.json", "w") as f:
                json.dump(snapshot, f, indent=2)

    run_config = RunConfig(
        storage_path=storage_path,
        name="cvic_study",
        callbacks=[_IntermediateResultsCallback()],
    )

    ray_address = getattr(args, "ray_address", None)
    ray.init(address=ray_address, ignore_reinit_error=True, namespace="cvic")
    logger.info(f"Ray initialized (address={ray_address or 'local'})")

    TrialCounter.options(name="trial_counter", lifetime="detached", get_if_exists=True).remote()

    tuner = tune.Tuner(
        trainable,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            search_alg=search_alg,
            num_samples=args.n_trials,
        ),
        run_config=run_config,
    )

    start_time = time.time()
    try:
        results = tuner.fit()
    except KeyboardInterrupt:
        logger.info("Interrupted — saving current results...")
        raise
    total_time = time.time() - start_time

    best = results.get_best_result(metric=args.tune_metric, mode="max")
    best_metric   = best.metrics[args.tune_metric]
    best_val_acc   = best.metrics.get("val_acc",   float("nan"))
    best_val_auroc = best.metrics.get("val_auroc", float("nan"))
    best_params    = {k: best.config[k] for k in hp_keys if k in best.config}

    all_trials = []
    completed = errored = 0
    for r in results:
        if r.error:
            errored += 1
            state = "ERROR"
        else:
            completed += 1
            state = "COMPLETE"
        all_trials.append({
            "val_acc":       r.metrics.get("val_acc")   if r.metrics else None,
            "val_auroc":     r.metrics.get("val_auroc") if r.metrics else None,
            args.tune_metric: r.metrics.get(args.tune_metric) if r.metrics else None,
            "params":        {k: r.config[k] for k in hp_keys if k in r.config} if r.config else {},
            "state":         state,
        })

    output = {
        "best_val_acc":        best_val_acc,
        "best_val_auroc":      best_val_auroc,
        f"best_{args.tune_metric}": best_metric,
        "best_params":         best_params,
        "model":               args.model,
        "dataset":             data_root,
        "num_classes":         num_classes,
        "n_trials":            args.n_trials,
        "epochs":              args.epochs,
        "n_folds":             args.folds,
        "n_repeats":           args.repeats,
        "stratified":          args.stratified,
        "pooling":             args.pooling,
        "tune_metric":         args.tune_metric,
        "completed_trials":    completed,
        "errored_trials":      errored,
        "total_time_seconds":  total_time,
        "all_trials":          all_trials,
    }

    with open(f"{args.prefix}.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nBest {args.tune_metric}: {best_metric:.4f}")
    if not np.isnan(best_val_acc):
        print(f"Best val_acc:   {best_val_acc:.4f}")
    if not np.isnan(best_val_auroc):
        print(f"Best val_auroc: {best_val_auroc:.4f}")
    print("Best params:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    print(f"\nResults saved to: {args.prefix}.json")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    from importlib.metadata import version as _version, PackageNotFoundError
    try:
        _ver = _version("krunic")
    except PackageNotFoundError:
        _ver = "dev"
    p = argparse.ArgumentParser(
        description=f"cvic {_ver} — cross-validation hyperparameter search for image classifiers",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {_ver}")
    p.add_argument("--data",    type=str, required=True,
                   help="Path to dataset root (ImageFolder layout with train/ subdirectory)")
    p.add_argument("--model",   type=str, default=None,
                   help="Any timm model name (e.g. resnet18, vit_small_patch16_224)")
    p.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True,
                   help="Use timm pretrained weights")
    p.add_argument("--n-trials",  type=int, default=30, dest="n_trials",
                   help="Number of Optuna trials")
    p.add_argument("--epochs",    type=int, default=30,
                   help="Training epochs per fold")
    p.add_argument("--folds",     type=int, default=5,
                   help="Number of CV folds")
    p.add_argument("--repeats",   type=int, default=1,
                   help="Number of times to repeat the full CV with different random splits")
    p.add_argument("--stratified", action=argparse.BooleanOptionalAction, default=True,
                   help="Use stratified (class-balanced) folds")
    p.add_argument("--pooling",   action="store_true", default=False,
                   help="Pool out-of-fold predictions and compute metric once (default: average per-fold metrics)")
    p.add_argument("--tune-metric", type=str, default="val_auroc", dest="tune_metric",
                   help="Metric for Optuna trial selection")
    p.add_argument("--batch-size",  type=int, default=32, dest="batch_size",
                   help="Batch size per fold")
    p.add_argument("--prefix",    type=str, default="cvic",
                   help="Prefix for output files")
    p.add_argument("--seed",      type=int, default=42,
                   help="Random seed")
    p.add_argument("--device",    type=str, default="auto",
                   help="Device: auto, cuda, mps, or cpu")
    p.add_argument("--workers",   type=int, default=4,
                   help="DataLoader worker count")
    p.add_argument("--img-size",  type=int, default=224, dest="img_size",
                   help="Input image resolution")
    p.add_argument("--freeze-backbone", type=int, default=0, dest="freeze_backbone",
                   help="Epochs to freeze backbone; 0 = no freeze")
    p.add_argument("--amp",       action="store_true", default=False,
                   help="Enable automatic mixed precision (BF16 on H100/A100, FP16 on older GPUs)")
    p.add_argument("--search-space", type=str, default=None, dest="search_space",
                   help="YAML file to override search space bounds")
    p.add_argument("--ray-address", type=str, default=None, dest="ray_address",
                   help="Ray cluster address (default: start local cluster)")
    p.add_argument("--ray-storage", type=str, default=None, dest="ray_storage",
                   help="Ray Tune storage path for trial checkpoints")
    return p.parse_args()


def main():
    args = parse_args()
    run_cv(args)


if __name__ == "__main__":
    main()
