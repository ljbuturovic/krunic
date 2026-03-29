#!/usr/bin/env -S python3 -B
"""tunic.py — Hyperparameter tuning for image classifiers using Ray Tune + timm."""

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.transforms import RandAugment

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kwargs):
        return it

try:
    import yaml
except ImportError:
    yaml = None

try:
    import ray
    import ray.train
    import ray.train.torch
    from ray import tune
    from ray.tune.search.optuna import OptunaSearch
    from ray.tune.schedulers import ASHAScheduler
    from ray.train import ScalingConfig
    from ray.tune import RunConfig
    from ray.train.torch import TorchTrainer
    @ray.remote
    class TrialCounter:
        def __init__(self): self._n = 0
        def next(self): self._n += 1; return self._n

    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("tunic")

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def validate_dataset_path(data_path: Path):
    if not data_path.exists():
        logger.error(f"Dataset path does not exist: {data_path}")
        sys.exit(1)
    if not (data_path / "train").exists() and not (data_path / "wds" / "train").exists():
        logger.error(f"Expected a 'train/' or 'wds/train/' subdirectory in {data_path}")
        sys.exit(1)


def _detect_format(data_root: str) -> str:
    """Return 'webdataset' or 'imagefolder' based on what's present at data_root."""
    if data_root.startswith("s3://"):
        return "webdataset"
    p = Path(data_root)
    if (p / "wds" / "dataset_info.json").exists():
        return "webdataset"
    return "imagefolder"


def _build_wds_loaders(data_root: str, batch_size: int, workers: int, seed: int,
                        train_tf, val_tf, collate_fn=None,
                        training_fraction: float = 1.0, val_fraction: float = 1.0):
    """Build train/val DataLoaders from WebDataset TAR shards (local or s3://)."""
    try:
        import webdataset as wds
    except ImportError:
        logger.error("webdataset not installed. Run: pip install webdataset")
        sys.exit(1)

    is_s3 = data_root.startswith("s3://")
    if is_s3:
        import subprocess
        info_url = data_root.rstrip("/") + "/wds/dataset_info.json"
        r = subprocess.run(["aws", "s3", "cp", info_url, "-"],
                           capture_output=True, text=True, check=True)
        meta = json.loads(r.stdout)
    else:
        with open(Path(data_root) / "wds" / "dataset_info.json") as f:
            meta = json.load(f)

    classes = meta["classes"]
    class_to_idx = {c: i for i, c in enumerate(classes)}
    num_classes = len(classes)

    def shard_urls(split):
        n = meta["splits"][split]["num_shards"]
        if is_s3:
            base = data_root.rstrip("/")
            return [f"pipe:aws s3 cp {base}/wds/{split}/shard-{i:06d}.tar -" for i in range(n)]
        d = Path(data_root) / "wds" / split
        return [str(d / f"shard-{i:06d}.tar") for i in range(n)]

    def decode_cls(b):
        return class_to_idx[b.decode().strip()]

    # webdataset >= 1.0 includes basichandlers in imagehandler, which maps
    # ".cls" -> int(data). Our .cls files store class name strings, so we
    # must intercept them before the built-in handler converts them to int.
    _img_decoder = wds.autodecode.imagehandler("pil")

    def _decoder(key, data):
        if key.endswith(".cls"):
            return data  # keep as raw bytes for decode_cls
        return _img_decoder(key, data)

    def make_loader(split, tf, is_train, fraction=1.0):
        urls = shard_urls(split)
        n_samples = meta["splits"][split]["num_samples"]
        n_take = max(batch_size, int(n_samples * fraction))
        if is_train:
            n_take = (n_take // batch_size) * batch_size  # trim to full batches

        # .slice() applies per worker, so divide n_take by the number of
        # active workers (shards are distributed round-robin, so workers
        # with no shards produce nothing; cap effective_workers by shard count).
        effective_workers = min(workers, len(urls)) if workers > 0 else 1
        slice_per_worker = max(batch_size, n_take // effective_workers)

        def apply_tf(img):
            return tf(img.convert("RGB"))

        dataset = (
            wds.WebDataset(urls, shardshuffle=is_train, seed=seed,
                           nodesplitter=wds.split_by_node, empty_check=False)
            .shuffle(1000 if is_train else 0)
            .decode(_decoder)
            .to_tuple("png", "cls")
            .map_tuple(apply_tf, decode_cls)
            .slice(slice_per_worker)
            .with_length(n_take)
        )
        return DataLoader(dataset, batch_size=batch_size, num_workers=workers,
                          pin_memory=False,
                          collate_fn=collate_fn if is_train else None)

    train_loader = make_loader("train", train_tf, is_train=True,  fraction=training_fraction)
    if "val" in meta.get("splits", {}):
        val_loader = make_loader("val",   val_tf,   is_train=False, fraction=val_fraction)
    else:
        logger.warning("No 'val' split in WebDataset — using 20%% of train as val")
        val_loader = make_loader("train", val_tf,   is_train=False, fraction=0.2)

    return train_loader, val_loader, num_classes


def build_transforms(img_size: int, randaug_magnitude: int = 0, randaug_num_ops: int = 2, is_train: bool = True):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if is_train:
        base = [
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
        ]
        if randaug_magnitude > 0:
            base.append(RandAugment(num_ops=randaug_num_ops, magnitude=randaug_magnitude))
        base += [transforms.ToTensor(), transforms.Normalize(mean, std)]
        return transforms.Compose(base)
    else:
        return transforms.Compose([
            transforms.Resize(int(img_size * 256 / 224)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


def make_stratified_split(dataset, val_fraction: float = 0.2, seed: int = 42):
    from collections import defaultdict
    class_to_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset.samples):
        class_to_indices[label].append(idx)

    train_indices, val_indices = [], []
    rng = random.Random(seed)
    for label, indices in class_to_indices.items():
        indices = list(indices)
        rng.shuffle(indices)
        split = max(1, int(len(indices) * val_fraction))
        val_indices.extend(indices[:split])
        train_indices.extend(indices[split:])

    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def _subsample(dataset, fraction: float, seed: int):
    """Return a Subset of dataset using a fixed fraction of its indices."""
    base = dataset.dataset if isinstance(dataset, Subset) else dataset
    indices = list(dataset.indices if isinstance(dataset, Subset) else range(len(dataset)))
    random.Random(seed).shuffle(indices)
    return Subset(base, indices[:max(1, int(len(indices) * fraction))])


def _build_loaders(data_path: Path, batch_size: int, workers: int, seed: int,
                   train_tf, val_tf, collate_fn=None,
                   training_fraction: float = 1.0, val_fraction: float = 1.0):
    """Build train/val DataLoaders, creating a stratified split if val/ is absent."""
    train_dir = data_path / "train"
    val_dir = data_path / "val"

    if val_dir.exists():
        train_dataset = datasets.ImageFolder(str(train_dir), transform=train_tf)
        val_dataset = datasets.ImageFolder(str(val_dir), transform=val_tf)
    else:
        logger.warning("No val/ directory found — creating a stratified 80/20 split from train/")
        base_dataset = datasets.ImageFolder(str(train_dir), transform=train_tf)
        train_subset, val_subset = make_stratified_split(base_dataset, val_fraction=0.2, seed=seed)
        train_dataset = train_subset
        val_base = datasets.ImageFolder(str(train_dir), transform=val_tf)
        val_dataset = Subset(val_base, val_subset.indices)

    if training_fraction < 1.0:
        n_before = len(train_dataset.indices if isinstance(train_dataset, Subset) else range(len(train_dataset)))
        train_dataset = _subsample(train_dataset, training_fraction, seed)
        logger.info(f"Using {len(train_dataset)}/{n_before} training samples (training_fraction={training_fraction})")

    if val_fraction < 1.0:
        n_before = len(val_dataset.indices if isinstance(val_dataset, Subset) else range(len(val_dataset)))
        val_dataset = _subsample(val_dataset, val_fraction, seed)
        logger.info(f"Using {len(val_dataset)}/{n_before} val samples (val_fraction={val_fraction})")

    num_classes = len(train_dataset.dataset.classes if isinstance(train_dataset, Subset) else train_dataset.classes)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=workers, pin_memory=True, drop_last=True,
                              collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=workers, pin_memory=True)
    return train_loader, val_loader, num_classes


class MixupCutmixCollator:
    def __init__(self, mixup_alpha: float, cutmix_alpha: float, num_classes: int):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.num_classes = num_classes

    def __call__(self, batch):
        images, labels = zip(*batch)
        images = torch.stack(images)
        labels = torch.tensor(labels, dtype=torch.long)

        if self.mixup_alpha > 0 and self.cutmix_alpha > 0:
            use_cutmix = random.random() > 0.5
        elif self.cutmix_alpha > 0:
            use_cutmix = True
        elif self.mixup_alpha > 0:
            use_cutmix = False
        else:
            return images, labels

        if use_cutmix:
            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            images, labels_a, labels_b = self._cutmix(images, labels, lam)
        else:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            idx = torch.randperm(images.size(0))
            images = lam * images + (1 - lam) * images[idx]
            labels_a = labels
            labels_b = labels[idx]

        labels_a_oh = nn.functional.one_hot(labels_a, self.num_classes).float()
        labels_b_oh = nn.functional.one_hot(labels_b, self.num_classes).float()
        mixed_labels = lam * labels_a_oh + (1 - lam) * labels_b_oh
        return images, mixed_labels

    def _cutmix(self, images, labels, lam):
        _, _, H, W = images.shape
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        x1 = np.clip(cx - cut_w // 2, 0, W)
        x2 = np.clip(cx + cut_w // 2, 0, W)
        y1 = np.clip(cy - cut_h // 2, 0, H)
        y2 = np.clip(cy + cut_h // 2, 0, H)
        idx = torch.randperm(images.size(0))
        images = images.clone()
        images[:, :, y1:y2, x1:x2] = images[idx, :, y1:y2, x1:x2]
        lam = 1 - (x2 - x1) * (y2 - y1) / (W * H)
        return images, labels, labels[idx]


def create_model(model_name: str, num_classes: int, pretrained: bool, drop_rate: float) -> nn.Module:
    try:
        model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes, drop_rate=drop_rate)
    except Exception as e:
        logger.error(f"Failed to create model '{model_name}': {e}")
        logger.error("Common alternatives: resnet50, efficientnet_b0, convnext_tiny, vit_small_patch16_224, mobilenetv3_large_100")
        sys.exit(1)
    return model


def freeze_backbone(model: nn.Module):
    head_keywords = {"head", "fc", "classifier"}
    for name, param in model.named_parameters():
        top = name.split(".")[0]
        if top not in head_keywords and not any(kw in name for kw in head_keywords):
            param.requires_grad = False


def unfreeze_all(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = True


def get_optimizer(model: nn.Module, optimizer_name: str, lr: float, weight_decay: float):
    params = filter(lambda p: p.requires_grad, model.parameters())
    if optimizer_name == "AdamW":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    else:
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9)


def build_scheduler(optimizer, epochs: int, steps_per_epoch: int, warmup_epochs: int = 5):
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, loader, optimizer, scheduler, criterion, device, use_soft_labels, trial_id="", epoch=0, epochs=0):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    epoch_str = f" epoch {epoch+1}/{epochs}" if epochs else ""
    desc = f"trial {trial_id}{epoch_str}" if trial_id else f"train{epoch_str}"
    bar = tqdm(loader, leave=False, desc=desc,
               bar_format="{l_bar}{bar}| batch {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]")
    for images, labels in bar:
        images = images.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        if use_soft_labels and labels.dim() == 2:
            labels = labels.to(device)
            loss = -(labels * nn.functional.log_softmax(outputs, dim=-1)).sum(dim=-1).mean()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels.argmax(dim=1)).sum().item()
        else:
            labels = labels.to(device)
            loss = criterion(outputs, labels)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * images.size(0)
        total += images.size(0)

        if hasattr(bar, "set_postfix"):
            bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.4f}")

    return total_loss / total, correct / total


def _compute_auroc(probs: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    present = np.unique(labels)
    try:
        if probs.shape[1] == 2:
            return roc_auc_score(labels, probs[:, 1])
        if len(present) < 2:
            return float("nan")
        return roc_auc_score(labels, probs[:, present], multi_class="ovr", average="macro", labels=present)
    except ValueError:
        return float("nan")


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += images.size(0)
            all_probs.append(torch.softmax(outputs, dim=1).cpu())
            all_labels.append(labels.cpu())

    probs = torch.cat(all_probs).numpy()
    labels_np = torch.cat(all_labels).numpy()
    return total_loss / total, correct / total, _compute_auroc(probs, labels_np)


def _evaluate_distributed(model, loader, criterion, device, world_size: int):
    """Like evaluate(), but reduces loss/accuracy across Ray Train workers."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += images.size(0)
            all_probs.append(torch.softmax(outputs, dim=1).cpu())
            all_labels.append(labels.cpu())

    if world_size > 1:
        import torch.distributed as dist
        stats = torch.tensor([correct, total, total_loss], dtype=torch.float64, device=device)
        dist.all_reduce(stats)
        accuracy = stats[0].item() / stats[1].item()
        avg_loss = stats[2].item() / stats[1].item()
    else:
        accuracy = correct / total
        avg_loss = total_loss / total

    probs = torch.cat(all_probs).numpy()
    labels_np = torch.cat(all_labels).numpy()
    return avg_loss, accuracy, _compute_auroc(probs, labels_np)


def load_search_space_overrides(path: str) -> dict:
    if yaml is None:
        logger.error("PyYAML is required for --search-space. Install with: pip install pyyaml")
        sys.exit(1)
    with open(path) as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------------------
# Ray Train worker function
# ---------------------------------------------------------------------------

def train_func_distributed(config: dict):
    """Ray Train worker function. Receives all hyperparams + fixed config as a dict."""
    data_path = Path(config["data"])
    model_name = config["model"]
    pretrained = config["pretrained"]
    epochs = config["epochs"]
    batch_size = config["batch_size"]
    img_size = config["img_size"]
    freeze_backbone_epochs = config["freeze_backbone"]
    base_seed = config["seed"]
    dataloader_workers = config["dataloader_workers"]
    training_fraction = config["training_fraction"]
    val_fraction = config["val_fraction"]
    num_classes = config["num_classes"]
    lr = config["lr"]
    weight_decay = config["weight_decay"]
    label_smoothing = config["label_smoothing"]
    drop_rate = config["drop_rate"]
    randaug_magnitude = config["randaugment_magnitude"]
    randaug_num_ops = config["randaugment_num_ops"]
    mixup_alpha = config["mixup_alpha"]
    cutmix_alpha = config["cutmix_alpha"]
    optimizer_name = config["optimizer"]

    rank = ray.train.get_context().get_world_rank()
    world_size = ray.train.get_context().get_world_size()
    if ray.train.get_context().get_world_rank() == 0:
        _counter = ray.get_actor("trial_counter")
        _trial_num = ray.get(_counter.next.remote())
    else:
        _trial_num = 0
    trial_id = f"{_trial_num}/{config['n_trials']}" if _trial_num else ""
    set_seed(base_seed + rank)
    device = ray.train.torch.get_device()

    use_mixup_cutmix = mixup_alpha > 0 or cutmix_alpha > 0
    collate_fn = MixupCutmixCollator(mixup_alpha, cutmix_alpha, num_classes) if use_mixup_cutmix else None

    train_tf = build_transforms(img_size, randaug_magnitude, randaug_num_ops, is_train=True)
    val_tf = build_transforms(img_size, is_train=False)
    train_loader, val_loader, _ = _build_loaders(
        data_path, batch_size, dataloader_workers, base_seed,
        train_tf, val_tf, collate_fn,
        training_fraction=training_fraction, val_fraction=val_fraction,
    )
    train_loader = ray.train.torch.prepare_data_loader(train_loader)
    val_loader = ray.train.torch.prepare_data_loader(val_loader)

    try:
        model = create_model(model_name, num_classes, pretrained, drop_rate)
        model = ray.train.torch.prepare_model(model)

        if freeze_backbone_epochs > 0:
            freeze_backbone(model)

        optimizer = get_optimizer(model, optimizer_name, lr, weight_decay)
        scheduler = build_scheduler(optimizer, epochs, len(train_loader))
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        val_criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            if freeze_backbone_epochs > 0 and epoch == freeze_backbone_epochs:
                unfreeze_all(model)
                optimizer = get_optimizer(model, optimizer_name, lr, weight_decay)

            if world_size > 1:
                train_loader.sampler.set_epoch(epoch)

            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, scheduler, criterion, device,
                use_soft_labels=use_mixup_cutmix, trial_id=trial_id, epoch=epoch, epochs=epochs,
            )
            val_loss, val_acc, val_auroc = _evaluate_distributed(
                model, val_loader, val_criterion, device, world_size,
            )

            if rank == 0:
                logger.info(
                    f"epoch {epoch+1}/{epochs} | loss={train_loss:.4f} acc={train_acc:.4f} | "
                    f"val_acc={val_acc:.4f} val_auroc={val_auroc:.4f}"
                )

            ray.train.report({"val_acc": val_acc, "val_auroc": val_auroc, "train_loss": train_loss})

    except torch.cuda.OutOfMemoryError:
        logger.warning("CUDA OOM — reporting 0.0")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        ray.train.report({"val_acc": 0.0, "val_auroc": float("nan"), "train_loss": float("inf")})


# ---------------------------------------------------------------------------
# Ray Tune plain-function trainable (single GPU/CPU per trial, no DDP)
# ---------------------------------------------------------------------------

def _tune_trial(config: dict):
    """Plain Ray Tune trainable. Each trial runs on one GPU (or CPU)."""
    data_root   = config["data"]
    data_format = config.get("data_format", "imagefolder")
    data_path   = Path(data_root) if data_format == "imagefolder" else None
    device = get_device(config["device"])
    _counter = ray.get_actor("trial_counter")
    _trial_num = ray.get(_counter.next.remote())
    trial_id = f"{_trial_num}/{config['n_trials']}"
    set_seed(config["seed"])

    epochs = config["epochs"]
    lr = config["lr"]
    weight_decay = config["weight_decay"]
    label_smoothing = config["label_smoothing"]
    drop_rate = config["drop_rate"]
    randaug_magnitude = config["randaugment_magnitude"]
    randaug_num_ops = config["randaugment_num_ops"]
    mixup_alpha = config["mixup_alpha"]
    cutmix_alpha = config["cutmix_alpha"]
    optimizer_name = config["optimizer"]
    num_classes = config["num_classes"]

    use_mixup_cutmix = mixup_alpha > 0 or cutmix_alpha > 0
    collate_fn = MixupCutmixCollator(mixup_alpha, cutmix_alpha, num_classes) if use_mixup_cutmix else None

    train_tf = build_transforms(config["img_size"], randaug_magnitude, randaug_num_ops, is_train=True)
    val_tf = build_transforms(config["img_size"], is_train=False)
    bs, workers, seed = config["batch_size"], config["dataloader_workers"], config["seed"]
    tf = config["training_fraction"]
    vf = config["val_fraction"]
    if data_format == "webdataset":
        train_loader, val_loader, _ = _build_wds_loaders(
            data_root, bs, workers, seed, train_tf, val_tf, collate_fn, tf, vf)
    else:
        train_loader, val_loader, _ = _build_loaders(
            data_path, bs, workers, seed, train_tf, val_tf, collate_fn, tf, vf)

    try:
        model = create_model(config["model"], num_classes, config["pretrained"], drop_rate)
        model = model.to(device)

        if config["freeze_backbone"] > 0:
            freeze_backbone(model)

        optimizer = get_optimizer(model, optimizer_name, lr, weight_decay)
        scheduler = build_scheduler(optimizer, epochs, len(train_loader))
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        val_criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            if config["freeze_backbone"] > 0 and epoch == config["freeze_backbone"]:
                unfreeze_all(model)
                optimizer = get_optimizer(model, optimizer_name, lr, weight_decay)

            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, scheduler, criterion, device,
                use_soft_labels=use_mixup_cutmix, trial_id=trial_id, epoch=epoch, epochs=epochs,
            )
            _, val_acc, val_auroc = evaluate(model, val_loader, val_criterion, device)

            tune.report({"val_acc": val_acc, "val_auroc": val_auroc, "train_loss": train_loss})

    except torch.cuda.OutOfMemoryError:
        logger.warning("CUDA OOM — reporting 0.0")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        tune.report({"val_acc": 0.0, "val_auroc": float("nan"), "train_loss": float("inf")})


# ---------------------------------------------------------------------------
# Final training
# ---------------------------------------------------------------------------

def _build_combined_loader(data_root, fmt, batch_size, workers, seed, train_tf, collate_fn):
    """Build a DataLoader combining train+val splits. Exits if val is missing."""
    if fmt == "webdataset":
        try:
            import webdataset as wds
        except ImportError:
            logger.error("webdataset not installed")
            sys.exit(1)
        is_s3 = data_root.startswith("s3://")
        if is_s3:
            import subprocess
            info_url = data_root.rstrip("/") + "/wds/dataset_info.json"
            r = subprocess.run(["aws", "s3", "cp", info_url, "-"],
                               capture_output=True, text=True, check=True)
            meta = json.loads(r.stdout)
        else:
            with open(Path(data_root) / "wds" / "dataset_info.json") as f:
                meta = json.load(f)
        if "val" not in meta.get("splits", {}):
            logger.error("--combine specified but no val split found in dataset_info.json")
            sys.exit(1)
        classes = meta["classes"]
        class_to_idx = {c: i for i, c in enumerate(classes)}
        num_classes = len(classes)

        def shard_urls(split):
            n = meta["splits"][split]["num_shards"]
            if is_s3:
                base = data_root.rstrip("/")
                return [f"pipe:aws s3 cp {base}/wds/{split}/shard-{i:06d}.tar -" for i in range(n)]
            d = Path(data_root) / "wds" / split
            return [str(d / f"shard-{i:06d}.tar") for i in range(n)]

        urls = shard_urls("train") + shard_urls("val")
        n_train = meta["splits"]["train"]["num_samples"]
        n_val = meta["splits"]["val"]["num_samples"]
        n_total = n_train + n_val if n_train > 0 and n_val > 0 else len(urls) * 5000

        _img_decoder = wds.autodecode.imagehandler("pil")
        def _decoder(key, data):
            if key.endswith(".cls"):
                return data
            return _img_decoder(key, data)
        def decode_cls(b):
            return class_to_idx[b.decode().strip()]
        def apply_tf(img):
            return train_tf(img.convert("RGB"))

        effective_workers = min(workers, len(urls)) if workers > 0 else 1
        slice_per_worker = max(batch_size, n_total // effective_workers)
        dataset = (
            wds.WebDataset(urls, shardshuffle=True, seed=seed,
                           nodesplitter=wds.split_by_node, empty_check=False)
            .shuffle(1000)
            .decode(_decoder)
            .to_tuple("png", "cls")
            .map_tuple(apply_tf, decode_cls)
            .slice(slice_per_worker)
            .with_length(n_total)
        )
        return DataLoader(dataset, batch_size=batch_size, num_workers=workers,
                          pin_memory=False, collate_fn=collate_fn), num_classes
    else:
        train_dir = Path(data_root) / "train"
        val_dir = Path(data_root) / "val"
        if not val_dir.exists():
            logger.error("--combine specified but no val/ directory found")
            sys.exit(1)
        from torch.utils.data import ConcatDataset
        train_ds = datasets.ImageFolder(str(train_dir), transform=train_tf)
        val_ds = datasets.ImageFolder(str(val_dir), transform=train_tf)
        num_classes = len(train_ds.classes)
        combined = ConcatDataset([train_ds, val_ds])
        return DataLoader(combined, batch_size=batch_size, shuffle=True,
                          num_workers=workers, pin_memory=True, drop_last=True,
                          collate_fn=collate_fn), num_classes


def _build_test_loader(data_root, fmt, batch_size, workers, seed, val_tf):
    """Build a test DataLoader (WDS or imagefolder). Returns None if no test split exists."""
    if fmt == "webdataset":
        try:
            import webdataset as wds
        except ImportError:
            logger.error("webdataset not installed")
            return None
        is_s3 = data_root.startswith("s3://")
        if is_s3:
            import subprocess
            info_url = data_root.rstrip("/") + "/wds/dataset_info.json"
            r = subprocess.run(["aws", "s3", "cp", info_url, "-"],
                               capture_output=True, text=True, check=True)
            meta = json.loads(r.stdout)
        else:
            with open(Path(data_root) / "wds" / "dataset_info.json") as f:
                meta = json.load(f)
        if "test" not in meta.get("splits", {}):
            return None
        classes = meta["classes"]
        class_to_idx = {c: i for i, c in enumerate(classes)}
        n_shards = meta["splits"]["test"]["num_shards"]
        n_samples = meta["splits"]["test"]["num_samples"]
        if is_s3:
            base = data_root.rstrip("/")
            urls = [f"pipe:aws s3 cp {base}/wds/test/shard-{i:06d}.tar -" for i in range(n_shards)]
        else:
            d = Path(data_root) / "wds" / "test"
            urls = [str(d / f"shard-{i:06d}.tar") for i in range(n_shards)]
        _img_decoder = wds.autodecode.imagehandler("pil")
        def _decoder(key, data):
            if key.endswith(".cls"):
                return data
            return _img_decoder(key, data)
        def decode_cls(b):
            return class_to_idx[b.decode().strip()]
        def apply_tf(img):
            return val_tf(img.convert("RGB"))
        effective_workers = min(workers, len(urls)) if workers > 0 else 1
        if n_samples > 0:
            slice_per_worker = max(batch_size, n_samples // effective_workers)
            length = n_samples
        else:
            slice_per_worker = 10 ** 9  # no limit
            length = n_shards * 5000    # rough estimate
        dataset = (
            wds.WebDataset(urls, shardshuffle=False, seed=seed,
                           nodesplitter=wds.split_by_node, empty_check=False)
            .decode(_decoder)
            .to_tuple("png", "cls")
            .map_tuple(apply_tf, decode_cls)
            .slice(slice_per_worker)
            .with_length(length)
        )
        return DataLoader(dataset, batch_size=batch_size, num_workers=workers, pin_memory=False)
    else:
        test_dir = Path(data_root) / "test"
        if not test_dir.exists():
            return None
        test_dataset = datasets.ImageFolder(str(test_dir), transform=val_tf)
        return DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                          num_workers=workers, pin_memory=True)


def run_final(args):
    final_json = Path(args.final)
    try:
        with open(final_json) as f:
            results = json.load(f)
    except FileNotFoundError:
        logger.error(f"--final path does not exist: {final_json}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.error(f"Malformed JSON in {final_json}: {e}")
        sys.exit(1)

    if "best_params" not in results:
        logger.error(f"No 'best_params' key in {final_json}")
        sys.exit(1)

    params = results["best_params"]
    model_name = args.model or results.get("model", "resnet50")
    num_classes = results.get("num_classes")
    data_root = args.data if args.data else results.get("dataset", ".")
    data_path = Path(data_root)
    epochs = args.final_epochs or results.get("epochs", 30)

    validate_dataset_path(data_path)

    device = get_device(args.device)
    set_seed(args.seed)

    use_mixup_cutmix = params.get("mixup_alpha", 0) > 0 or params.get("cutmix_alpha", 0) > 0
    collate_fn = MixupCutmixCollator(params.get("mixup_alpha", 0), params.get("cutmix_alpha", 0), num_classes) if use_mixup_cutmix else None

    train_tf = build_transforms(args.img_size, params.get("randaugment_magnitude", 0),
                                params.get("randaugment_num_ops", 2), is_train=True)
    val_tf = build_transforms(args.img_size, is_train=False)

    fmt = _detect_format(data_root)
    if args.combine:
        train_loader, inferred_classes = _build_combined_loader(
            data_root, fmt, args.batch_size, args.workers, args.seed, train_tf, collate_fn)
        val_loader = None
    elif fmt == "webdataset":
        train_loader, val_loader, inferred_classes = _build_wds_loaders(
            data_root, args.batch_size, args.workers, args.seed,
            train_tf, val_tf, collate_fn,
            training_fraction=args.training_fraction, val_fraction=args.val_fraction)
    else:
        train_loader, val_loader, inferred_classes = _build_loaders(
            data_path, args.batch_size, args.workers, args.seed,
            train_tf, val_tf, collate_fn,
            training_fraction=args.training_fraction, val_fraction=args.val_fraction)
    if num_classes is None:
        num_classes = inferred_classes

    model = create_model(model_name, num_classes, args.pretrained, params.get("drop_rate", 0.0))
    model = model.to(device)

    if args.freeze_backbone > 0:
        freeze_backbone(model)

    optimizer = get_optimizer(model, params.get("optimizer", "AdamW"), params["lr"], params.get("weight_decay", 1e-4))
    scheduler = build_scheduler(optimizer, epochs, len(train_loader))
    criterion = nn.CrossEntropyLoss(label_smoothing=params.get("label_smoothing", 0.0))
    val_criterion = nn.CrossEntropyLoss()

    best_val_auroc = 0.0
    best_val_acc = 0.0
    best_epoch = 0
    best_state = None

    for epoch in range(epochs):
        if args.freeze_backbone > 0 and epoch == args.freeze_backbone:
            unfreeze_all(model)
            optimizer = get_optimizer(model, params.get("optimizer", "AdamW"), params["lr"], params.get("weight_decay", 1e-4))

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device,
            use_soft_labels=use_mixup_cutmix, epoch=epoch, epochs=epochs
        )

        if val_loader is not None:
            _, val_acc, val_auroc = evaluate(model, val_loader, val_criterion, device)
            logger.info(f"Epoch {epoch+1}/{epochs} — train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f} val_auroc={val_auroc:.4f}")
            if val_auroc > best_val_auroc:
                best_val_auroc = val_auroc
                best_val_acc = val_acc
                best_epoch = epoch + 1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            logger.info(f"Epoch {epoch+1}/{epochs} — train_loss={train_loss:.4f} train_acc={train_acc:.4f}")
            best_epoch = epoch + 1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    checkpoint_path = "tunic_final.pt"
    torch.save({
        "model_state_dict": best_state,
        "best_val_acc": best_val_acc,
        "best_val_auroc": best_val_auroc,
        "epoch": best_epoch,
        "params": params,
        "model_name": model_name,
        "num_classes": num_classes,
    }, checkpoint_path)

    print(f"\nFinal training complete.")
    if val_loader is not None:
        print(f"  Best val accuracy: {best_val_acc:.4f}  val AUROC: {best_val_auroc:.4f} (epoch {best_epoch})")
    else:
        print(f"  Trained on train+val combined for {best_epoch} epochs (no val tracking)")
    print(f"  Checkpoint saved to: {checkpoint_path}")

    test_loader = _build_test_loader(data_root, fmt, args.batch_size, args.workers, args.seed, val_tf)
    if test_loader is not None:
        model.load_state_dict(best_state)
        model.to(device)
        _, test_acc, test_auroc = evaluate(model, test_loader, val_criterion, device)
        print(f"  Test accuracy:      {test_acc:.4f}  test AUROC: {test_auroc:.4f}")
    else:
        print(f"  No test split found — skipping test evaluation.")


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def run_smoke_test(args):
    import tempfile
    from PIL import Image

    logger.info("Running smoke test...")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        for split in ["train", "val"]:
            for cls in ["cat", "dog", "bird"]:
                cls_dir = tmpdir / split / cls
                cls_dir.mkdir(parents=True)
                for i in range(10):
                    img = Image.fromarray(
                        np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
                    )
                    img.save(cls_dir / f"img_{i}.jpg")

        smoke_args = argparse.Namespace(
            data=str(tmpdir),
            model="resnet18",
            pretrained=False,
            n_trials=2,
            epochs=2,
            batch_size=4,
            output=str(tmpdir / "smoke_results.json"),
            seed=0,
            device=args.device,
            workers=0,
            img_size=64,
            freeze_backbone=0,
            training_fraction=1.0,
            val_fraction=1.0,
            resume=None,
            search_space=None,
            final=None,
            final_epochs=None,
            smoke_test=False,
            num_train_workers=1,
        )

        run_tuning(smoke_args)

        # Final mode
        smoke_final_args = argparse.Namespace(**vars(smoke_args))
        smoke_final_args.final = smoke_args.output
        smoke_final_args.final_epochs = 2
        run_final(smoke_final_args)

    logger.info("Smoke test passed.")
    sys.exit(0)


# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------

def run_tuning(args):
    if not RAY_AVAILABLE:
        logger.error("Ray is not installed. Install with: pip install 'ray[tune,train]' optuna")
        sys.exit(1)

    if not ray.is_initialized():
        address = args.ray_address or None
        ray.init(address=address, ignore_reinit_error=True)
        logger.info(f"Ray initialized (address={address or 'local'})")

    data_root   = args.data
    data_format = _detect_format(data_root)
    set_seed(args.seed)

    if data_format == "webdataset":
        if not data_root.startswith("s3://"):
            validate_dataset_path(Path(data_root))
        with open(Path(data_root) / "wds" / "dataset_info.json") as f:
            wds_meta = json.load(f)
        num_classes = len(wds_meta["classes"])
        data_key = data_root
    else:
        data_path = Path(data_root)
        validate_dataset_path(data_path)
        tmp_ds = datasets.ImageFolder(str(data_path / "train"))
        num_classes = len(tmp_ds.classes)
        data_key = str(data_path.resolve())

    logger.info(f"Dataset: {data_root} | Format: {data_format} | Classes: {num_classes} | Model: {args.model}")

    ss = {}
    if args.search_space:
        ss = load_search_space_overrides(args.search_space)

    hp_keys = ["lr", "weight_decay", "label_smoothing", "drop_rate",
               "randaugment_magnitude", "randaugment_num_ops",
               "mixup_alpha", "cutmix_alpha", "optimizer"]

    use_gpu = args.device != "cpu"

    search_space = {
        "data": data_key,
        "data_format": data_format,
        "model": args.model,
        "pretrained": args.pretrained,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "img_size": args.img_size,
        "freeze_backbone": args.freeze_backbone,
        "seed": args.seed,
        "dataloader_workers": args.workers,
        "training_fraction": args.training_fraction,
        "val_fraction": args.val_fraction,
        "num_classes": num_classes,
        "device": args.device,
        "n_trials": args.n_trials,
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
        _tune_trial,
        resources={"GPU": 1 if use_gpu else 0, "CPU": max(1, args.workers)},
    )

    search_alg = OptunaSearch(metric=args.tune_metric, mode="max", seed=args.seed)
    scheduler = ASHAScheduler(
        metric=args.tune_metric,
        mode="max",
        max_t=args.epochs,
        grace_period=max(1, args.epochs // 5),
    )

    if args.ray_storage:
        storage_path = args.ray_storage
    else:
        storage_path = str(Path(args.output).parent.resolve() / "ray_results")

    run_config = RunConfig(
        storage_path=storage_path,
        name="tunic_study",
    )

    if args.resume:
        resume_path = str(Path(args.resume).resolve())
        logger.info(f"Loading previous results from {resume_path} to warm-start search")
        prior_grid = tune.ExperimentAnalysis(resume_path)
        points_to_evaluate, evaluated_rewards = [], []
        for trial in prior_grid.trials:
            if trial.last_result and trial.status == "TERMINATED":
                cfg = {k: trial.config[k] for k in hp_keys if k in trial.config}
                acc = trial.last_result.get("val_acc")
                if cfg and acc is not None:
                    points_to_evaluate.append(cfg)
                    evaluated_rewards.append(acc)
        logger.info(f"Warm-starting from {len(points_to_evaluate)} prior trials, running {args.n_trials} new trials")
        search_alg = OptunaSearch(
            metric="val_acc", mode="max", seed=args.seed,
            points_to_evaluate=points_to_evaluate,
            evaluated_rewards=evaluated_rewards,
        )

    TrialCounter.options(name="trial_counter", lifetime="detached", get_if_exists=True).remote()

    tuner = tune.Tuner(
        trainable,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            search_alg=search_alg,
            scheduler=scheduler,
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
    best_val_acc = best.metrics["val_acc"]
    best_val_auroc = best.metrics.get("val_auroc", float("nan"))
    best_params = {k: best.config[k] for k in hp_keys if k in best.config}

    all_trials = []
    completed = 0
    errored = 0
    for r in results:
        state = "ERROR" if r.error else "COMPLETE"
        if r.error:
            errored += 1
        else:
            completed += 1
        all_trials.append({
            "val_acc": r.metrics.get("val_acc") if r.metrics else None,
            "val_auroc": r.metrics.get("val_auroc") if r.metrics else None,
            "params": {k: r.config[k] for k in hp_keys if k in r.config} if r.config else {},
            "state": state,
        })

    output = {
        "best_val_acc": best_val_acc,
        "best_val_auroc": best_val_auroc,
        "best_params": best_params,
        "model": args.model,
        "dataset": data_key,
        "num_classes": num_classes,
        "n_trials": args.n_trials,
        "epochs": args.epochs,
        "completed_trials": completed,
        "errored_trials": errored,
        "total_time_seconds": total_time,
        "all_trials": all_trials,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nBest val accuracy: {best_val_acc:.4f}  AUROC: {best_val_auroc:.4f}")
    print("Best params:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    print(f"\nResults saved to: {args.output}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="tunic — hyperparameter tuning for image classifiers",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data", type=str, help="Path to dataset root (ImageFolder layout)")
    p.add_argument("--model", type=str, default="resnet50",
                   help="Any timm model name (e.g. resnet50, efficientnet_b0, convnext_tiny)")
    p.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True,
                   help="Use timm pretrained weights")
    p.add_argument("--n_trials", type=int, default=80, dest="n_trials",
                   help="Number of Optuna trials")
    p.add_argument("--epochs", type=int, default=30,
                   help="Training epochs per trial")
    p.add_argument("--batch-size", type=int, default=32, dest="batch_size",
                   help="Batch size (fixed across trials)")
    p.add_argument("--output", type=str, default="tunic_results.json",
                   help="Path for output JSON with best hyperparameters (default: tunic_results.json)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility")
    p.add_argument("--device", type=str, default="auto",
                   help="Device to use: auto detects CUDA/MPS/CPU, or specify explicitly")
    p.add_argument("--workers", type=int, default=4,
                   help="DataLoader worker count")
    p.add_argument("--img-size", type=int, default=224, dest="img_size",
                   help="Input image resolution")
    p.add_argument("--training_fraction", type=float, default=1.0,
                   help="Fraction of training data to use (e.g. 0.1 for 10%%); same subset across all trials")
    p.add_argument("--val_fraction", type=float, default=None,
                   help="Fraction of val data to use (defaults to --training_fraction)")
    p.add_argument("--freeze-backbone", type=int, default=0, dest="freeze_backbone",
                   help="Epochs to freeze backbone; 0 = no freeze")
    p.add_argument("--final", type=str, default=None,
                   help="Path to tunic_results.json — skip tuning, train final model")
    p.add_argument("--final-epochs", type=int, default=None, dest="final_epochs",
                   help="Override epoch count for final training run (defaults to tuning epochs)")
    p.add_argument("--combine", action="store_true", default=False,
                   help="Train final model on train+val combined (requires --final; exits if no val split)")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to a previous Ray Tune experiment directory; warm-starts Optuna search from those results and runs --n_trials new trials")
    p.add_argument("--search-space", type=str, default=None, dest="search_space",
                   help="YAML file to override search space ranges")
    p.add_argument("--smoke-test", action="store_true", dest="smoke_test",
                   help="Run end-to-end smoke test with synthetic data")
    p.add_argument("--num-train-workers", type=int, default=1, dest="num_train_workers",
                   help="Ray Train workers per trial (= GPUs per trial)")
    p.add_argument("--ray-address", type=str, default=None, dest="ray_address",
                   help="Ray cluster address to connect to (e.g. localhost:6385); defaults to local")
    p.add_argument("--ray-storage", type=str, default=None, dest="ray_storage",
                   help="Ray Tune storage path (local dir or S3 URI, e.g. s3://bucket/ray-results)")
    p.add_argument("--tune-metric", type=str, default="val_auroc", dest="tune_metric",
                   help="Metric used by Optuna and ASHA for trial selection and pruning (default: val_auroc)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.val_fraction is None:
        args.val_fraction = 1.0

    if args.smoke_test:
        run_smoke_test(args)
        return

    if args.final:
        run_final(args)
        return

    if not args.data:
        logger.error("--data is required for tuning mode")
        sys.exit(1)

    t0 = time.time()
    run_tuning(args)
    elapsed = time.time() - t0
    if elapsed < 600:
        logger.info(f"Elapsed time: {elapsed:.1f}s")
    else:
        logger.info(f"Elapsed time: {elapsed/60:.1f}m")


if __name__ == "__main__":
    main()
