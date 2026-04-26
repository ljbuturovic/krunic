"""common_krunic.py — Shared utilities for tunic.py and cvic.py."""

import logging
import random
import sys
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
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

logger = logging.getLogger("krunic")


# ---------------------------------------------------------------------------
# Seeds / device
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_amp_dtype() -> torch.dtype:
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def validate_dataset_path(data_path: Path):
    if not data_path.exists():
        logger.error(f"Dataset path does not exist: {data_path}")
        sys.exit(1)
    if not (data_path / "train").exists() and not (data_path / "wds" / "train").exists():
        logger.error(f"Expected a 'train/' or 'wds/train/' subdirectory in {data_path}")
        sys.exit(1)


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


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def build_transforms(img_size: int, randaug_magnitude: int = 0, randaug_num_ops: int = 2, is_train: bool = True):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
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


# ---------------------------------------------------------------------------
# Mixup / CutMix
# ---------------------------------------------------------------------------

class MixupCutmixCollator:
    def __init__(self, mixup_alpha: float, cutmix_alpha: float, num_classes: int):
        self.mixup_alpha  = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.num_classes  = num_classes

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
            images   = lam * images + (1 - lam) * images[idx]
            labels_a = labels
            labels_b = labels[idx]

        labels_a_oh  = nn.functional.one_hot(labels_a, self.num_classes).float()
        labels_b_oh  = nn.functional.one_hot(labels_b, self.num_classes).float()
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


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def create_model(model_name: str, num_classes: int, pretrained: bool, drop_rate: float) -> nn.Module:
    try:
        model = timm.create_model(model_name, pretrained=pretrained,
                                  num_classes=num_classes, drop_rate=drop_rate)
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


# ---------------------------------------------------------------------------
# Optimizer / scheduler
# ---------------------------------------------------------------------------

def get_optimizer(model: nn.Module, optimizer_name: str, lr: float, weight_decay: float):
    params = filter(lambda p: p.requires_grad, model.parameters())
    if optimizer_name == "AdamW":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    else:
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9)


def build_scheduler(optimizer, epochs: int, steps_per_epoch: int, warmup_epochs: int = 5):
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps  = epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Training / evaluation primitives
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, scheduler, criterion, device,
                    use_soft_labels, trial_id="", epoch=0, epochs=0,
                    use_amp=False, show_progress=True):
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0
    use_amp    = use_amp and device.type == "cuda"
    amp_dtype  = get_amp_dtype()
    scaler     = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    epoch_str = f" epoch {epoch+1}/{epochs}" if epochs else ""
    desc = f"trial {trial_id}{epoch_str}" if trial_id else f"train{epoch_str}"
    bar = tqdm(loader, leave=False, desc=desc, disable=not show_progress,
               bar_format="{l_bar}{bar}| batch {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]")

    for images, labels in bar:
        images = images.to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            outputs = model(images)
            if use_soft_labels and labels.dim() == 2:
                labels = labels.to(device)
                loss = -(labels * nn.functional.log_softmax(outputs, dim=-1)).sum(dim=-1).mean()
            else:
                labels = labels.to(device)
                loss = criterion(outputs, labels)

        preds = outputs.argmax(dim=1)
        if use_soft_labels and labels.dim() == 2:
            correct += (preds == labels.argmax(dim=1)).sum().item()
        else:
            correct += (preds == labels).sum().item()

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item() * images.size(0)
        total      += images.size(0)

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
        return roc_auc_score(labels, probs[:, present], multi_class="ovr",
                             average="macro", labels=present)
    except ValueError:
        return float("nan")


# ---------------------------------------------------------------------------
# Search space overrides
# ---------------------------------------------------------------------------

def load_search_space_overrides(path: str) -> dict:
    if yaml is None:
        logger.error("PyYAML is required for --search-space. Install with: pip install pyyaml")
        sys.exit(1)
    with open(path) as f:
        return yaml.safe_load(f) or {}
