#!/usr/bin/env python3
"""
TinyViT fine-tuning on CIFAR-10.
This script mirrors train_resnet_cifar.py but tailors defaults for
tiny_vit_5m_224 to compare transformer vs CNN baselines.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

try:
    import timm
    from timm.data import create_transform
except ImportError as err:
    raise SystemExit(
        "timm is required for this script. Install it with `pip install timm`."
    ) from err

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class TrainStats:
    loss: float
    acc: float
    samples_per_sec: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TinyViT CIFAR-10 training script")
    parser.add_argument(
        "--model",
        default="tiny_vit_5m_224.dist_in22k_ft_in1k",
        help="timm model name",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        action="store_true",
        help="load pretrained weights",
    )
    parser.add_argument(
        "--no-pretrained",
        dest="pretrained",
        action="store_false",
        help="disable pretrained weights",
    )
    parser.set_defaults(pretrained=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument(
        "--optimizer",
        choices=["adamw", "sgd"],
        default="adamw",
        help="optimizer choice",
    )
    parser.add_argument(
        "--lr-scheduler",
        choices=["cosine", "none"],
        default="cosine",
        help="learning rate scheduler",
    )
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--interpolation", default="bicubic")
    parser.add_argument("--auto-augment", default="rand-m9-mstd0.5-inc1")
    parser.add_argument("--reprob", type=float, default=0.25)
    parser.add_argument("--remode", default="pixel")
    parser.add_argument("--recount", type=int, default=1)
    parser.add_argument("--mixup", type=float, default=0.0, help="mixup alpha (0 to disable)")
    parser.add_argument("--cutmix", type=float, default=0.0, help="cutmix alpha (0 to disable)")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="runs/tinyvit_ft")
    parser.add_argument(
        "--no-amp",
        dest="use_amp",
        action="store_false",
        help="disable AMP",
    )
    parser.add_argument(
        "--amp",
        dest="use_amp",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(use_amp=True)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transforms(args: argparse.Namespace):
    common = dict(
        input_size=(3, args.img_size, args.img_size),
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        interpolation=args.interpolation,
    )
    train_transform = create_transform(
        is_training=True,
        auto_augment=args.auto_augment if args.auto_augment != "none" else None,
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
        **common,
    )
    eval_transform = create_transform(is_training=False, **common)
    return train_transform, eval_transform


def build_datasets(
    args: argparse.Namespace, train_transform, eval_transform
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset]:
    train_dataset = datasets.CIFAR10(
        root=args.data_dir, train=True, transform=train_transform, download=True
    )
    eval_train_dataset = datasets.CIFAR10(
        root=args.data_dir, train=True, transform=eval_transform, download=False
    )
    test_dataset = datasets.CIFAR10(
        root=args.data_dir, train=False, transform=eval_transform, download=True
    )

    if args.val_split <= 0:
        return train_dataset, test_dataset, test_dataset

    if args.val_split < 1:
        val_len = int(len(train_dataset) * args.val_split)
    else:
        val_len = int(args.val_split)
    val_len = max(1, min(val_len, len(train_dataset) - 1))
    generator = torch.Generator().manual_seed(args.seed)
    all_indices = torch.randperm(len(train_dataset), generator=generator)
    val_indices = all_indices[:val_len]
    train_indices = all_indices[val_len:]
    train_dataset = Subset(train_dataset, train_indices.tolist())
    val_dataset = Subset(eval_train_dataset, val_indices.tolist())
    return train_dataset, val_dataset, test_dataset


def create_dataloaders(args: argparse.Namespace):
    train_transform, eval_transform = build_transforms(args)
    train_dataset, val_dataset, test_dataset = build_datasets(
        args, train_transform, eval_transform
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader


def create_model(args: argparse.Namespace) -> nn.Module:
    model = timm.create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=10,
    )
    return model


def create_optimizer(model: nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=True,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    return optimizer


def create_scheduler(
    optimizer: torch.optim.Optimizer, args: argparse.Namespace
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    if args.lr_scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr * 0.1
        )
    return None


def accuracy_from_logits(logits: torch.Tensor, target: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return (preds == target).float().sum().item()


def maybe_mix_targets(
    images: torch.Tensor,
    targets: torch.Tensor,
    mixup_alpha: float,
    cutmix_alpha: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    lam = 1.0
    shuffled_targets = targets
    if mixup_alpha > 0.0 and cutmix_alpha > 0.0:
        raise ValueError("Enable either mixup or cutmix, not both.")
    if mixup_alpha > 0.0:
        lam = torch.distributions.Beta(mixup_alpha, mixup_alpha).sample().item()
        index = torch.randperm(images.size(0), device=images.device)
        mixed_images = lam * images + (1 - lam) * images[index, :]
        return mixed_images, targets, targets[index], lam
    if cutmix_alpha > 0.0:
        lam = torch.distributions.Beta(cutmix_alpha, cutmix_alpha).sample().item()
        bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(-2), images.size(-1), lam)
        index = torch.randperm(images.size(0), device=images.device)
        new_images = images.clone()
        new_images[:, :, bbx1:bbx2, bby1:bby2] = images[index, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size(-2) * images.size(-1)))
        return new_images, targets, targets[index], lam
    return images, targets, targets, lam


def rand_bbox(height: int, width: int, lam: float) -> Tuple[int, int, int, int]:
    cut_rat = torch.sqrt(torch.tensor(1.0 - lam))
    cut_w = (width * cut_rat).to(torch.int)
    cut_h = (height * cut_rat).to(torch.int)
    cx = torch.randint(0, width, (1,)).item()
    cy = torch.randint(0, height, (1,)).item()
    bbx1 = max(cx - cut_w // 2, 0)
    bby1 = max(cy - cut_h // 2, 0)
    bbx2 = min(cx + cut_w // 2, width)
    bby2 = min(cy + cut_h // 2, height)
    return bbx1, bby1, bbx2, bby2


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler,
    epoch: int,
    log_interval: int,
    use_amp: bool,
    mixup_alpha: float,
    cutmix_alpha: float,
) -> TrainStats:
    model.train()
    running_loss = 0.0
    running_correct = 0.0
    total_samples = 0
    start_time = time.time()

    for step, (images, targets) in enumerate(loader, start=1):
        # Step 1: move current batch to GPU/CPU 设备，non_blocking=True 让数据拷贝与计算重叠
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        # Step 2: 根据 mixup/cutmix 设置返回增强后的图像和“软标签” (targets_a/targets_b + lam)
        images_aug, targets_a, targets_b, lam = maybe_mix_targets(
            images, targets, mixup_alpha, cutmix_alpha
        )

        optimizer.zero_grad(set_to_none=True)  # 清理上一轮梯度，set_to_none=True 节省显存
        with autocast(enabled=use_amp):  # Step 3: 在混合精度环境执行前向与损失计算
            outputs = model(images_aug)  # 前向推理
            if targets_a is targets_b:  # 没有 mixup/cutmix，直接计算交叉熵
                loss = criterion(outputs, targets_a)
            else:  # 有 mixup/cutmix，用 lam 组合两份标签的损失
                loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(
                    outputs, targets_b
                )

        # Step 4: 反向传播 & 权重更新（GradScaler 负责缩放/恢复，避免 AMP 下数值溢出）
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        running_correct += accuracy_from_logits(outputs, targets)
        total_samples += batch_size

        if step % log_interval == 0 or step == len(loader):
            avg_loss = running_loss / total_samples
            avg_acc = (running_correct / total_samples) * 100.0
            print(
                f"Epoch {epoch:03d} Step {step:04d}/{len(loader):04d} "
                f"Loss {avg_loss:.4f} Acc {avg_acc:.2f}%"
            )

    elapsed = time.time() - start_time
    samples_per_sec = total_samples / max(elapsed, 1e-8)
    avg_loss = running_loss / total_samples
    avg_acc = (running_correct / total_samples) * 100.0
    return TrainStats(avg_loss, avg_acc, samples_per_sec)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> TrainStats:
    model.eval()
    running_loss = 0.0
    running_correct = 0.0
    total_samples = 0
    start_time = time.time()

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, targets)
        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        running_correct += accuracy_from_logits(outputs, targets)
        total_samples += batch_size

    elapsed = time.time() - start_time
    samples_per_sec = total_samples / max(elapsed, 1e-8)
    avg_loss = running_loss / total_samples
    avg_acc = (running_correct / total_samples) * 100.0
    return TrainStats(avg_loss, avg_acc, samples_per_sec)


def save_checkpoint(state: Dict, path: Path) -> None:
    torch.save(state, path)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = create_dataloaders(args)
    model = create_model(args).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(model, args)
    scheduler = create_scheduler(optimizer, args)
    scaler = GradScaler(enabled=args.use_amp and device.type == "cuda")

    best_val_acc = 0.0
    best_ckpt = output_dir / "best.pt"
    last_ckpt = output_dir / "last.pt"
    metrics_path = output_dir / "metrics.csv"
    write_header = not metrics_path.exists()

    for epoch in range(1, args.epochs + 1):
        train_stats = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler,
            epoch,
            args.log_interval,
            args.use_amp and device.type == "cuda",
            args.mixup,
            args.cutmix,
        )
        val_stats = evaluate(model, val_loader, criterion, device)
        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step()

        print(
            f"Epoch {epoch:03d} Train Loss {train_stats.loss:.4f} Acc {train_stats.acc:.2f}% "
            f"| Val Loss {val_stats.loss:.4f} Acc {val_stats.acc:.2f}% | LR {current_lr:.6e}"
        )

        with open(metrics_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(
                    ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"]
                )
                write_header = False
            writer.writerow(
                [
                    epoch,
                    train_stats.loss,
                    train_stats.acc,
                    val_stats.loss,
                    val_stats.acc,
                    current_lr,
                ]
            )

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "args": vars(args),
        }
        save_checkpoint(checkpoint, last_ckpt)
        if val_stats.acc > best_val_acc:
            best_val_acc = val_stats.acc
            save_checkpoint(checkpoint, best_ckpt)
            print(f"New best model saved with val acc {best_val_acc:.2f}%")

    print("Evaluating best checkpoint on test set...")
    if best_ckpt.exists():
        state = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(state["model"])
    test_stats = evaluate(model, test_loader, criterion, device)
    print(
        f"Test Loss {test_stats.loss:.4f} | Test Acc {test_stats.acc:.2f}% "
        f"| Throughput {test_stats.samples_per_sec:.1f} samples/s"
    )


if __name__ == "__main__":
    main()
