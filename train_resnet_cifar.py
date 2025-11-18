#!/usr/bin/env python3
"""
Minimal ResNet baseline training on CIFAR-10.
This script covers data loading, training/validation loops, checkpointing,
and metric logging so other experiments can reuse the same workflow.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from dataclasses import dataclass
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
    parser = argparse.ArgumentParser(description="ResNet baseline on CIFAR-10")
    parser.add_argument("--model", default="resnet18", help="timm model name")
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        action="store_true",
        help="load ImageNet pretrained weights",
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
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--val-split", type=float, default=0.1, help="fraction or count")
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for split initialization"
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="runs/resnet_baseline")
    parser.add_argument(
        "--no-amp",
        dest="use_amp",
        action="store_false",
        help="disable mixed precision",
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
    args: argparse.Namespace,
    train_transform,
    eval_transform,
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
        val_dataset = test_dataset
        return train_dataset, val_dataset, test_dataset

    if args.val_split < 1:
        val_len = int(len(train_dataset) * args.val_split)
    else:
        val_len = int(args.val_split)
    val_len = max(1, min(val_len, len(train_dataset) - 1))
    train_len = len(train_dataset) - val_len

    generator = torch.Generator().manual_seed(args.seed)
    all_indices = torch.randperm(len(train_dataset), generator=generator)
    val_indices = all_indices[:val_len]
    train_indices = all_indices[val_len:]
    train_dataset = Subset(train_dataset, train_indices.tolist())
    val_dataset = Subset(eval_train_dataset, val_indices.tolist())
    return train_dataset, val_dataset, test_dataset


def create_dataloaders(
    args: argparse.Namespace,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
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
        drop_last=False,
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
            optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
        )
    return None


def accuracy_from_logits(logits: torch.Tensor, target: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return (preds == target).float().sum().item()


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
) -> TrainStats:
    model.train()
    running_loss = 0.0
    running_correct = 0.0
    total_samples = 0
    start_time = time.time()

    for step, (images, targets) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        with autocast(enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, targets)

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


def save_checkpoint(
    state: Dict,
    path: str,
) -> None:
    torch.save(state, path)


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8") as f:
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
    best_checkpoint_path = os.path.join(args.output_dir, "best.pt")
    last_checkpoint_path = os.path.join(args.output_dir, "last.pt")
    metrics_path = os.path.join(args.output_dir, "metrics.csv")
    write_header = not os.path.exists(metrics_path)

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
        )
        val_stats = evaluate(model, val_loader, criterion, device)
        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step()

        print(
            f"Epoch {epoch:03d} Train Loss {train_stats.loss:.4f} Acc {train_stats.acc:.2f}% "
            f"| Val Loss {val_stats.loss:.4f} Acc {val_stats.acc:.2f}% | LR {current_lr:.6f}"
        )

        row = [
            epoch,
            train_stats.loss,
            train_stats.acc,
            val_stats.loss,
            val_stats.acc,
            current_lr,
        ]
        with open(metrics_path, "a", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            if write_header:
                writer.writerow(
                    ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"]
                )
                write_header = False
            writer.writerow(row)

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "args": vars(args),
        }
        save_checkpoint(checkpoint, last_checkpoint_path)
        if val_stats.acc > best_val_acc:
            best_val_acc = val_stats.acc
            save_checkpoint(checkpoint, best_checkpoint_path)
            print(f"New best model saved with val acc {best_val_acc:.2f}%")

    print("Evaluating best checkpoint on test set...")
    if os.path.exists(best_checkpoint_path):
        state = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(state["model"])
    test_stats = evaluate(model, test_loader, criterion, device)
    print(
        f"Test Loss {test_stats.loss:.4f} | Test Acc {test_stats.acc:.2f}% "
        f"| Throughput {test_stats.samples_per_sec:.1f} samples/s"
    )


if __name__ == "__main__":
    main()
