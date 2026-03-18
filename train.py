"""
Training script for PointMAE self-supervised pre-training on KITTI LiDAR.

Usage:
    python train.py --config configs/pointmae_kitti.yaml

    # Override any config key on the command line:
    python train.py --config configs/pointmae_kitti.yaml training.batch_size=16
"""
import argparse
import os
import sys
import time
import math

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import PointMAE
from data import (
    KITTIPointCloudDataset,
    Compose,
    NormalizePointCloud,
    RandomJitter,
    RandomRotate,
    RandomScale,
    RandomShuffle,
)
from utils import AverageMeter


# ──────────────────────────────────────────────────────────────────────────────
# Config helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Apply 'section.key=value' CLI overrides to a nested config dict."""
    for override in overrides:
        key_path, _, value = override.partition("=")
        keys = key_path.split(".")
        node = cfg
        for k in keys[:-1]:
            node = node[k]
        # Try to cast to int / float, else leave as string
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                pass
        node[keys[-1]] = value
    return cfg


# ──────────────────────────────────────────────────────────────────────────────
# Learning-rate schedule with linear warm-up + cosine decay
# ──────────────────────────────────────────────────────────────────────────────

def get_lr(epoch: int, cfg: dict) -> float:
    warmup = cfg["training"]["warmup_epochs"]
    total = cfg["training"]["epochs"]
    base_lr = cfg["training"]["lr"]
    min_lr = cfg["training"]["min_lr"]

    if epoch < warmup:
        return base_lr * (epoch + 1) / warmup

    progress = (epoch - warmup) / max(total - warmup, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine


# ──────────────────────────────────────────────────────────────────────────────
# Dataset / DataLoader builder
# ──────────────────────────────────────────────────────────────────────────────

def build_loaders(cfg: dict):
    dcfg = cfg["data"]

    train_transform = Compose([
        NormalizePointCloud(),
        RandomRotate(max_angle_deg=180),
        RandomScale(lo=0.8, hi=1.2),
        RandomJitter(sigma=0.01, clip=0.05),
        RandomShuffle(),
    ]) if dcfg.get("augment", True) else NormalizePointCloud()

    val_transform = NormalizePointCloud()

    train_ds = KITTIPointCloudDataset(
        root=dcfg["root"],
        sequences=dcfg["sequences_train"],
        n_points=dcfg["n_points"],
        transform=train_transform,
    )
    val_ds = KITTIPointCloudDataset(
        root=dcfg["root"],
        sequences=dcfg["sequences_val"],
        n_points=dcfg["n_points"],
        transform=val_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=True,
    )
    return train_loader, val_loader


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, device, epoch, cfg):
    model.train()
    meter = AverageMeter("train_loss")
    log_interval = cfg["training"]["log_interval"]

    for i, batch in enumerate(loader):
        xyz = batch.to(device)                        # (B, N, 3)
        loss, _, _ = model(xyz)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        meter.update(loss.item(), n=xyz.shape[0])

        if (i + 1) % log_interval == 0:
            print(
                f"  [Epoch {epoch:03d}] step {i+1}/{len(loader)} "
                f"| loss={meter.avg:.4f}"
            )

    return meter.avg


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    meter = AverageMeter("val_loss")
    for batch in loader:
        xyz = batch.to(device)
        loss, _, _ = model(xyz)
        meter.update(loss.item(), n=xyz.shape[0])
    return meter.avg


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train PointMAE on KITTI LiDAR")
    parser.add_argument("--config", default="configs/pointmae_kitti.yaml")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument("overrides", nargs="*", help="Config overrides: section.key=value")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.overrides)

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Model ─────────────────────────────────────────────────────────────────
    mcfg = cfg["model"]
    model = PointMAE(
        n_centers=mcfg["n_centers"],
        k=mcfg["k"],
        embed_dim=mcfg["embed_dim"],
        enc_n_heads=mcfg["enc_n_heads"],
        enc_n_layers=mcfg["enc_n_layers"],
        enc_ffn_dim=mcfg["enc_ffn_dim"],
        dec_n_heads=mcfg["dec_n_heads"],
        dec_n_layers=mcfg["dec_n_layers"],
        dec_ffn_dim=mcfg["dec_ffn_dim"],
        mask_ratio=mcfg["mask_ratio"],
        dropout=mcfg["dropout"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    tcfg = cfg["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tcfg["lr"],
        weight_decay=tcfg["weight_decay"],
    )

    start_epoch = 0
    best_val_loss = float("inf")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"Resumed from epoch {ckpt['epoch']} (val_loss={best_val_loss:.4f})")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader = build_loaders(cfg)
    print(f"Train: {len(train_loader.dataset)} scans | Val: {len(val_loader.dataset)} scans")

    # ── Output dir ────────────────────────────────────────────────────────────
    out_dir = tcfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    # ── Training ──────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, tcfg["epochs"]):
        # Update learning rate
        lr = get_lr(epoch, cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, cfg)
        val_loss = validate(model, val_loader, device)
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:03d}/{tcfg['epochs']} | "
            f"lr={lr:.2e} | "
            f"train={train_loss:.4f} | val={val_loss:.4f} | "
            f"time={elapsed:.1f}s"
        )

        # Save periodic checkpoint
        if (epoch + 1) % tcfg["save_interval"] == 0:
            ckpt_path = os.path.join(out_dir, f"ckpt_epoch{epoch:03d}.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "best_val_loss": best_val_loss,
                    "config": cfg,
                },
                ckpt_path,
            )

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "best_val_loss": best_val_loss,
                    "config": cfg,
                },
                os.path.join(out_dir, "best_model.pth"),
            )
            print(f"  ✓ New best val_loss={val_loss:.4f} — checkpoint saved.")

    print(f"\nTraining complete. Best val_loss: {best_val_loss:.4f}")
    print(f"Encoder weights saved to {out_dir}/best_model.pth")


if __name__ == "__main__":
    main()
