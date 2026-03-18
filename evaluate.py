"""
Downstream evaluation: freeze the PointMAE encoder and train a lightweight
linear head on top to measure representation quality.

Currently supports:
  - Reconstruction loss on a held-out set (self-supervised quality check)
  - Linear probing on a labelled classification dataset (future extension)

Usage:
    # Self-supervised reconstruction evaluation
    python evaluate.py --checkpoint checkpoints/best_model.pth \
                       --config configs/pointmae_kitti.yaml \
                       --mode reconstruct

    # Visualise predicted vs. ground-truth patches for a single scan
    python evaluate.py --checkpoint checkpoints/best_model.pth \
                       --config configs/pointmae_kitti.yaml \
                       --mode visualize --scan path/to/scan.bin
"""
import argparse
import os

import numpy as np
import torch
import yaml

from models import PointMAE
from data import KITTIPointCloudDataset, NormalizePointCloud
from utils import AverageMeter


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, cfg: dict, device: torch.device) -> PointMAE:
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

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} (val_loss={ckpt['val_loss']:.4f})")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Reconstruction evaluation
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_reconstruct(model: PointMAE, cfg: dict, device: torch.device):
    """Measure average Chamfer Distance on validation sequences."""
    from torch.utils.data import DataLoader

    val_ds = KITTIPointCloudDataset(
        root=cfg["data"]["root"],
        sequences=cfg["data"]["sequences_val"],
        n_points=cfg["data"]["n_points"],
        transform=NormalizePointCloud(),
    )
    loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=4)

    model.eval()
    meter = AverageMeter("chamfer")

    for batch in loader:
        xyz = batch.to(device)
        loss, _, _ = model(xyz)
        meter.update(loss.item(), n=xyz.shape[0])

    print(f"Validation Chamfer Distance: {meter.avg:.6f}")
    return meter.avg


# ──────────────────────────────────────────────────────────────────────────────
# Visualisation (requires open3d)
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def visualize_reconstruction(model: PointMAE, scan_path: str, device: torch.device, n_points: int = 8192):
    """
    Load a single .bin scan, run PointMAE, and display predicted vs. ground-truth
    masked patches side-by-side using Open3D.
    """
    try:
        import open3d as o3d
    except ImportError:
        print("open3d is required for visualisation: pip install open3d")
        return

    # Load scan
    scan = np.fromfile(scan_path, dtype=np.float32).reshape(-1, 4)
    xyz = scan[:, :3]
    choice = np.random.choice(len(xyz), min(n_points, len(xyz)), replace=False)
    xyz = xyz[choice]

    transform = NormalizePointCloud()
    xyz = transform(xyz)

    xyz_t = torch.from_numpy(xyz).float().unsqueeze(0).to(device)  # (1, N, 3)

    model.eval()
    _, pred, target = model(xyz_t)

    # Flatten patches for display
    pred_pts = pred.squeeze(0).reshape(-1, 3).cpu().numpy()       # (n_mask*k, 3)
    tgt_pts = target.squeeze(0).reshape(-1, 3).cpu().numpy()

    def make_pcd(pts, color):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.paint_uniform_color(color)
        return pcd

    pred_pcd = make_pcd(pred_pts, [0.0, 0.6, 1.0])   # blue = predicted
    tgt_pcd = make_pcd(tgt_pts, [1.0, 0.4, 0.0])      # orange = ground-truth

    # Offset for side-by-side display
    tgt_pcd.translate([3.0, 0, 0])

    print("Blue = predicted patches | Orange = ground-truth patches")
    o3d.visualization.draw_geometries(
        [pred_pcd, tgt_pcd],
        window_name="PointMAE Reconstruction",
        width=1200,
        height=600,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate PointMAE encoder")
    parser.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint")
    parser.add_argument("--config", default="configs/pointmae_kitti.yaml")
    parser.add_argument(
        "--mode",
        choices=["reconstruct", "visualize"],
        default="reconstruct",
    )
    parser.add_argument("--scan", default=None, help="Path to .bin scan (visualize mode)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_model(args.checkpoint, cfg, device)

    if args.mode == "reconstruct":
        eval_reconstruct(model, cfg, device)

    elif args.mode == "visualize":
        if args.scan is None:
            print("--scan path required for visualize mode")
            return
        visualize_reconstruction(model, args.scan, device, n_points=cfg["data"]["n_points"])


if __name__ == "__main__":
    main()
