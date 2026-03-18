"""
PointMAE Demo — trains on synthetic point clouds, no dataset required.
Runs entirely on CPU in ~2 minutes. Shows loss decreasing over time.

Usage:
    python demo.py
"""
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from models import PointMAE


# ── Tiny synthetic dataset ─────────────────────────────────────────────────────
class SyntheticPointCloudDataset(Dataset):
    """
    Generates random point clouds shaped like noisy spheres and cubes.
    No files needed — everything is generated in memory.
    """
    SHAPES = ["sphere", "cube", "cylinder"]

    def __init__(self, n_samples: int = 200, n_points: int = 512):
        self.n_samples = n_samples
        self.n_points = n_points
        torch.manual_seed(42)
        self.data = [self._make_shape(i % len(self.SHAPES)) for i in range(n_samples)]

    def _make_shape(self, shape_id: int) -> torch.Tensor:
        n = self.n_points
        if shape_id == 0:  # sphere
            pts = torch.randn(n, 3)
            pts = pts / pts.norm(dim=1, keepdim=True)
        elif shape_id == 1:  # cube surface
            pts = torch.rand(n, 3) * 2 - 1
            face = torch.randint(0, 3, (n,))
            sign = (torch.randint(0, 2, (n,)).float() * 2 - 1)
            for i in range(n):
                pts[i, face[i]] = sign[i]
        else:  # cylinder
            theta = torch.rand(n) * 2 * 3.14159
            h = torch.rand(n) * 2 - 1
            pts = torch.stack([theta.cos(), theta.sin(), h], dim=1)

        pts = pts + torch.randn_like(pts) * 0.02   # add tiny noise
        pts = pts / pts.abs().max()                 # normalize to [-1, 1]
        return pts.float()

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.data[idx]


# ── Progress bar ───────────────────────────────────────────────────────────────
def progress_bar(current, total, loss, width=30):
    filled = int(width * current / total)
    bar = "█" * filled + "░" * (width - filled)
    print(f"\r  [{bar}] {current}/{total}  loss={loss:.4f}", end="", flush=True)


# ── Main demo ──────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  PointMAE — Self-Supervised LiDAR Encoder Demo")
    print("  GM AI/ML Intern — Sensor Encoder Project")
    print("=" * 60)

    # Small model config so it runs fast on CPU
    CONFIG = dict(
        n_centers=16, k=16, embed_dim=64,
        enc_n_heads=2, enc_n_layers=3, enc_ffn_dim=128,
        dec_n_heads=2, dec_n_layers=2, dec_ffn_dim=128,
        mask_ratio=0.75, dropout=0.1,
    )
    EPOCHS     = 10
    BATCH_SIZE = 8
    LR         = 1e-3
    N_SAMPLES  = 200
    N_POINTS   = 512

    device = torch.device("cpu")

    # ── Dataset ────────────────────────────────────────────────────────────────
    print(f"\n[1/4] Generating synthetic point cloud dataset...")
    dataset = SyntheticPointCloudDataset(n_samples=N_SAMPLES, n_points=N_POINTS)
    train_ds, val_ds = torch.utils.data.random_split(dataset, [160, 40])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    print(f"    Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")
    print(f"    Each scan: {N_POINTS} points (x, y, z)")

    # ── Model ──────────────────────────────────────────────────────────────────
    print(f"\n[2/4] Building PointMAE model...")
    model = PointMAE(**CONFIG).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"    Parameters : {n_params:,}")
    print(f"    Mask ratio : {CONFIG['mask_ratio']*100:.0f}% of patches hidden during training")
    print(f"    Patch count: {CONFIG['n_centers']} patches × {CONFIG['k']} points each")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    # ── Training ───────────────────────────────────────────────────────────────
    print(f"\n[3/4] Training for {EPOCHS} epochs...")
    print(f"    (Loss = Chamfer Distance — lower is better)\n")

    history = []
    t_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0

        for i, batch in enumerate(train_loader):
            xyz = batch.to(device)
            loss, _, _ = model(xyz)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            progress_bar(i + 1, len(train_loader), train_loss / (i + 1))

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                loss, _, _ = model(batch.to(device))
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss   /= len(val_loader)
        history.append((train_loss, val_loss))
        elapsed = time.time() - t_start

        print(f"\r  Epoch {epoch:02d}/{EPOCHS} | "
              f"train={train_loss:.4f}  val={val_loss:.4f} | "
              f"lr={scheduler.get_last_lr()[0]:.2e} | "
              f"{elapsed:.0f}s elapsed")

    # ── Results ────────────────────────────────────────────────────────────────
    first_loss = history[0][0]
    final_loss = history[-1][0]
    improvement = (first_loss - final_loss) / first_loss * 100

    print(f"\n[4/4] Evaluating encoder representations...")
    model.eval()
    all_features = []
    with torch.no_grad():
        for batch in val_loader:
            feats = model.encode(batch.to(device))   # (B, embed_dim)
            all_features.append(feats)
    features = torch.cat(all_features, dim=0)

    print(f"\n{'=' * 60}")
    print(f"  RESULTS SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Initial train loss   : {first_loss:.4f}")
    print(f"  Final train loss     : {final_loss:.4f}")
    print(f"  Improvement          : {improvement:.1f}% reduction in Chamfer Distance")
    print(f"  Encoder output shape : {features.shape}  (samples × embed_dim)")
    print(f"  Total training time  : {time.time() - t_start:.1f}s on CPU")
    print(f"{'=' * 60}")
    print(f"\n  The encoder is now producing {CONFIG['embed_dim']}-dim feature vectors")
    print(f"  for each point cloud — ready to use for downstream tasks like")
    print(f"  3D object detection or semantic segmentation.\n")
    print(f"  On a GPU with full KITTI data, this scales to 5.5M parameters")
    print(f"  and 300 epochs for publication-quality representations.")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
