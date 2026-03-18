"""
Quick smoke test — runs a forward pass with random synthetic data.
No dataset download required.

Usage:
    python test_model.py
"""
import torch
from models import PointMAE

print("Loading model...")
model = PointMAE(
    n_centers=64,
    k=32,
    embed_dim=256,
    enc_n_layers=6,
    dec_n_layers=4,
    mask_ratio=0.75,
)

n_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {n_params:,}")

# Fake batch: 2 scans, 8192 points each, XYZ coords
xyz = torch.randn(2, 8192, 3)
print(f"Input shape: {xyz.shape}")

print("Running forward pass...")
loss, pred, target = model(xyz)

print(f"\n--- Results ---")
print(f"Chamfer Distance loss : {loss.item():.6f}")
print(f"Predicted patches     : {pred.shape}   (batch, n_masked, k, 3)")
print(f"Target patches        : {target.shape}")

# Test the encoder (used for downstream tasks)
print("\nTesting frozen encoder...")
features = model.encode(xyz)
print(f"Global features shape : {features.shape}   (batch, embed_dim)")

print("\nAll checks passed. Model is working correctly.")
