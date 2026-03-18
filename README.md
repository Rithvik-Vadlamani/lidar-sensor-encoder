# LiDAR Sensor Encoder — Self-Supervised PointMAE

A self-supervised **Masked Autoencoder for 3D LiDAR point clouds**, implemented in PyTorch and trained on the **KITTI odometry dataset**.

Inspired by [Point-MAE (Pang et al., 2022)](https://arxiv.org/abs/2203.06604), this project learns rich 3D representations **without any labels** by masking ~75% of point cloud patches and training a Transformer encoder-decoder to reconstruct the missing geometry.

---

## Architecture

```
Raw LiDAR Scan (N × 3)
        │
        ▼
  Point Tokenizer
  ┌─────────────────────────────────────────┐
  │  FPS → k-NN Grouping → mini-PointNet   │
  │  (B, N, 3) → (B, M, D)                 │
  └─────────────────────────────────────────┘
        │
        ▼
  Random Masking (75% of tokens hidden)
        │
        ▼
  Transformer Encoder  (visible tokens only)
        │
        ▼
  Insert [MASK] tokens + positional encoding
        │
        ▼
  Transformer Decoder  (full sequence)
        │
        ▼
  Prediction Head → reconstruct masked patches
        │
        ▼
  Chamfer Distance Loss (masked patches only)
```

| Component | Details |
|---|---|
| Tokenizer | FPS (64 centers) + k-NN (k=32) + mini-PointNet |
| Positional encoding | Learnable MLP from 3D center coordinates |
| Encoder | 6-layer Transformer, d=256, 4 heads |
| Decoder | 4-layer Transformer, d=256, 4 heads |
| Mask ratio | 75% |
| Loss | Symmetric Chamfer Distance |

---

## Dataset

Download **KITTI Odometry** (Velodyne laser data) from:
> http://www.cvlibs.net/datasets/kitti/eval_odometry.php

Place it at `data/kitti/` with the following structure:

```
data/kitti/
└── sequences/
    ├── 00/
    │   └── velodyne/
    │       ├── 000000.bin
    │       └── ...
    ├── 01/
    └── ...
```

---

## Setup

```bash
pip install -r requirements.txt
```

For visualisation support (optional):
```bash
pip install open3d
```

---

## Training

```bash
python train.py --config configs/pointmae_kitti.yaml
```

Override config values on the command line:
```bash
python train.py --config configs/pointmae_kitti.yaml training.batch_size=16 training.epochs=100
```

Resume from a checkpoint:
```bash
python train.py --config configs/pointmae_kitti.yaml --resume checkpoints/ckpt_epoch050.pth
```

---

## Evaluation

**Reconstruction quality** (Chamfer Distance on validation sequences):
```bash
python evaluate.py --checkpoint checkpoints/best_model.pth \
                   --config configs/pointmae_kitti.yaml \
                   --mode reconstruct
```

**Visualise predicted vs. ground-truth patches** (requires `open3d`):
```bash
python evaluate.py --checkpoint checkpoints/best_model.pth \
                   --config configs/pointmae_kitti.yaml \
                   --mode visualize \
                   --scan data/kitti/sequences/09/velodyne/000000.bin
```

---

## Downstream Use (Frozen Encoder)

After pre-training, the encoder can be used as a feature extractor:

```python
import torch
from models import PointMAE

model = PointMAE()
ckpt = torch.load("checkpoints/best_model.pth")
model.load_state_dict(ckpt["model"])

xyz = torch.randn(1, 8192, 3)   # your point cloud (B, N, 3)
features = model.encode(xyz)     # (B, 256) — global embedding
```

Attach a lightweight head (e.g. 2-layer MLP or linear layer) for tasks like:
- 3D object detection
- Semantic segmentation
- Scene classification

---

## Project Structure

```
lidar-sensor-encoder/
├── configs/
│   └── pointmae_kitti.yaml     # all hyperparameters
├── data/
│   ├── kitti_dataset.py        # KITTI dataloader
│   └── transforms.py           # point cloud augmentations
├── models/
│   ├── point_tokenizer.py      # FPS + k-NN + mini-PointNet
│   ├── transformer.py          # Transformer encoder & decoder
│   └── pointmae.py             # full PointMAE model + Chamfer loss
├── utils/
│   └── metrics.py              # AverageMeter, Chamfer helper
├── train.py                    # training entry point
├── evaluate.py                 # evaluation & visualisation
└── requirements.txt
```

---

## References

- Pang et al. (2022). *Masked Autoencoders for Point Cloud Self-supervised Learning*. [arXiv:2203.06604](https://arxiv.org/abs/2203.06604)
- He et al. (2021). *Masked Autoencoders Are Scalable Vision Learners*. [arXiv:2111.06377](https://arxiv.org/abs/2111.06377)
- Geiger et al. (2012). *Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite*. CVPR 2012.
