"""
KITTI LiDAR point cloud dataset loader.

KITTI Velodyne scans are stored as binary float32 files with shape (N, 4):
columns are [x, y, z, reflectance]. We use only XYZ.

Dataset structure expected:
    <root>/
        sequences/
            00/ ... 21/
                velodyne/
                    000000.bin
                    000001.bin
                    ...

Download: http://www.cvlibs.net/datasets/kitti/eval_odometry.php
"""
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset


class KITTIPointCloudDataset(Dataset):
    """
    Loads individual LiDAR scans from KITTI odometry sequences.

    Args:
        root:        Path to the KITTI odometry dataset root.
        sequences:   List of sequence IDs to load (e.g. ['00', '01']).
                     If None, loads all sequences found.
        n_points:    Number of points to sample per scan (random subsample).
        transform:   Optional callable applied to the (N, 3) numpy array.
    """

    def __init__(
        self,
        root: str,
        sequences=None,
        n_points: int = 8192,
        transform=None,
    ):
        self.root = root
        self.n_points = n_points
        self.transform = transform

        velodyne_root = os.path.join(root, "sequences")
        if sequences is None:
            sequences = sorted(os.listdir(velodyne_root))

        self.files = []
        for seq in sequences:
            pattern = os.path.join(velodyne_root, seq, "velodyne", "*.bin")
            self.files.extend(sorted(glob.glob(pattern)))

        if len(self.files) == 0:
            raise FileNotFoundError(
                f"No .bin files found under {velodyne_root}. "
                "Please download the KITTI odometry dataset."
            )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        scan = np.fromfile(self.files[idx], dtype=np.float32).reshape(-1, 4)
        xyz = scan[:, :3]                             # drop reflectance

        # Random subsample to fixed size
        if len(xyz) >= self.n_points:
            choice = np.random.choice(len(xyz), self.n_points, replace=False)
        else:
            choice = np.random.choice(len(xyz), self.n_points, replace=True)
        xyz = xyz[choice]

        if self.transform is not None:
            xyz = self.transform(xyz)

        return torch.from_numpy(xyz).float()          # (n_points, 3)
