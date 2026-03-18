"""
Point cloud augmentation transforms compatible with numpy arrays (N, 3).
"""
import numpy as np


class RandomJitter:
    """Add Gaussian noise to each point."""
    def __init__(self, sigma: float = 0.01, clip: float = 0.05):
        self.sigma = sigma
        self.clip = clip

    def __call__(self, xyz: np.ndarray) -> np.ndarray:
        noise = np.clip(
            np.random.normal(0, self.sigma, xyz.shape),
            -self.clip, self.clip
        )
        return xyz + noise


class RandomRotate:
    """Random rotation around the Z-axis (gravity-aligned for LiDAR)."""
    def __init__(self, max_angle_deg: float = 180.0):
        self.max_angle_rad = np.radians(max_angle_deg)

    def __call__(self, xyz: np.ndarray) -> np.ndarray:
        angle = np.random.uniform(-self.max_angle_rad, self.max_angle_rad)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        R = np.array([
            [cos_a, -sin_a, 0],
            [sin_a,  cos_a, 0],
            [0,      0,     1],
        ], dtype=np.float32)
        return xyz @ R.T


class RandomScale:
    """Randomly scale the point cloud."""
    def __init__(self, lo: float = 0.8, hi: float = 1.2):
        self.lo = lo
        self.hi = hi

    def __call__(self, xyz: np.ndarray) -> np.ndarray:
        scale = np.random.uniform(self.lo, self.hi)
        return xyz * scale


class RandomShuffle:
    """Randomly shuffle point order (order-invariant sanity check)."""
    def __call__(self, xyz: np.ndarray) -> np.ndarray:
        idx = np.random.permutation(len(xyz))
        return xyz[idx]


class NormalizePointCloud:
    """
    Translate to centroid and scale so that the point cloud fits
    inside a unit sphere (max distance from origin = 1).
    """
    def __call__(self, xyz: np.ndarray) -> np.ndarray:
        centroid = xyz.mean(axis=0)
        xyz = xyz - centroid
        max_dist = np.sqrt((xyz ** 2).sum(axis=1)).max()
        if max_dist > 0:
            xyz = xyz / max_dist
        return xyz


class Compose:
    """Chain multiple transforms."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, xyz: np.ndarray) -> np.ndarray:
        for t in self.transforms:
            xyz = t(xyz)
        return xyz
