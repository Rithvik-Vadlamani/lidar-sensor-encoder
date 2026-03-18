"""
Point cloud tokenizer: Furthest Point Sampling (FPS) + k-NN grouping + mini-PointNet.
Each local patch of points is encoded into a single embedding token.
"""
import torch
import torch.nn as nn


def furthest_point_sample(xyz: torch.Tensor, n_centers: int) -> torch.Tensor:
    """
    Iterative furthest point sampling.
    Args:
        xyz: (B, N, 3) point cloud
        n_centers: number of centers to sample
    Returns:
        idx: (B, n_centers) indices of sampled centers
    """
    B, N, _ = xyz.shape
    device = xyz.device
    centroids = torch.zeros(B, n_centers, dtype=torch.long, device=device)
    distances = torch.full((B, N), float("inf"), device=device)

    # Start from a random point
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)

    for i in range(n_centers):
        centroids[:, i] = farthest
        center = xyz[torch.arange(B), farthest, :].unsqueeze(1)  # (B, 1, 3)
        dist = ((xyz - center) ** 2).sum(dim=-1)                  # (B, N)
        distances = torch.min(distances, dist)
        farthest = distances.argmax(dim=-1)

    return centroids


def knn_group(xyz: torch.Tensor, centers: torch.Tensor, k: int) -> torch.Tensor:
    """
    For each center, find the k nearest neighbors and return their coordinates
    relative to the center (local frame).
    Args:
        xyz:     (B, N, 3) full point cloud
        centers: (B, M, 3) center positions
        k:       number of neighbors
    Returns:
        patches: (B, M, k, 3) local patches in center-relative coordinates
    """
    B, N, _ = xyz.shape
    _, M, _ = centers.shape

    # Pairwise squared distances between centers and all points
    # (B, M, N)
    dists = (
        centers.unsqueeze(2) - xyz.unsqueeze(1)
    ).pow(2).sum(dim=-1)

    # (B, M, k) indices of k nearest neighbors
    idx = dists.topk(k, dim=-1, largest=False).indices

    # Gather neighbor coordinates: (B, M, k, 3)
    idx_expanded = idx.unsqueeze(-1).expand(-1, -1, -1, 3)
    xyz_expanded = xyz.unsqueeze(1).expand(-1, M, -1, -1)
    patches = xyz_expanded.gather(2, idx_expanded)

    # Make coordinates relative to each center
    patches = patches - centers.unsqueeze(2)

    return patches


class MiniPointNet(nn.Module):
    """
    Shared MLP applied per-point within a patch, followed by max-pooling,
    to produce a single patch embedding.
    """
    def __init__(self, in_dim: int = 3, embed_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
        )

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patches: (B, M, k, 3)
        Returns:
            tokens: (B, M, embed_dim)
        """
        B, M, k, C = patches.shape
        x = patches.view(B * M * k, C)
        x = self.mlp(x)
        x = x.view(B * M, k, -1)
        x = x.max(dim=1).values          # max-pool over neighbors → (B*M, embed_dim)
        return x.view(B, M, -1)


class PointTokenizer(nn.Module):
    """
    Full tokenizer: FPS → k-NN grouping → mini-PointNet.
    Outputs:
        tokens:  (B, M, embed_dim)
        centers: (B, M, 3)
    """
    def __init__(self, n_centers: int = 64, k: int = 32, embed_dim: int = 256):
        super().__init__()
        self.n_centers = n_centers
        self.k = k
        self.mini_pointnet = MiniPointNet(in_dim=3, embed_dim=embed_dim)

    def forward(self, xyz: torch.Tensor):
        """
        Args:
            xyz: (B, N, 3)
        Returns:
            tokens:  (B, n_centers, embed_dim)
            centers: (B, n_centers, 3)
        """
        center_idx = furthest_point_sample(xyz, self.n_centers)       # (B, M)
        B, M = center_idx.shape

        # Gather center coordinates
        centers = xyz.gather(
            1,
            center_idx.unsqueeze(-1).expand(-1, -1, 3)
        )                                                              # (B, M, 3)

        patches = knn_group(xyz, centers, self.k)                     # (B, M, k, 3)
        tokens = self.mini_pointnet(patches)                          # (B, M, embed_dim)

        return tokens, centers
