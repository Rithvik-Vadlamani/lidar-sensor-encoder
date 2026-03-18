"""
Training utilities: metric tracking and standalone Chamfer Distance.
"""
import torch


class AverageMeter:
    """Tracks a running mean of a scalar (e.g. loss, accuracy)."""

    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        return f"{self.name}: {self.avg:.4f}"


def compute_chamfer(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Convenience wrapper that returns Chamfer Distance as a Python float.
    Args:
        pred:   (B, M, k, 3)
        target: (B, M, k, 3)
    """
    from models.pointmae import chamfer_distance
    with torch.no_grad():
        return chamfer_distance(pred, target).item()
