"""
Transformer encoder and decoder blocks used in PointMAE.
"""
import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Learnable positional encoding projected from 3D center coordinates.
    Maps (x, y, z) → embed_dim so the model knows where each token came from.
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim),
        )

    def forward(self, centers: torch.Tensor) -> torch.Tensor:
        """
        Args:
            centers: (B, M, 3)
        Returns:
            pos_embed: (B, M, embed_dim)
        """
        return self.mlp(centers)


class TransformerBlock(nn.Module):
    """Single transformer block: multi-head self-attention + feed-forward."""
    def __init__(self, embed_dim: int, n_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, key_padding_mask=None) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x


class TransformerEncoder(nn.Module):
    """Stack of transformer blocks used to encode visible tokens."""
    def __init__(
        self,
        embed_dim: int = 256,
        n_heads: int = 4,
        n_layers: int = 6,
        ffn_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, ffn_dim, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, n_visible, embed_dim)
        Returns:
            x: (B, n_visible, embed_dim)
        """
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class TransformerDecoder(nn.Module):
    """
    Lightweight transformer decoder.
    Takes the encoded visible tokens + learnable mask tokens and reconstructs
    the full token sequence (visible + masked positions).
    """
    def __init__(
        self,
        embed_dim: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        ffn_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, ffn_dim, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, M, embed_dim) — full sequence with mask tokens inserted
        Returns:
            x: (B, M, embed_dim)
        """
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
