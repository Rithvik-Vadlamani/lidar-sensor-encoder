"""
PointMAE: Masked Autoencoder for 3D Point Clouds.

Pipeline:
  1. Tokenize point cloud via FPS + k-NN + mini-PointNet  → (B, M, D)
  2. Randomly mask ~75% of tokens
  3. Encode visible tokens with a deep Transformer
  4. Insert learnable [MASK] tokens at masked positions + add positional encoding
  5. Decode full sequence with a lightweight Transformer
  6. Predict original point coordinates for masked patches
  7. Compute Chamfer Distance loss on masked patches only

Reference: Point-MAE (Pang et al., 2022) https://arxiv.org/abs/2203.06604
"""
import torch
import torch.nn as nn

from .point_tokenizer import PointTokenizer, knn_group
from .transformer import TransformerEncoder, TransformerDecoder, PositionalEncoding


class PointMAE(nn.Module):
    def __init__(
        self,
        # Tokenizer
        n_centers: int = 64,
        k: int = 32,
        # Model dimensions
        embed_dim: int = 256,
        # Encoder
        enc_n_heads: int = 4,
        enc_n_layers: int = 6,
        enc_ffn_dim: int = 512,
        # Decoder
        dec_n_heads: int = 4,
        dec_n_layers: int = 4,
        dec_ffn_dim: int = 512,
        # MAE
        mask_ratio: float = 0.75,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.n_centers = n_centers
        self.k = k
        self.mask_ratio = mask_ratio
        self.embed_dim = embed_dim

        # --- Tokenizer ---
        self.tokenizer = PointTokenizer(n_centers=n_centers, k=k, embed_dim=embed_dim)

        # --- Positional encoding (shared by encoder and decoder) ---
        self.pos_enc = PositionalEncoding(embed_dim)

        # --- Encoder ---
        self.encoder = TransformerEncoder(
            embed_dim=embed_dim,
            n_heads=enc_n_heads,
            n_layers=enc_n_layers,
            ffn_dim=enc_ffn_dim,
            dropout=dropout,
        )

        # --- Learnable mask token ---
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # --- Decoder input projection (encoder dim → decoder dim, same here) ---
        self.decoder_embed = nn.Linear(embed_dim, embed_dim)

        # --- Decoder ---
        self.decoder = TransformerDecoder(
            embed_dim=embed_dim,
            n_heads=dec_n_heads,
            n_layers=dec_n_layers,
            ffn_dim=dec_ffn_dim,
            dropout=dropout,
        )

        # --- Prediction head: embed_dim → k * 3 (reconstruct k points per patch) ---
        self.pred_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, k * 3),
        )

    # ------------------------------------------------------------------
    # Masking helpers
    # ------------------------------------------------------------------
    def _random_mask(self, B: int, M: int, device: torch.device):
        """
        Returns:
            visible_idx: (B, n_visible)  indices of unmasked tokens
            masked_idx:  (B, n_masked)   indices of masked tokens
        """
        n_masked = int(M * self.mask_ratio)
        noise = torch.rand(B, M, device=device)
        ids_shuffle = noise.argsort(dim=-1)

        masked_idx = ids_shuffle[:, :n_masked]
        visible_idx = ids_shuffle[:, n_masked:]
        return visible_idx, masked_idx

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, xyz: torch.Tensor):
        """
        Args:
            xyz: (B, N, 3) raw point cloud

        Returns (during training):
            loss:       scalar Chamfer Distance on masked patches
            pred:       (B, n_masked, k, 3) predicted point patches
            target:     (B, n_masked, k, 3) ground-truth point patches
        """
        B = xyz.shape[0]
        device = xyz.device

        # 1. Tokenize
        tokens, centers = self.tokenizer(xyz)          # (B, M, D), (B, M, 3)
        M = tokens.shape[1]

        # 2. Positional encoding
        pos = self.pos_enc(centers)                    # (B, M, D)

        # 3. Random masking
        vis_idx, mask_idx = self._random_mask(B, M, device)
        n_vis = vis_idx.shape[1]
        n_mask = mask_idx.shape[1]

        # Gather visible tokens + positions
        def gather_idx(x, idx):
            """x: (B, M, D), idx: (B, n) → (B, n, D)"""
            return x.gather(1, idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]))

        vis_tokens = gather_idx(tokens, vis_idx) + gather_idx(pos, vis_idx)

        # 4. Encode visible tokens
        encoded = self.encoder(vis_tokens)             # (B, n_vis, D)

        # 5. Build full sequence for decoder
        #    Place encoded visible tokens and mask tokens back in original order
        full_tokens = torch.zeros(B, M, self.embed_dim, device=device)
        full_tokens.scatter_(
            1,
            vis_idx.unsqueeze(-1).expand(-1, -1, self.embed_dim),
            self.decoder_embed(encoded),
        )
        mask_tokens = self.mask_token.expand(B, n_mask, -1)
        full_tokens.scatter_(
            1,
            mask_idx.unsqueeze(-1).expand(-1, -1, self.embed_dim),
            mask_tokens,
        )
        # Add positional encoding for all positions
        full_tokens = full_tokens + pos

        # 6. Decode
        decoded = self.decoder(full_tokens)            # (B, M, D)

        # 7. Predict masked patches only
        masked_decoded = gather_idx(decoded, mask_idx) # (B, n_mask, D)
        pred_flat = self.pred_head(masked_decoded)      # (B, n_mask, k*3)
        pred = pred_flat.view(B, n_mask, self.k, 3)    # (B, n_mask, k, 3)

        # 8. Ground-truth patches for masked centers
        masked_centers = gather_idx(centers, mask_idx) # (B, n_mask, 3)
        target = knn_group(xyz, masked_centers, self.k) # (B, n_mask, k, 3)

        # 9. Chamfer Distance loss (masked patches only)
        loss = chamfer_distance(pred, target)

        return loss, pred, target

    @torch.no_grad()
    def encode(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Extract global point cloud representation (no masking).
        Used for downstream evaluation with a frozen encoder.

        Args:
            xyz: (B, N, 3)
        Returns:
            features: (B, embed_dim)  — max-pooled over all tokens
        """
        tokens, centers = self.tokenizer(xyz)
        pos = self.pos_enc(centers)
        encoded = self.encoder(tokens + pos)
        return encoded.max(dim=1).values               # (B, embed_dim)


# ------------------------------------------------------------------
# Chamfer Distance
# ------------------------------------------------------------------
def chamfer_distance(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Symmetric Chamfer Distance between two point sets.
    Args:
        pred:   (B, M, k, 3)
        target: (B, M, k, 3)
    Returns:
        scalar loss
    """
    B, M, k, _ = pred.shape
    p = pred.view(B * M, k, 3)
    t = target.view(B * M, k, 3)

    # (B*M, k, k) pairwise squared distances
    diff = p.unsqueeze(2) - t.unsqueeze(1)
    dist = diff.pow(2).sum(dim=-1)

    # Each predicted point → nearest target point
    loss_p2t = dist.min(dim=2).values.mean()
    # Each target point → nearest predicted point
    loss_t2p = dist.min(dim=1).values.mean()

    return loss_p2t + loss_t2p
