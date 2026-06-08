"""
vision_transformer.py
=====================
A clean, modular from-scratch Vision Transformer (ViT) for robust
learning experiments. Architecture follows the design used in the
rSDNet extension experiments (Parts 2 and 2b).

This module is intentionally self-contained — it can be imported
without any other module from this repository.

Usage
-----
from vision_transformer import build_vit, VisionTransformer

# Default (4M-param model used in medical imaging experiments)
model = build_vit(img_size=32, in_ch=3, num_classes=9)

# Custom configuration
model = VisionTransformer(
    img_size=32, patch_size=4, in_ch=3, num_classes=10,
    d_model=256, num_heads=8, ffn_dim=512, num_layers=6, dropout=0.1
)

References
----------
Dosovitskiy et al. (2021). "An image is worth 16x16 words: Transformers
for image recognition at scale." ICLR 2021.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """Linear projection of non-overlapping image patches.

    Splits an image into (img_size / patch_size)^2 patches,
    each flattened to a vector of size (patch_size^2 * in_ch),
    then linearly projected to d_model.

    Args:
        img_size:   Height (= width) of the input image in pixels.
        patch_size: Size of each square patch (must divide img_size).
        in_ch:      Number of input channels (1 for grayscale, 3 for RGB).
        d_model:    Token embedding dimension.
    """

    def __init__(self, img_size: int, patch_size: int, in_ch: int, d_model: int):
        super().__init__()
        assert img_size % patch_size == 0, (
            f"img_size ({img_size}) must be divisible by patch_size ({patch_size})"
        )
        self.n_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.proj = nn.Linear(patch_size * patch_size * in_ch, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor[B, in_ch, H, W]
        Returns:
            Tensor[B, n_patches, d_model]
        """
        B, C, H, W = x.shape
        p = self.patch_size
        # Extract patches: (B, C, H//p, p, W//p, p) → (B, n_patches, C*p*p)
        x = x.unfold(2, p, p).unfold(3, p, p)               # (B, C, H//p, W//p, p, p)
        x = x.contiguous().view(B, C, -1, p * p)             # (B, C, n_patches, p²)
        x = x.permute(0, 2, 1, 3).contiguous().view(B, -1, C * p * p)  # (B, n_patches, C·p²)
        return self.proj(x)                                   # (B, n_patches, d_model)


class TransformerEncoderBlock(nn.Module):
    """Pre-norm Transformer encoder block (LayerNorm before attention).

    Architecture: LN → MSA → residual → LN → FFN → residual

    Args:
        d_model:   Token embedding dimension.
        num_heads: Number of attention heads. Must divide d_model.
        ffn_dim:   Hidden dimension of the feed-forward network.
        dropout:   Dropout rate applied after attention and FFN.
    """

    def __init__(self, d_model: int, num_heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor[B, n_tokens, d_model]
        Returns:
            Tensor[B, n_tokens, d_model]
        """
        # Self-attention with pre-norm and residual
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop1(h)
        # Feed-forward with pre-norm and residual
        h = self.norm2(x)
        return x + self.drop2(self.ffn(h))


class VisionTransformer(nn.Module):
    """From-scratch Vision Transformer for image classification.

    Architecture:
        PatchEmbedding → Positional Embedding → N × TransformerEncoderBlock
        → Global Average Pool → LayerNorm → Linear classifier

    No [CLS] token — uses global average pooling over patch tokens.
    This is simpler and works well for small images (≤ 64×64).

    Args:
        img_size:    Input image size (height = width). Default: 32.
        patch_size:  Patch size. Default: 4 (gives 64 patches for 32×32).
        in_ch:       Number of input channels. Default: 3 (RGB).
        num_classes: Number of output classes. Default: 10.
        d_model:     Token embedding dimension. Default: 256.
        num_heads:   Number of attention heads. Default: 8.
        ffn_dim:     FFN hidden dimension. Default: 512.
        num_layers:  Number of Transformer encoder blocks. Default: 6.
        dropout:     Dropout rate. Default: 0.1.

    Total parameters: ~4M with default settings (d=256, L=6).
    """

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_ch: int = 3,
        num_classes: int = 10,
        d_model: int = 256,
        num_heads: int = 8,
        ffn_dim: int = 512,
        num_layers: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_ch, d_model)
        n_patches = self.patch_embed.n_patches

        # Learned positional embeddings (one per patch position)
        self.pos_embed = nn.Embedding(n_patches, d_model)

        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier uniform for linear layers, normal for embeddings."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor[B, in_ch, H, W] — pixel values in [0, 1]
        Returns:
            Tensor[B, num_classes] — raw logits (no softmax)
        """
        # Patch embedding
        x = self.patch_embed(x)                        # (B, n_patches, d_model)
        # Add positional embeddings
        pos = self.pos_embed(torch.arange(x.size(1), device=x.device))
        x = x + pos
        # Transformer encoder blocks
        for block in self.blocks:
            x = block(x)
        # Global average pooling + final norm
        x = self.norm(x.mean(dim=1))                  # (B, d_model)
        return self.head(x)                            # (B, num_classes)

    @property
    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_vit(
    img_size: int,
    in_ch: int,
    num_classes: int,
    patch_size: int = 4,
    d_model: int = 256,
    num_heads: int = 8,
    ffn_dim: int = 512,
    num_layers: int = 6,
    dropout: float = 0.1,
    device: str | torch.device = "cpu",
) -> VisionTransformer:
    """Build and return a VisionTransformer with the given config.

    Args:
        img_size:    Input image size (H = W).
        in_ch:       Number of input channels.
        num_classes: Number of output classes.
        patch_size:  Patch grid size. Default: 4 (→ 64 patches for 32×32).
        d_model:     Embedding dimension. Default: 256.
        num_heads:   Attention heads. Default: 8.
        ffn_dim:     FFN hidden dim. Default: 512.
        num_layers:  Encoder depth. Default: 6.
        dropout:     Dropout. Default: 0.1.
        device:      Target device. Default: 'cpu'.

    Returns:
        VisionTransformer on the specified device.
    """
    model = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_ch=in_ch,
        num_classes=num_classes,
        d_model=d_model,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    n_params = model.num_parameters
    print(
        f"[ViT] {n_params / 1e6:.2f}M params | "
        f"d={d_model}, L={num_layers}, patch={patch_size}, heads={num_heads} | "
        f"device={device}"
    )
    return model
