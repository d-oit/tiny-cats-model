"""src/dit.py

Tiny Diffusion Transformer (TinyDiT) for cat image generation.
Implements flow matching with breed conditioning.

References:
- DiT Paper: https://arxiv.org/pdf/2212.09748
- Flow Matching: https://arxiv.org/pdf/2210.02747
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """Patchify input images."""

    def __init__(
        self,
        image_size: int = 128,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 384,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Patchify and flatten spatial dimensions.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Patch embeddings (B, N, D) where N = num_patches
        """
        x = self.proj(x)  # (B, D, H/p, W/p)
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        return x


class Attention(nn.Module):
    """Multi-head self-attention."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 6,
        qkv_bias: bool = False,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply self-attention.

        Args:
            x: Input tensor (B, N, D)

        Returns:
            Output tensor (B, N, D)
        """
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class Mlp(nn.Module):
    """MLP block for transformer."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply MLP.

        Args:
            x: Input tensor (B, N, D)

        Returns:
            Output tensor (B, N, D)
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class TimestepEmbedder(nn.Module):
    """Embeds timesteps into sinusoidal embeddings."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(
        t: torch.Tensor, dim: int, max_period: int = 10000
    ) -> torch.Tensor:
        """Create sinusoidal timestep embeddings.

        Args:
            t: Timestep tensor (B,)
            dim: Embedding dimension
            max_period: Maximum period for sinusoids

        Returns:
            Timestep embeddings (B, dim)
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        )
        freqs = freqs.to(t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Embed timesteps.

        Args:
            t: Timestep tensor (B,)

        Returns:
            Timestep embeddings (B, D)
        """
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class BreedEmbedder(nn.Module):
    """Embeds breed labels (one-hot) into conditioning vectors."""

    def __init__(self, num_breeds: int = 13, embed_dim: int = 384) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_breeds, embed_dim)
        self.num_breeds = num_breeds

    def forward(self, breeds: torch.Tensor) -> torch.Tensor:
        """Embed breed labels.

        Args:
            breeds: Breed indices (B,) or one-hot (B, num_breeds)

        Returns:
            Breed embeddings (B, D)
        """
        if breeds.dim() == 1:
            return self.embedding(breeds)
        else:
            # One-hot input
            breed_indices = breeds.argmax(dim=-1)
            return self.embedding(breed_indices)


class DiTBlock(nn.Module):
    """DiT block with AdaLN conditioning."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = Attention(embed_dim, num_heads=num_heads)

        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = Mlp(in_features=embed_dim, hidden_features=mlp_hidden_dim)

        # AdaLN conditioning
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, 6 * embed_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
    ) -> torch.Tensor:
        """Apply DiT block with conditioning.

        Args:
            x: Input tensor (B, N, D)
            c: Conditioning vector (B, D) from timestep + breed

        Returns:
            Output tensor (B, N, D)
        """
        mod_values = self.adaLN_modulation(c).chunk(6, dim=-1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod_values

        # Attention with AdaLN
        x = x + gate_msa.unsqueeze(1) * self.attn(
            self.norm1(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        )

        # MLP with AdaLN
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            self.norm2(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        )

        return x


class FinalLayer(nn.Module):
    """Final layer to unpatchify back to image space."""

    def __init__(
        self,
        embed_dim: int,
        patch_size: int,
        out_channels: int = 3,
    ) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(embed_dim)
        self.linear = nn.Linear(embed_dim, patch_size * patch_size * out_channels)

        # AdaLN conditioning
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, 2 * embed_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
    ) -> torch.Tensor:
        """Unpatchify to image space.

        Args:
            x: Input tensor (B, N, D)
            c: Conditioning vector (B, D)

        Returns:
            Image tensor (B, C, H, W)
        """
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = self.linear(
            self.norm_final(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        )
        return x


class TinyDiT(nn.Module):
    """Tiny Diffusion Transformer for cat image generation.

    Architecture:
    - Patch embedding
    - 12 transformer blocks with AdaLN conditioning
    - Final unpatchify layer
    - Breed conditioning via learned embeddings
    """

    def __init__(
        self,
        image_size: int = 128,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 13,  # 12 cat breeds + other
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # Patch embedding
        self.patch_embed = PatchEmbed(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )

        # Class embedding (breed conditioning)
        self.breed_embedder = BreedEmbedder(num_breeds=num_classes, embed_dim=embed_dim)

        # Timestep embedding
        self.t_embedder = TimestepEmbedder(hidden_size=embed_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                DiTBlock(embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )

        # Final layer
        self.final_layer = FinalLayer(
            embed_dim=embed_dim, patch_size=patch_size, out_channels=in_channels
        )

        # Initialize weights
        self.initialize_weights()

    def initialize_weights(self) -> None:
        """Initialize model weights."""

        # Initialize transformer layers
        def _basic_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize patch embedding like ViT
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Zero-out final layer
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        breeds: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input noisy image (B, C, H, W)
            t: Timestep (B,)
            breeds: Breed indices (B,) or one-hot (B, num_classes)

        Returns:
            Velocity prediction (B, C, H, W) for flow matching
        """
        # Patchify
        x = self.patch_embed(x)  # (B, N, D)

        # Get conditioning
        t_emb = self.t_embedder(t)  # (B, D)
        breed_emb = self.breed_embedder(breeds)  # (B, D)
        c = t_emb + breed_emb  # (B, D)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, c)

        # Unpatchify
        x = self.final_layer(x, c)  # (B, N, P*P*C)

        # Reshape to image (use explicit batch size for ONNX compatibility)
        batch_size = x.shape[0]
        x = x.transpose(1, 2).reshape(
            batch_size, self.in_channels, self.image_size, self.image_size
        )
        return x

    def forward_with_cfg(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        breeds: torch.Tensor,
        cfg_scale: float = 1.5,
    ) -> torch.Tensor:
        """Forward pass with classifier-free guidance.

        Args:
            x: Input noisy image (B, C, H, W)
            t: Timestep (B,)
            breeds: Breed indices (B,)
            cfg_scale: CFG scale (1.0 = no guidance)

        Returns:
            Velocity prediction with CFG (B, C, H, W)
        """
        if cfg_scale == 1.0:
            return self.forward(x, t, breeds)

        # Unconditional pass (use special "other" class)
        uncond = torch.full_like(breeds, self.num_classes - 1)
        pred_uncond = self.forward(x, t, uncond)
        pred_cond = self.forward(x, t, breeds)

        # CFG interpolation
        return pred_uncond + cfg_scale * (pred_cond - pred_uncond)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def tinydit_128(num_classes: int = 13) -> TinyDiT:
    """TinyDiT for 128x128 images.

    Args:
        num_classes: Number of breed classes

    Returns:
        TinyDiT model
    """
    return TinyDiT(
        image_size=128,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        num_classes=num_classes,
    )


def tinydit_256(num_classes: int = 13) -> TinyDiT:
    """TinyDiT for 256x256 images.

    Args:
        num_classes: Number of breed classes

    Returns:
        TinyDiT model
    """
    return TinyDiT(
        image_size=256,
        patch_size=16,
        embed_dim=512,
        depth=12,
        num_heads=8,
        num_classes=num_classes,
    )
