"""src/flow_matching.py

Flow matching training for TinyDiT.

Implements:
- Flow matching loss (velocity prediction)
- Sampling with ODE integration
- EMA weight averaging

References:
- Flow Matching: https://arxiv.org/pdf/2210.02747
- DiT: https://arxiv.org/pdf/2212.09748
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# Supported timestep distributions for flow-matching training.
TimestepSampling = Literal["uniform", "logit_normal"]


class FlowMatchingLoss(nn.Module):
    """Flow matching loss for velocity prediction."""

    def __init__(self) -> None:
        """Initialize flow matching loss."""
        super().__init__()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute flow matching loss.

        Args:
            pred: Model prediction (velocity)
            target: Target velocity (x1 - x0)

        Returns:
            Scalar loss
        """
        return F.mse_loss(pred, target)


def sample_t(
    batch_size: int,
    device: torch.device,
    t_min: float = 0.0,
    t_max: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample timesteps uniformly.

    Args:
        batch_size: Number of samples
        device: Device to sample on
        t_min: Minimum timestep
        t_max: Maximum timestep
        generator: Optional seeded generator, so validation loss can reuse the
            exact same timesteps across evaluations

    Returns:
        Timestep tensor (batch_size,)
    """
    return (
        torch.rand(batch_size, device=device, generator=generator) * (t_max - t_min)
        + t_min
    )


def sample_t_logit_normal(
    batch_size: int,
    device: torch.device,
    mean: float = 0.0,
    std: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample timesteps from a logit-normal distribution (SD3-style).

    ``t = sigmoid(mean + std * N(0, 1))`` concentrates training on the middle of
    the trajectory, where the velocity field carries the most information, and
    is the standard flow-matching recipe (Esser et al., "Scaling Rectified Flow
    Transformers for High-Resolution Image Synthesis", 2024). Uniform sampling
    spends a third of its capacity on t near 0 and near 1, where the target is
    dominated by either pure noise or the target image itself.

    Args:
        batch_size: Number of samples
        device: Device to sample on
        mean: Mean of the underlying normal (0.0 matches SD3)
        std: Standard deviation of the underlying normal (1.0 matches SD3)
        generator: Optional seeded generator for reproducible evaluation

    Returns:
        Timestep tensor (batch_size,) in (0, 1)
    """
    normal = torch.randn(batch_size, device=device, generator=generator)
    return torch.sigmoid(normal * std + mean)


def sample_timesteps(
    batch_size: int,
    device: torch.device,
    sampling: TimestepSampling = "uniform",
    t_min: float = 0.0,
    t_max: float = 1.0,
    logit_normal_mean: float = 0.0,
    logit_normal_std: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample training timesteps from the configured distribution.

    Args:
        batch_size: Number of samples
        device: Device to sample on
        sampling: "uniform" or "logit_normal"
        t_min: Minimum timestep (uniform only)
        t_max: Maximum timestep (uniform only)
        logit_normal_mean: Mean of the logit-normal (logit_normal only)
        logit_normal_std: Std of the logit-normal (logit_normal only)
        generator: Optional seeded generator

    Returns:
        Timestep tensor (batch_size,)

    Raises:
        ValueError: If ``sampling`` is not a supported distribution.
    """
    if sampling == "logit_normal":
        return sample_t_logit_normal(
            batch_size,
            device,
            mean=logit_normal_mean,
            std=logit_normal_std,
            generator=generator,
        )
    if sampling != "uniform":
        raise ValueError(
            f"Unsupported timestep sampling: {sampling!r}. "
            "Use 'uniform' or 'logit_normal'."
        )
    return sample_t(batch_size, device, t_min=t_min, t_max=t_max, generator=generator)


def flow_matching_step(
    model: nn.Module,
    x0: torch.Tensor,
    x1: torch.Tensor,
    t: torch.Tensor,
    breeds: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute flow matching training step.

    Args:
        model: TinyDiT model
        x0: Source noise (B, C, H, W)
        x1: Target image (B, C, H, W)
        t: Timestep (B,)
        breeds: Breed conditioning (B,)

    Returns:
        Tuple of (prediction, target_velocity)
    """
    # Interpolate between x0 and x1
    t_expanded = t.view(-1, 1, 1, 1)
    xt = t_expanded * x1 + (1 - t_expanded) * x0

    # Target velocity is constant: x1 - x0
    target = x1 - x0

    # Model prediction
    pred = model(xt, t, breeds)

    return pred, target


@torch.no_grad()
def sample(
    model: nn.Module,
    breeds: torch.Tensor,
    num_steps: int = 50,
    device: torch.device | None = None,
    image_size: int = 128,
    cfg_scale: float = 1.5,
    progress: bool = False,
) -> torch.Tensor:
    """Sample images using flow matching ODE integration.

    Args:
        model: TinyDiT model
        breeds: Breed indices (B,)
        num_steps: Number of ODE integration steps
        device: Device to sample on
        image_size: Output image size
        cfg_scale: Classifier-free guidance scale
        progress: Show progress bar

    Returns:
        Generated images (B, C, H, W)
    """
    if device is None:
        device = next(model.parameters()).device

    batch_size = len(breeds)

    # Start with Gaussian noise
    x = torch.randn(batch_size, 3, image_size, image_size, device=device)

    # Euler integration
    dt = 1.0 / num_steps
    timesteps = torch.linspace(0, 1, num_steps + 1, device=device)

    iterator = range(num_steps)
    if progress:
        iterator = tqdm(iterator, desc="Sampling")

    for i in iterator:
        t = timesteps[i].expand(batch_size)

        # Get velocity
        if cfg_scale > 1.0:
            velocity = model.forward_with_cfg(x, t, breeds, cfg_scale)
        else:
            velocity = model(x, t, breeds)

        # Euler step: x(t+dt) = x(t) + dt * v
        x = x + dt * velocity

    return x


class EMA:
    """Exponential Moving Average for model weights."""

    def __init__(self, beta: float = 0.9999) -> None:
        """Initialize EMA.

        Args:
            beta: EMA decay rate
        """
        self.beta = beta
        self.shadow_params: dict[str, torch.Tensor] = {}
        self.step = 0

    def init(self, model: nn.Module) -> None:
        """Initialize EMA with model parameters.

        Args:
            model: Model to track
        """
        self.shadow_params = {
            name: param.clone().detach()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    def update(self, model: nn.Module) -> None:
        """Update EMA with current model parameters.

        Args:
            model: Current model
        """
        self.step += 1
        beta = min(self.beta, 1 - (1 - self.beta) / self.step)

        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow_params:
                # Move shadow param to model param device on mismatch
                # (e.g. after loading checkpoint with map_location="cpu")
                if self.shadow_params[name].device != param.device:
                    self.shadow_params[name] = self.shadow_params[name].to(param.device)
                self.shadow_params[name].mul_(beta).add_(param.data, alpha=1 - beta)

    def apply(self, model: nn.Module) -> None:
        """Apply EMA weights to model.

        Args:
            model: Model to update
        """
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow_params:
                param.data.copy_(self.shadow_params[name].to(param.device))

    def save(self, path: str) -> None:
        """Save EMA weights.

        Args:
            path: Path to save checkpoint
        """
        torch.save({"shadow_params": self.shadow_params, "step": self.step}, path)

    def load(self, path: str) -> None:
        """Load EMA weights.

        Args:
            path: Path to checkpoint
        """
        checkpoint = torch.load(path, map_location="cpu")
        self.shadow_params = checkpoint["shadow_params"]
        self.step = checkpoint["step"]
