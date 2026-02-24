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

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class FlowMatchingLoss(nn.Module):
    """Flow matching loss for velocity prediction."""

    def __init__(self, prediction_type: str = "velocity") -> None:
        """Initialize flow matching loss.

        Args:
            prediction_type: Type of prediction ("velocity" or "noise")
        """
        super().__init__()
        self.prediction_type = prediction_type

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute flow matching loss.

        Args:
            pred: Model prediction (velocity or noise)
            target: Target velocity (x1 - x0)

        Returns:
            Scalar loss
        """
        if self.prediction_type == "velocity":
            return F.mse_loss(pred, target)
        elif self.prediction_type == "noise":
            # Convert to noise prediction
            return F.mse_loss(pred, target)
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")


def sample_t(
    batch_size: int,
    device: torch.device,
    t_min: float = 0.0,
    t_max: float = 1.0,
) -> torch.Tensor:
    """Sample timesteps uniformly.

    Args:
        batch_size: Number of samples
        device: Device to sample on
        t_min: Minimum timestep
        t_max: Maximum timestep

    Returns:
        Timestep tensor (batch_size,)
    """
    return torch.rand(batch_size, device=device) * (t_max - t_min) + t_min


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
                self.shadow_params[name].mul_(beta).add_(param.data, alpha=1 - beta)

    def apply(self, model: nn.Module) -> None:
        """Apply EMA weights to model.

        Args:
            model: Model to update
        """
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow_params:
                param.data.copy_(self.shadow_params[name].data)

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
