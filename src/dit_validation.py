"""src/dit_validation.py

Held-out evaluation for TinyDiT flow matching.

The training loop's running loss is averaged over augmented, freshly resampled
*training* batches, which makes it a noisy proxy for generator quality — yet it
was the only signal driving checkpoint selection and early stopping (ADR-059).

These helpers evaluate a fixed window of validation batches with a seeded
generator, so the returned number is comparable across steps: a change reflects
the weights, not resampled noise. They also support evaluating the EMA weights,
which is the checkpoint that actually gets sampled and shipped.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from flow_matching import (
    EMA,
    TimestepSampling,
    flow_matching_step,
    sample_timesteps,
)

logger = logging.getLogger(__name__)

__all__ = ["evaluate_flow_loss", "evaluate_model_and_ema"]


@torch.no_grad()
def evaluate_flow_loss(
    model: nn.Module,
    loader: Iterable[Any],
    device: torch.device,
    num_batches: int = 8,
    seed: int = 42,
    timestep_sampling: TimestepSampling = "uniform",
    logit_normal_mean: float = 0.0,
    logit_normal_std: float = 1.0,
) -> float:
    """Mean flow-matching MSE over a fixed window of validation batches.

    Args:
        model: TinyDiT model.
        loader: Validation DataLoader yielding (images, breeds).
        device: Device to evaluate on.
        num_batches: Maximum number of batches to evaluate (0 = all).
        seed: Seed for timestep/noise sampling, making the metric comparable
            from one evaluation to the next.
        timestep_sampling: Timestep distribution; must match training so the
            held-out number measures the objective actually being optimised.
        logit_normal_mean: Mean of the logit-normal sampler.
        logit_normal_std: Std of the logit-normal sampler.

    Returns:
        Mean MSE, or NaN when the loader yielded no batches.
    """
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    was_training = model.training
    model.eval()

    total = 0.0
    batches = 0
    for images, breeds in loader:
        if num_batches > 0 and batches >= num_batches:
            break
        images = images.to(device, non_blocking=True)
        breeds = breeds.to(device, non_blocking=True)

        t = sample_timesteps(
            images.shape[0],
            device,
            sampling=timestep_sampling,
            logit_normal_mean=logit_normal_mean,
            logit_normal_std=logit_normal_std,
            generator=generator,
        )
        x0 = torch.randn(images.shape, device=device, generator=generator)
        pred, target = flow_matching_step(model, x0, images, t, breeds)
        total += F.mse_loss(pred, target).item()
        batches += 1

    if was_training:
        model.train()

    if batches == 0:
        logger.warning("Validation loader produced no batches; loss is NaN.")
        return float("nan")
    return total / batches


@torch.no_grad()
def evaluate_model_and_ema(
    model: nn.Module,
    loader: Iterable[Any],
    device: torch.device,
    num_batches: int = 8,
    seed: int = 42,
    ema: EMA | None = None,
    timestep_sampling: TimestepSampling = "uniform",
    logit_normal_mean: float = 0.0,
    logit_normal_std: float = 1.0,
) -> tuple[float, float | None]:
    """Evaluate the raw and the EMA weights on the same validation window.

    The EMA weights are applied in place and restored afterwards, so the caller
    keeps its training-time parameters exactly as they were.

    Args:
        model: TinyDiT model (restored to its pre-call weights).
        loader: Validation DataLoader.
        device: Device to evaluate on.
        num_batches: Maximum number of batches to evaluate.
        seed: Seed for the deterministic evaluation window.
        ema: Optional EMA tracker; when absent only the raw loss is returned.
        timestep_sampling: Timestep distribution, matching training.
        logit_normal_mean: Mean of the logit-normal sampler.
        logit_normal_std: Std of the logit-normal sampler.

    Returns:
        Tuple of (raw loss, EMA loss). EMA loss is None when no EMA weights are
        available.
    """
    eval_kwargs: dict[str, Any] = {
        "num_batches": num_batches,
        "seed": seed,
        "timestep_sampling": timestep_sampling,
        "logit_normal_mean": logit_normal_mean,
        "logit_normal_std": logit_normal_std,
    }
    raw_loss = evaluate_flow_loss(model, loader, device, **eval_kwargs)

    if ema is None or not ema.shadow_params:
        return raw_loss, None

    backup = {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    try:
        ema.apply(model)
        ema_loss = evaluate_flow_loss(model, loader, device, **eval_kwargs)
    finally:
        for name, param in model.named_parameters():
            if name in backup:
                param.data.copy_(backup[name].to(param.device))

    return raw_loss, ema_loss
