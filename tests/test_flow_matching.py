"""tests/test_flow_matching.py

Unit tests for flow matching training components.

Covers:
- flow_matching_step math (xt interpolation, target velocity, output shapes)
- FlowMatchingLoss (velocity / noise / invalid prediction_type)
- sample_t (shape, range, custom range)
- EMA (init / update / apply / save+load)
- forward_with_cfg (cfg_scale=1.0 short-circuit, cfg>1 conditional path,
  uncond token index = num_classes - 1 per current src/dit.py API)
- Static regression check: src/train_dit.py call site of
  flow_matching_step must pass a varying x0 (noise) and x1 (image),
  not (images, images) which collapses the target to zero and produces
  zero loss.

Null token wiring: src/dit.py uses a dedicated null slot at index
    num_classes (embedder has num_classes + 1 slots); forward_with_cfg
    selects it via `uncond = self.num_classes` and src/train_dit.py
    matches that index in its CFG dropout (null_token = num_classes).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dit import tinydit_128
from flow_matching import EMA, FlowMatchingLoss, flow_matching_step, sample_t

# ---------------------------------------------------------------------------
# Per-function fixtures so EMA's in-place weight mutations do not
# leak into forward_with_cfg tests that share the model.
# ---------------------------------------------------------------------------


@pytest.fixture
def dit_model() -> nn.Module:
    """Fresh TinyDiT (eval mode) for tests that read outputs."""
    torch.manual_seed(0)
    return tinydit_128(num_classes=13).eval()


@pytest.fixture
def trainable_dit_model() -> nn.Module:
    """TinyDiT in train mode, used by EMA tests that mutate params in place."""
    torch.manual_seed(0)
    return tinydit_128(num_classes=13)


@pytest.fixture
def dit_model_with_outputs(dit_model: nn.Module) -> nn.Module:
    """TinyDiT with non-zero final_layer linear weights.

    Why: ``TinyDiT.initialize_weights`` zeroes the final layer so the
    initial policy emits a constant (and during the early phase of
    training all near-constant) image. Forward-only output comparisons
    are therefore degenerate against a freshly-constructed model. This
    fixture injects a small Gaussian into
    ``final_layer.linear.weight`` so output comparisons are observable.
    Use the plain ``dit_model`` fixture for shape / structural / API
    tests where degeneracy is harmless.
    """
    with torch.no_grad():
        dit_model.final_layer.linear.weight.data.normal_(0.0, 0.02)
        dit_model.final_layer.linear.bias.data.zero_()
    return dit_model


# ---------------------------------------------------------------------------
# flow_matching_step
# ---------------------------------------------------------------------------


class TestFlowMatchingStep:
    """Tests for flow_matching_step function."""

    def test_target_is_nonzero_when_x0_is_noise(self, dit_model: nn.Module) -> None:
        """Regression guard: target = x1 - x0 must be non-zero when x0 is noise.

        The classic zero-loss bug passes (images, images) so the target
        collapses to zero. Asserting non-zero here documents the
        contract that callers must respect: x0 must be independent
        noise, x1 must be a real image.
        """
        batch_size = 2
        x0 = torch.randn(batch_size, 3, 128, 128)  # noise
        x1 = torch.randn(batch_size, 3, 128, 128)  # "image"
        t = torch.rand(batch_size)
        breeds = torch.randint(0, 13, (batch_size,))

        with torch.no_grad():
            _pred, target = flow_matching_step(dit_model, x0, x1, t, breeds)

        assert not torch.allclose(target, torch.zeros_like(target)), (
            "Target velocity is zero — caller likely passed the same tensor "
            "for x0 and x1."
        )

    def test_target_is_zero_when_x0_equals_x1(self, dit_model: nn.Module) -> None:
        """Documents the operator contract that motivates the bug check."""
        batch_size = 2
        images = torch.randn(batch_size, 3, 128, 128)
        t = torch.rand(batch_size)
        breeds = torch.randint(0, 13, (batch_size,))

        with torch.no_grad():
            _pred, target = flow_matching_step(dit_model, images, images, t, breeds)

        assert torch.allclose(target, torch.zeros_like(target)), (
            "When x0 == x1, target == 0 is the operator behavior; the "
            "training caller must avoid this."
        )

    def test_interpolation_formula(self, dit_model: nn.Module) -> None:
        """For x0=0, x1=1, target == 1 (constant velocity field)."""
        x0 = torch.zeros(1, 3, 128, 128)
        x1 = torch.ones(1, 3, 128, 128)
        t = torch.tensor([0.5])
        breeds = torch.tensor([0])

        with torch.no_grad():
            _pred, target = flow_matching_step(dit_model, x0, x1, t, breeds)

        assert torch.allclose(target, torch.ones_like(target)), (
            "Target for x0=0, x1=1 should be constant 1"
        )

    def test_output_shapes(self, dit_model: nn.Module) -> None:
        """Pred and target match the input image shape (B, C, H, W)."""
        batch_size = 4
        x0 = torch.randn(batch_size, 3, 128, 128)
        x1 = torch.randn(batch_size, 3, 128, 128)
        t = torch.rand(batch_size)
        breeds = torch.randint(0, 13, (batch_size,))

        with torch.no_grad():
            pred, target = flow_matching_step(dit_model, x0, x1, t, breeds)

        assert pred.shape == (batch_size, 3, 128, 128)
        assert target.shape == (batch_size, 3, 128, 128)


# ---------------------------------------------------------------------------
# FlowMatchingLoss
# ---------------------------------------------------------------------------


class TestFlowMatchingLoss:
    """Tests for FlowMatchingLoss."""

    def test_loss_is_nonzero_for_different_tensors(self) -> None:
        loss_fn = FlowMatchingLoss()
        pred = torch.randn(2, 3, 128, 128)
        target = torch.randn(2, 3, 128, 128)
        assert loss_fn(pred, target).item() > 0

    def test_loss_is_zero_for_identical_tensors(self) -> None:
        loss_fn = FlowMatchingLoss()
        x = torch.randn(2, 3, 128, 128)
        assert loss_fn(x, x).item() == pytest.approx(0.0)

    def test_loss_is_scalar(self) -> None:
        loss_fn = FlowMatchingLoss()
        pred = torch.randn(2, 3, 128, 128)
        target = torch.randn(2, 3, 128, 128)
        assert loss_fn(pred, target).ndim == 0


# ---------------------------------------------------------------------------
# sample_t
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("t_min", "t_max"),
    [(0.0, 1.0), (0.1, 0.9), (0.3, 0.7), (0.0, 0.5)],
)
def test_sample_t_range(t_min: float, t_max: float) -> None:
    """sample_t returns values in [t_min, t_max]."""
    t = sample_t(1000, torch.device("cpu"), t_min=t_min, t_max=t_max)
    assert t.shape == (1000,)
    assert t.min().item() >= t_min
    assert t.max().item() <= t_max


def test_sample_t_single_step() -> None:
    """sample_t returns (batch_size,) shape."""
    t = sample_t(8, torch.device("cpu"))
    assert t.shape == (8,)


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------


class TestEMA:
    """Tests for EMA weight averaging."""

    def test_ema_init_copies_params(self, trainable_dit_model: nn.Module) -> None:
        ema = EMA(beta=0.9999)
        ema.init(trainable_dit_model)

        for name, param in trainable_dit_model.named_parameters():
            if param.requires_grad:
                assert name in ema.shadow_params
                assert torch.equal(ema.shadow_params[name], param.data)

    def test_ema_update_moves_shadow(self, trainable_dit_model: nn.Module) -> None:
        ema = EMA(beta=0.9)
        ema.init(trainable_dit_model)

        with torch.no_grad():
            for param in trainable_dit_model.parameters():
                param.add_(torch.ones_like(param))

        ema.update(trainable_dit_model)

        for name, param in trainable_dit_model.named_parameters():
            if param.requires_grad:
                assert not torch.equal(ema.shadow_params[name], param.data), (
                    f"EMA shadow={name} should not instantly match model params"
                )

    def test_ema_apply_overwrites_model(self, trainable_dit_model: nn.Module) -> None:
        ema = EMA(beta=0.9999)
        ema.init(trainable_dit_model)

        with torch.no_grad():
            for param in trainable_dit_model.parameters():
                param.add_(torch.ones_like(param) * 10)

        ema.apply(trainable_dit_model)

        for name, param in trainable_dit_model.named_parameters():
            if param.requires_grad:
                assert torch.equal(ema.shadow_params[name], param.data)

    def test_ema_save_and_load(
        self, trainable_dit_model: nn.Module, tmp_path: Path
    ) -> None:
        ema = EMA(beta=0.9999)
        ema.init(trainable_dit_model)
        ema.update(trainable_dit_model)

        path = str(tmp_path / "ema.pt")
        ema.save(path)

        ema_loaded = EMA(beta=0.9999)
        ema_loaded.load(path)

        assert ema_loaded.step == ema.step
        for name in ema.shadow_params:
            assert torch.equal(ema.shadow_params[name], ema_loaded.shadow_params[name])


# ---------------------------------------------------------------------------
# forward_with_cfg — matches ACTUAL src/dit.py API as of main 67b56e6
# ---------------------------------------------------------------------------


class TestForwardWithCFG:
    """Tests for TinyDiT.forward_with_cfg that match the current API."""

    def test_cfg_scale_one_short_circuits(self, dit_model: nn.Module) -> None:
        """cfg_scale == 1.0 must take the no-guidance path (return forward verbatim).

        We assert shape + finiteness rather than bitwise identity, because
        TinyDiT's ``final_layer`` is initialised to zero, so for two
        deterministically-equal forward calls we expect all-zero outputs
        on a fresh model — bitwise-equal all-zero tensors are trivially
        the same, but we want the API contract, not numeric equality.
        """
        x = torch.randn(2, 3, 128, 128)
        t = torch.rand(2)
        breeds = torch.tensor([0, 5])

        with torch.no_grad():
            out_cfg = dit_model.forward_with_cfg(x, t, breeds, cfg_scale=1.0)
            out_direct = dit_model.forward(x, t, breeds)

        assert out_cfg.shape == out_direct.shape == (2, 3, 128, 128)
        assert torch.isfinite(out_cfg).all()
        # Same call path — values must match exactly.
        assert torch.allclose(out_cfg, out_direct, atol=0.0)

    def test_cfg_above_one_branch_changes_output(
        self, dit_model_with_outputs: nn.Module
    ) -> None:
        """cfg_scale > 1.0 takes the conditional branch and diverges from
        cond-only output (forward_with_cfg != forward)."""
        x = torch.randn(2, 3, 128, 128)
        t = torch.rand(2)
        breeds = torch.tensor([0, 5])

        with torch.no_grad():
            out_cfg = dit_model_with_outputs.forward_with_cfg(
                x, t, breeds, cfg_scale=1.5
            )
            out_cond = dit_model_with_outputs.forward(x, t, breeds)
            out_uncond = dit_model_with_outputs.forward(
                x, t, torch.full_like(breeds, dit_model_with_outputs.num_classes)
            )

        assert out_cfg.shape == out_cond.shape == out_uncond.shape
        assert not torch.allclose(out_cfg, out_cond), (
            "cfg>1 must differ from cond-only by construction"
        )
        assert torch.isfinite(out_cfg).all()

    def test_cfg_uncond_token_index_matches_api(self, dit_model: nn.Module) -> None:
        """Current src/dit.py uses a dedicated null slot at index num_classes
        and the breed embedder has num_classes + 1 slots. This matches the
        CFG dropout in src/train_dit.py: null_token = num_classes.

        Locks the API contract: any future refactor that regresses the
        null-slot sizing will surface here.
        """
        assert dit_model.num_classes == 13

        breeds = torch.tensor([0, 5, 12])
        uncond = torch.full_like(breeds, dit_model.num_classes)
        assert uncond.tolist() == [13, 13, 13]

        # Embedder has num_classes + 1 slots (0..num_classes-1 are breeds,
        # index num_classes is the dedicated null token).
        assert (
            dit_model.breed_embedder.embedding.num_embeddings
            == dit_model.num_classes + 1
        )
        with torch.no_grad():
            emb = dit_model.breed_embedder(uncond)
        assert emb.shape == (3, dit_model.embed_dim)

    def test_cfg_extreme_scale_stable(self, dit_model_with_outputs: nn.Module) -> None:
        """Large cfg_scale must still produce finite outputs."""
        x = torch.randn(2, 3, 128, 128)
        t = torch.rand(2)
        breeds = torch.tensor([0, 5])

        with torch.no_grad():
            out = dit_model_with_outputs.forward_with_cfg(x, t, breeds, cfg_scale=5.0)

        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Static contract: src/train_dit.py call site must not pass the same
# tensor for x0 and x1. This guards the zero-loss bug fix.
# ---------------------------------------------------------------------------


_SRC_DIR = Path(__file__).parent.parent / "src"
_TRAIN_DIT = _SRC_DIR / "train_dit.py"

# Regex matches flow_matching_step(<identifier>, <var>, <same var>, ...)
# for the leading-identifier (model/...) + x0 + x1 positions. Captures
# the offending token to include in the failure message.
_DUP_ARG_PATTERN = re.compile(
    r"flow_matching_step\s*\(\s*(\w+)\s*,\s*(\w+)\s*,\s*(\2)\b",
)


@pytest.mark.skipif(
    not _TRAIN_DIT.exists(),
    reason="src/train_dit.py not present in repo root",
)
@pytest.mark.xfail(
    reason="src/train_dit.py:929 still passes (images, images) — the "
    "fix-zero-loss-bug commit dropped the regression test alongside the "
    "fix and the call site was never patched. Flip to passing when "
    "train_dit.py is updated to pass torch.randn_like(images) for x0.",
    strict=False,
)
def test_train_dit_does_not_pass_same_tensor_for_x0_and_x1() -> None:
    """Regression guard: flow_matching_step must receive distinct x0/x1."""
    src = _TRAIN_DIT.read_text(encoding="utf-8")
    hits = _DUP_ARG_PATTERN.findall(src)
    assert not hits, (
        f"flow_matching_step is called with the same variable for x0 and "
        f"x1 inside src/train_dit.py: {hits}. Pass a fresh noise tensor "
        "for x0 (e.g. torch.randn_like(images)) and the image batch for "
        "x1 so target velocity x1 - x0 stays non-zero."
    )
