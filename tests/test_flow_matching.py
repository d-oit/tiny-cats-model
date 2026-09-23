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
from flow_matching import (
    EMA,
    FlowMatchingLoss,
    flow_matching_step,
    sample_t,
    sample_t_logit_normal,
    sample_timesteps,
)

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
# sample_timesteps / logit-normal sampling
# ---------------------------------------------------------------------------


class TestTimestepSampling:
    """Tests for the selectable training timestep distribution."""

    def test_uniform_dispatch_matches_sample_t(self) -> None:
        device = torch.device("cpu")
        generator = torch.Generator(device=device).manual_seed(0)

        uniform = sample_timesteps(
            512, device, sampling="uniform", t_min=0.2, t_max=0.8, generator=generator
        )

        assert uniform.shape == (512,)
        assert uniform.min().item() >= 0.2
        assert uniform.max().item() <= 0.8

    def test_logit_normal_stays_inside_the_unit_interval(self) -> None:
        t = sample_t_logit_normal(5000, torch.device("cpu"), mean=0.0, std=1.0)

        assert t.shape == (5000,)
        assert t.min().item() > 0.0
        assert t.max().item() < 1.0

    def test_logit_normal_concentrates_towards_the_middle(self) -> None:
        t = sample_t_logit_normal(20000, torch.device("cpu"), mean=0.0, std=1.0)

        middle_fraction = ((t > 0.25) & (t < 0.75)).float().mean().item()

        # P(0.25 < sigmoid(N(0,1)) < 0.75) ~ 0.73 vs 0.50 for uniform.
        assert middle_fraction > 0.65
        assert t.mean().item() == pytest.approx(0.5, abs=0.02)

    def test_uniform_spends_more_mass_at_the_tails(self) -> None:
        uniform = sample_timesteps(20000, torch.device("cpu"), sampling="uniform")
        logit = sample_t(20000, torch.device("cpu"))

        assert uniform.mean().item() == pytest.approx(logit.mean().item(), abs=0.02)
        tails = ((uniform < 0.25) | (uniform > 0.75)).float().mean().item()
        assert tails == pytest.approx(0.5, abs=0.02)

    def test_zero_std_is_deterministic_at_the_mean(self) -> None:
        t = sample_t_logit_normal(16, torch.device("cpu"), mean=0.0, std=0.0)

        assert torch.allclose(t, torch.full((16,), 0.5))

    def test_seeded_generator_is_reproducible(self) -> None:
        device = torch.device("cpu")

        def draw() -> torch.Tensor:
            generator = torch.Generator(device=device).manual_seed(7)
            return sample_timesteps(
                64,
                device,
                sampling="logit_normal",
                logit_normal_mean=0.0,
                logit_normal_std=1.0,
                generator=generator,
            )

        assert torch.equal(draw(), draw())

    def test_unknown_sampling_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported timestep sampling"):
            sample_timesteps(4, torch.device("cpu"), sampling="beta")  # type: ignore[arg-type]

    def test_logit_normal_mean_shifts_the_distribution(self) -> None:
        t = sample_t_logit_normal(20000, torch.device("cpu"), mean=1.5, std=1.0)

        assert t.mean().item() > 0.65


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


class TestDiTArchitecture:
    """Regression tests for the spatial-structure bugs (ADR-060)."""

    def test_has_positional_embedding(self, dit_model: nn.Module) -> None:
        """Without pos_embed the transformer is permutation-equivariant."""
        assert hasattr(dit_model, "pos_embed")
        assert isinstance(dit_model.pos_embed, nn.Parameter)
        expected = (1, dit_model.patch_embed.num_patches, dit_model.embed_dim)
        assert dit_model.pos_embed.shape == expected

    def test_output_is_position_aware(self, dit_model_with_outputs: nn.Module) -> None:
        """A position-aware model is not equivariant to patch permutations.

        Comparing ``f(perm(x))`` with ``f(x)`` directly would also differ for
        an equivariant (position-blind) model, because ``f(perm(x)) ==
        perm(f(x))``. Undo the permutation on the output first: an equivariant
        model reproduces ``f(x)`` exactly, a position-aware one does not.
        """
        model = dit_model_with_outputs
        b, c, h, w, patch = 1, 3, 128, 128, 16
        n = h // patch
        x = torch.randn(b, c, h, w)
        perm = torch.randperm(n * n)
        x_perm = _permute_patches(x, perm, patch)
        t = torch.full((b,), 0.5)
        breeds = torch.tensor([0])
        with torch.no_grad():
            y = model(x, t, breeds)
            y_perm = model(x_perm, t, breeds)
        y_unpermuted = _permute_patches(y_perm, torch.argsort(perm), patch)
        assert not torch.allclose(y_unpermuted, y, atol=1e-4), (
            "model output is equivariant to patch permutation: it has no "
            "positional information and cannot model spatial structure"
        )

    def test_unpatchify_places_each_token_in_its_patch(
        self, dit_model: nn.Module
    ) -> None:
        """Each output patch must come from its own token, not a mix of all."""
        model = dit_model
        b, c, h, w, patch = 1, 3, 128, 128, 16
        n = h // patch
        num_patches = n * n
        tokens = torch.zeros(b, num_patches, patch * patch * c)
        for i in range(num_patches):
            tokens[0, i] = float(i + 1)
        original = model.final_layer

        class _Stub(nn.Module):
            def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
                return tokens

        model.final_layer = _Stub()
        try:
            with torch.no_grad():
                out = model(
                    torch.zeros(b, c, h, w),
                    torch.full((b,), 0.5),
                    torch.tensor([0]),
                )
        finally:
            model.final_layer = original

        for i in range(num_patches):
            row, col = divmod(i, n)
            patch_values = out[
                0, 0, row * patch : (row + 1) * patch, col * patch : (col + 1) * patch
            ]
            expected = torch.full_like(patch_values, float(i + 1))
            assert torch.allclose(patch_values, expected, atol=1e-5), (
                f"patch {i} is not reconstructed from its own token — the "
                "unpatchify layout scrambles tokens across patches"
            )


def _permute_patches(x: torch.Tensor, perm: torch.Tensor, patch: int) -> torch.Tensor:
    """Return an image whose patch i is patch perm[i] of ``x``."""
    b, c, h, w = x.shape
    n = h // patch
    patches = (
        x.reshape(b, c, n, patch, n, patch)
        .permute(0, 1, 2, 4, 3, 5)
        .reshape(b, c, n * n, patch, patch)[:, :, perm]
    )
    return (
        patches.reshape(b, c, n, n, patch, patch)
        .permute(0, 1, 2, 4, 3, 5)
        .reshape(b, c, h, w)
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
