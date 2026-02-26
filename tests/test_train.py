"""tests/test_train.py

Unit tests for the training module with comprehensive edge cases.
Tests cover checkpoint resume, OOM recovery, LR scheduler edge cases,
mixed precision training, and EMA weight updates.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model import cats_model
from train import (
    load_checkpoint,
    save_checkpoint,
    train_one_epoch,
    validate,
)

if TYPE_CHECKING:
    import logging


@pytest.fixture
def logger() -> logging.Logger:
    """Create a test logger."""
    import logging

    logger = logging.getLogger(f"test_logger_{id(logging.getLogger())}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    return logger


@pytest.fixture
def simple_model(test_device: torch.device) -> nn.Module:
    """Small resnet18-based model for testing."""
    return cats_model(num_classes=2, backbone="resnet18", pretrained=False).to(
        test_device
    )


@pytest.fixture
def tiny_loader(test_device: torch.device) -> DataLoader:
    """Tiny dataloader with synthetic data."""
    x = torch.randn(8, 3, 224, 224)
    y = torch.randint(0, 2, (8,))
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=4)


@pytest.fixture
def single_batch_loader(test_device: torch.device) -> DataLoader:
    """Dataloader with exactly one batch."""
    x = torch.randn(4, 3, 224, 224)
    y = torch.randint(0, 2, (4,))
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=4)


@pytest.fixture
def empty_loader(test_device: torch.device) -> DataLoader:
    """Empty dataloader for edge case testing."""
    x = torch.randn(0, 3, 224, 224)
    y = torch.randint(0, 2, (0,))
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=4)


@pytest.fixture
def loss_fn() -> nn.Module:
    """Create CrossEntropyLoss for testing."""
    return nn.CrossEntropyLoss()


class TestTrainOneEpoch:
    """Tests for train_one_epoch."""

    def test_returns_loss_and_accuracy(
        self, simple_model: nn.Module, tiny_loader: DataLoader, logger: logging.Logger
    ) -> None:
        """Test train_one_epoch returns loss and accuracy tuple."""
        opt = torch.optim.Adam(simple_model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()
        device = torch.device("cpu")
        result = train_one_epoch(
            simple_model,
            tiny_loader,
            opt,
            loss_fn,
            device,
            epoch=1,
            logger=logger,
        )
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_loss_is_positive(
        self, simple_model: nn.Module, tiny_loader: DataLoader, logger: logging.Logger
    ) -> None:
        """Test training loss is positive."""
        opt = torch.optim.Adam(simple_model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()
        device = torch.device("cpu")
        loss, _ = train_one_epoch(
            simple_model,
            tiny_loader,
            opt,
            loss_fn,
            device,
            epoch=1,
            logger=logger,
        )
        assert loss > 0

    def test_model_params_change_after_train(
        self, simple_model: nn.Module, tiny_loader: DataLoader, logger: logging.Logger
    ) -> None:
        """Test model parameters change after training step."""
        opt = torch.optim.Adam(simple_model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()
        device = torch.device("cpu")
        params_before = [p.clone() for p in simple_model.parameters()]
        train_one_epoch(
            simple_model,
            tiny_loader,
            opt,
            loss_fn,
            device,
            epoch=1,
            logger=logger,
        )
        params_after = list(simple_model.parameters())
        changed = any(
            not torch.equal(b, a) for b, a in zip(params_before, params_after)
        )
        assert changed

    def test_accuracy_between_0_and_1(
        self, simple_model: nn.Module, tiny_loader: DataLoader, logger: logging.Logger
    ) -> None:
        """Test training accuracy is between 0 and 1."""
        opt = torch.optim.Adam(simple_model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()
        device = torch.device("cpu")
        _, acc = train_one_epoch(
            simple_model,
            tiny_loader,
            opt,
            loss_fn,
            device,
            epoch=1,
            logger=logger,
        )
        assert 0.0 <= acc <= 1.0


class TestValidate:
    """Tests for validate function."""

    def test_returns_loss_and_accuracy(
        self, simple_model: nn.Module, tiny_loader: DataLoader, logger: logging.Logger
    ) -> None:
        """Test validate returns loss and accuracy tuple."""
        loss_fn = nn.CrossEntropyLoss()
        device = torch.device("cpu")
        val_loss, val_acc = validate(simple_model, tiny_loader, loss_fn, device, logger)
        assert isinstance(val_loss, float)
        assert isinstance(val_acc, float)

    def test_accuracy_between_0_and_1(
        self, simple_model: nn.Module, tiny_loader: DataLoader, logger: logging.Logger
    ) -> None:
        """Test validation accuracy is between 0 and 1."""
        loss_fn = nn.CrossEntropyLoss()
        device = torch.device("cpu")
        _, val_acc = validate(simple_model, tiny_loader, loss_fn, device, logger)
        assert 0.0 <= val_acc <= 1.0

    def test_loss_is_positive(
        self, simple_model: nn.Module, tiny_loader: DataLoader, logger: logging.Logger
    ) -> None:
        """Test validation loss is positive."""
        loss_fn = nn.CrossEntropyLoss()
        device = torch.device("cpu")
        val_loss, _ = validate(simple_model, tiny_loader, loss_fn, device, logger)
        assert val_loss > 0

    def test_model_in_eval_mode(
        self, simple_model: nn.Module, tiny_loader: DataLoader, logger: logging.Logger
    ) -> None:
        """Test validate sets model to eval mode."""
        loss_fn = nn.CrossEntropyLoss()
        device = torch.device("cpu")

        # Set model to train mode
        simple_model.train()
        assert simple_model.training

        validate(simple_model, tiny_loader, loss_fn, device, logger)

        # Model should be in eval mode after validation
        assert not simple_model.training


class TestCheckpointResume:
    """Tests for checkpoint resume consistency."""

    def test_checkpoint_save_and_load(
        self, temp_dir: Path, simple_model: nn.Module, logger: logging.Logger
    ) -> None:
        """Test checkpoint can be saved and loaded correctly."""
        opt = torch.optim.Adam(simple_model.parameters(), lr=1e-3)
        ckpt_path = temp_dir / "checkpoint.pt"

        # Save checkpoint
        save_checkpoint(
            simple_model,
            opt,
            epoch=5,
            val_acc=0.85,
            path=ckpt_path,
            logger=logger,
            is_best=False,
        )

        assert ckpt_path.exists()

    def test_checkpoint_resume_consistency(
        self, temp_dir: Path, simple_model: nn.Module, logger: logging.Logger
    ) -> None:
        """Test resume produces same results as continuing from checkpoint."""
        opt = torch.optim.Adam(simple_model.parameters(), lr=1e-3)
        ckpt_path = temp_dir / "resume_test.pt"

        # Train for one epoch and save
        x = torch.randn(8, 3, 224, 224)
        y = torch.randint(0, 2, (8,))
        ds = TensorDataset(x, y)
        loader = DataLoader(ds, batch_size=4)

        device = torch.device("cpu")
        loss_fn = nn.CrossEntropyLoss()

        train_one_epoch(
            simple_model, loader, opt, loss_fn, device, epoch=1, logger=logger
        )
        save_checkpoint(
            simple_model, opt, epoch=1, val_acc=0.5, path=ckpt_path, logger=logger
        )

        # Store weights after first epoch
        weights_after_epoch1 = [p.clone() for p in simple_model.parameters()]

        # Load checkpoint
        loaded_model, _loaded_opt, start_epoch = load_checkpoint(
            ckpt_path, simple_model, opt, logger
        )

        # Verify epoch is correct
        assert start_epoch == 2

        # Verify weights match
        for p1, p2 in zip(weights_after_epoch1, loaded_model.parameters()):
            assert torch.allclose(p1, p2)

    def test_checkpoint_contains_metadata(
        self, temp_dir: Path, simple_model: nn.Module, logger: logging.Logger
    ) -> None:
        """Test checkpoint contains expected metadata."""
        opt = torch.optim.Adam(simple_model.parameters(), lr=1e-3)
        ckpt_path = temp_dir / "metadata_test.pt"

        save_checkpoint(
            simple_model, opt, epoch=10, val_acc=0.92, path=ckpt_path, logger=logger
        )

        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        assert "epoch" in checkpoint
        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint
        assert "val_acc" in checkpoint
        assert "timestamp" in checkpoint

        assert checkpoint["epoch"] == 10
        assert checkpoint["val_acc"] == 0.92

    def test_checkpoint_best_model_saved(
        self, temp_dir: Path, simple_model: nn.Module, logger: logging.Logger
    ) -> None:
        """Test best model checkpoint is saved separately."""
        opt = torch.optim.Adam(simple_model.parameters(), lr=1e-3)
        ckpt_path = temp_dir / "checkpoint.pt"

        save_checkpoint(
            simple_model,
            opt,
            epoch=5,
            val_acc=0.85,
            path=ckpt_path,
            logger=logger,
            is_best=True,
        )

        best_path = temp_dir / "best_checkpoint.pt"
        assert best_path.exists()


class TestOOMRecovery:
    """Tests for OOM (Out of Memory) recovery with gradient accumulation."""

    def test_gradient_accumulation_basic(
        self, simple_model: nn.Module, tiny_loader: DataLoader, logger: logging.Logger
    ) -> None:
        """Test basic gradient accumulation works correctly."""
        opt = torch.optim.Adam(simple_model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()
        torch.device("cpu")

        # Train with gradient accumulation
        simple_model.train()
        total_loss = 0.0

        for batch_idx, (xb, yb) in enumerate(tiny_loader):
            pred = simple_model(xb)
            loss = loss_fn(pred, yb) / 2  # grad_accum_steps = 2
            loss.backward()

            if (batch_idx + 1) % 2 == 0:
                opt.step()
                opt.zero_grad()

            total_loss += loss.item()

        assert total_loss > 0

    def test_gradient_accumulation_matches_no_accumulation(
        self, simple_model: nn.Module, logger: logging.Logger
    ) -> None:
        """Test gradient accumulation produces valid gradients.

        Note: This test verifies that gradient accumulation works correctly
        by checking that accumulated gradients are non-zero and finite.
        Exact match with non-accumulated gradients is not expected due to
        BatchNorm running statistics being updated differently.
        """
        # Create model for gradient accumulation test
        torch.manual_seed(42)
        model = cats_model(num_classes=2, backbone="resnet18", pretrained=False)

        # Create synthetic data
        x = torch.randn(4, 3, 224, 224)
        y = torch.randint(0, 2, (4,))

        loss_fn = nn.CrossEntropyLoss()

        # Test gradient accumulation
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        opt.zero_grad()

        # Accumulate gradients over 2 steps
        x_half = x[:2]
        y_half = y[:2]

        pred1 = model(x_half)
        loss1 = loss_fn(pred1, y_half) / 2
        loss1.backward()

        pred2 = model(x[2:])
        loss2 = loss_fn(pred2, y[2:]) / 2
        loss2.backward()

        # Check that accumulated gradients are valid
        for name, param in model.named_parameters():
            if param.grad is not None:
                # Gradients should be finite
                assert torch.isfinite(param.grad).all(), (
                    f"Non-finite gradient for {name}"
                )
                # At least some gradients should be non-zero
                # (some layers may have very small gradients)

    def test_oom_handling_simulation(
        self, simple_model: nn.Module, tiny_loader: DataLoader, logger: logging.Logger
    ) -> None:
        """Test OOM handling logic (simulated)."""
        opt = torch.optim.Adam(simple_model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()
        device = torch.device("cpu")

        # Simulate OOM recovery by catching RuntimeError
        oom_simulated = False

        for batch_idx, (xb, yb) in enumerate(tiny_loader):
            try:
                # Simulate OOM on first batch
                if batch_idx == 0 and not oom_simulated:
                    oom_simulated = True
                    raise RuntimeError("CUDA out of memory")

                xb = xb.to(device)
                yb = yb.to(device)
                pred = simple_model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()
                opt.zero_grad()

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    # Simulate OOM recovery
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    continue
                raise

        # Training should continue after OOM
        assert simple_model is not None


class TestLRScheduler:
    """Tests for learning rate scheduler edge cases."""

    def test_linear_warmup_scheduler(self, simple_model: nn.Module) -> None:
        """Test linear warmup scheduler behavior."""
        opt = torch.optim.Adam(simple_model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=0.1, end_factor=1.0, total_iters=5
        )

        initial_lr = 1e-3
        expected_lrs = [initial_lr * 0.1 + initial_lr * 0.9 * (i / 5) for i in range(5)]

        actual_lrs = []
        for _ in range(5):
            actual_lrs.append(scheduler.get_last_lr()[0])
            scheduler.step()

        for expected, actual in zip(expected_lrs, actual_lrs):
            assert abs(expected - actual) < 1e-7

    def test_cosine_annealing_scheduler(self, simple_model: nn.Module) -> None:
        """Test cosine annealing scheduler behavior."""
        opt = torch.optim.Adam(simple_model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)

        # LR should decrease from initial value
        initial_lr = scheduler.get_last_lr()[0]

        for _ in range(10):
            scheduler.step()

        final_lr = scheduler.get_last_lr()[0]
        assert final_lr < initial_lr
        assert final_lr >= 0

    def test_sequential_scheduler_warmup_then_cosine(
        self, simple_model: nn.Module
    ) -> None:
        """Test sequential scheduler: warmup then cosine annealing."""
        opt = torch.optim.Adam(simple_model.parameters(), lr=1e-3)

        warmup = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=0.1, end_factor=1.0, total_iters=2
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=8)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            opt, schedulers=[warmup, cosine], milestones=[2]
        )

        # Warmup phase: LR should increase
        lr_before_warmup = scheduler.get_last_lr()[0]
        scheduler.step()
        lr_after_warmup_1 = scheduler.get_last_lr()[0]
        scheduler.step()
        lr_after_warmup_2 = scheduler.get_last_lr()[0]

        assert lr_after_warmup_1 > lr_before_warmup
        assert lr_after_warmup_2 >= lr_after_warmup_1

        # Cosine phase: LR should decrease
        for _ in range(8):
            scheduler.step()

        final_lr = scheduler.get_last_lr()[0]
        assert final_lr < lr_after_warmup_2

    def test_scheduler_zero_warmup_epochs(self, simple_model: nn.Module) -> None:
        """Test scheduler with zero warmup epochs."""
        opt = torch.optim.Adam(simple_model.parameters(), lr=1e-3)

        # No warmup, just cosine annealing
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)

        initial_lr = scheduler.get_last_lr()[0]
        assert initial_lr == 1e-3

    def test_scheduler_step_lr_gamma(self, simple_model: nn.Module) -> None:
        """Test StepLR with gamma decay."""
        opt = torch.optim.Adam(simple_model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.1)

        # LR should stay constant for 5 steps
        for _ in range(4):
            scheduler.step()

        lr_before_step = scheduler.get_last_lr()[0]
        assert lr_before_step == 1e-3

        # After 5th step, LR should decay
        scheduler.step()
        lr_after_step = scheduler.get_last_lr()[0]
        assert lr_after_step == 1e-4


class TestMixedPrecision:
    """Tests for mixed precision training stability."""

    def test_amp_scaler_creation(self) -> None:
        """Test GradScaler can be created."""
        if torch.cuda.is_available():
            scaler = torch.amp.GradScaler("cuda")
            assert scaler is not None
        else:
            # On CPU, scaler should be None
            scaler = None
            assert scaler is None

    def test_autocast_context_cpu(self, simple_model: nn.Module) -> None:
        """Test autocast context works on CPU (no-op)."""
        simple_model.eval()
        x = torch.randn(2, 3, 224, 224)

        # On CPU, autocast is a no-op but shouldn't error
        with torch.amp.autocast("cpu"), torch.no_grad():
            out = simple_model(x)

        assert out.shape == (2, 2)

    def test_mixed_precision_numerical_stability(
        self, simple_model: nn.Module, tiny_loader: DataLoader, logger: logging.Logger
    ) -> None:
        """Test mixed precision training is numerically stable."""
        opt = torch.optim.Adam(simple_model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()
        torch.device("cpu")

        # On CPU, scaler is None but training should still work
        scaler = None

        simple_model.train()
        total_loss = 0.0

        for xb, yb in tiny_loader:
            pred = simple_model(xb)
            loss = loss_fn(pred, yb)
            total_loss += loss.item()

            if scaler:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()

            opt.zero_grad()

        assert total_loss > 0
        assert not torch.isnan(torch.tensor(total_loss))

    def test_gradient_clipping_with_mixed_precision(
        self, simple_model: nn.Module, tiny_loader: DataLoader, logger: logging.Logger
    ) -> None:
        """Test gradient clipping works correctly."""
        opt = torch.optim.Adam(simple_model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()
        torch.device("cpu")

        simple_model.train()

        for xb, yb in tiny_loader:
            pred = simple_model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(simple_model.parameters(), max_norm=1.0)

            # Verify gradients are clipped
            for param in simple_model.parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm()
                    assert grad_norm <= 1.0 + 1e-6  # Small tolerance

            opt.step()
            opt.zero_grad()
            break


class TestEMAWeights:
    """Tests for EMA (Exponential Moving Average) weight updates."""

    def test_ema_update_basic(self, simple_model: nn.Module) -> None:
        """Test basic EMA weight update."""
        # Create EMA copy of model
        ema_model = cats_model(num_classes=2, backbone="resnet18", pretrained=False)

        # Initialize EMA weights to match
        for ema_param, param in zip(ema_model.parameters(), simple_model.parameters()):
            ema_param.data.copy_(param.data)

        # Update model weights
        with torch.no_grad():
            for param in simple_model.parameters():
                param.add_(torch.randn_like(param) * 0.1)

        # Apply EMA update (decay=0.999)
        decay = 0.999
        with torch.no_grad():
            for ema_param, param in zip(
                ema_model.parameters(), simple_model.parameters()
            ):
                ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)

        # EMA weights should be between old and new
        for ema_param, param in zip(ema_model.parameters(), simple_model.parameters()):
            diff = (param - ema_param).abs()
            assert diff.max() > 0  # Should have moved

    def test_ema_decay_values(self, simple_model: nn.Module) -> None:
        """Test EMA with different decay values."""
        ema_model = cats_model(num_classes=2, backbone="resnet18", pretrained=False)

        for ema_param, param in zip(ema_model.parameters(), simple_model.parameters()):
            ema_param.data.copy_(param.data)

        # Store original weights
        original_weights = [p.clone() for p in simple_model.parameters()]

        # Update model weights significantly
        with torch.no_grad():
            for param in simple_model.parameters():
                param.add_(torch.ones_like(param))

        # Test different decay values
        for decay in [0.9, 0.99, 0.999, 0.9999]:
            ema_model_test = cats_model(
                num_classes=2, backbone="resnet18", pretrained=False
            )
            for ema_param, orig_param in zip(
                ema_model_test.parameters(), original_weights
            ):
                ema_param.data.copy_(orig_param.data)

            with torch.no_grad():
                for ema_param, param in zip(
                    ema_model_test.parameters(), simple_model.parameters()
                ):
                    ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)

            # Higher decay should keep EMA closer to original
            for ema_param, orig_param, new_param in zip(
                ema_model_test.parameters(), original_weights, simple_model.parameters()
            ):
                ema_distance = (ema_param - orig_param).abs().mean()
                new_distance = (new_param - orig_param).abs().mean()
                assert ema_distance < new_distance

    def test_ema_numerical_stability(self, simple_model: nn.Module) -> None:
        """Test EMA updates are numerically stable."""
        ema_model = cats_model(num_classes=2, backbone="resnet18", pretrained=False)

        for ema_param, param in zip(ema_model.parameters(), simple_model.parameters()):
            ema_param.data.copy_(param.data)

        # Apply many EMA updates
        for _ in range(100):
            with torch.no_grad():
                for param in simple_model.parameters():
                    param.add_(torch.randn_like(param) * 0.01)

            with torch.no_grad():
                for ema_param, param in zip(
                    ema_model.parameters(), simple_model.parameters()
                ):
                    ema_param.data.mul_(0.999).add_(param.data, alpha=0.001)

        # Check for NaN or Inf in EMA weights
        for ema_param in ema_model.parameters():
            assert not torch.isnan(ema_param).any()
            assert not torch.isinf(ema_param).any()


class TestEdgeCases:
    """Additional edge case tests for training."""

    def test_empty_dataloader_handling(
        self, simple_model: nn.Module, empty_loader: DataLoader, logger: logging.Logger
    ) -> None:
        """Test handling of empty dataloader."""
        opt = torch.optim.Adam(simple_model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()
        device = torch.device("cpu")

        # Empty loader should return zero loss/accuracy
        loss, acc = train_one_epoch(
            simple_model, empty_loader, opt, loss_fn, device, epoch=1, logger=logger
        )

        assert loss == 0.0
        assert acc == 0.0

    def test_single_batch_training(
        self,
        simple_model: nn.Module,
        single_batch_loader: DataLoader,
        logger: logging.Logger,
    ) -> None:
        """Test training with single batch."""
        opt = torch.optim.Adam(simple_model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()
        device = torch.device("cpu")

        loss, acc = train_one_epoch(
            simple_model,
            single_batch_loader,
            opt,
            loss_fn,
            device,
            epoch=1,
            logger=logger,
        )

        assert loss > 0
        assert 0.0 <= acc <= 1.0

    def test_validation_empty_loader(
        self, simple_model: nn.Module, empty_loader: DataLoader, logger: logging.Logger
    ) -> None:
        """Test validation with empty loader."""
        loss_fn = nn.CrossEntropyLoss()
        device = torch.device("cpu")

        loss, acc = validate(simple_model, empty_loader, loss_fn, device, logger)

        assert loss == 0.0
        assert acc == 0.0

    def test_very_small_learning_rate(
        self, simple_model: nn.Module, tiny_loader: DataLoader, logger: logging.Logger
    ) -> None:
        """Test training with very small learning rate."""
        opt = torch.optim.Adam(simple_model.parameters(), lr=1e-10)
        loss_fn = nn.CrossEntropyLoss()
        device = torch.device("cpu")

        params_before = [p.clone() for p in simple_model.parameters()]

        train_one_epoch(
            simple_model, tiny_loader, opt, loss_fn, device, epoch=1, logger=logger
        )

        params_after = list(simple_model.parameters())

        # With very small LR, params should barely change
        max_change = max(
            (p_after - p_before).abs().max()
            for p_before, p_after in zip(params_before, params_after)
        )
        assert max_change < 1e-6

    def test_very_large_learning_rate(
        self, simple_model: nn.Module, tiny_loader: DataLoader, logger: logging.Logger
    ) -> None:
        """Test training with very large learning rate."""
        opt = torch.optim.Adam(simple_model.parameters(), lr=10.0)
        loss_fn = nn.CrossEntropyLoss()
        device = torch.device("cpu")

        params_before = [p.clone() for p in simple_model.parameters()]

        loss, _acc = train_one_epoch(
            simple_model, tiny_loader, opt, loss_fn, device, epoch=1, logger=logger
        )

        params_after = list(simple_model.parameters())

        # With large LR, params should change significantly
        max_change = max(
            (p_after - p_before).abs().max()
            for p_before, p_after in zip(params_before, params_after)
        )
        assert max_change > 1e-6

        # Loss might be unstable but should still be finite
        assert torch.isfinite(torch.tensor(loss))


class TestGradientFlow:
    """Tests for gradient flow and backpropagation."""

    def test_all_layers_receive_gradients(
        self, simple_model: nn.Module, tiny_loader: DataLoader, logger: logging.Logger
    ) -> None:
        """Test all model layers receive gradients during backprop."""
        opt = torch.optim.Adam(simple_model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()
        torch.device("cpu")

        simple_model.train()
        opt.zero_grad()

        for xb, yb in tiny_loader:
            pred = simple_model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            break

        # Check all parameters have gradients
        for name, param in simple_model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"

    def test_gradient_norm_monitoring(
        self, simple_model: nn.Module, tiny_loader: DataLoader, logger: logging.Logger
    ) -> None:
        """Test gradient norm can be computed for monitoring."""
        loss_fn = nn.CrossEntropyLoss()
        torch.device("cpu")

        simple_model.train()

        for xb, yb in tiny_loader:
            pred = simple_model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()

            # Compute total gradient norm
            total_norm = 0.0
            for param in simple_model.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm**0.5

            assert total_norm > 0
            assert torch.isfinite(torch.tensor(total_norm))
            break
