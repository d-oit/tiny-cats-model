"""Tests for the training module."""
import os
import tempfile
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from model import cats_model
from train import (
    get_optimizer,
    get_scheduler,
    train_one_epoch,
    validate,
    save_checkpoint,
    load_checkpoint,
)


@pytest.fixture
def simple_model():
    """Small resnet18-based model for testing."""
    return cats_model(num_classes=2, backbone="resnet18", pretrained=False)


@pytest.fixture
def tiny_loader():
    """Tiny dataloader with synthetic data."""
    x = torch.randn(8, 3, 224, 224)
    y = torch.randint(0, 2, (8,))
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=4)


class TestGetOptimizer:
    """Tests for get_optimizer."""

    def test_returns_adam(self, simple_model):
        opt = get_optimizer(simple_model, optimizer_name="adam", lr=1e-3)
        assert isinstance(opt, torch.optim.Adam)

    def test_returns_sgd(self, simple_model):
        opt = get_optimizer(simple_model, optimizer_name="sgd", lr=1e-2)
        assert isinstance(opt, torch.optim.SGD)

    def test_returns_adamw(self, simple_model):
        opt = get_optimizer(simple_model, optimizer_name="adamw", lr=1e-3)
        assert isinstance(opt, torch.optim.AdamW)

    def test_unsupported_optimizer_raises(self, simple_model):
        with pytest.raises(ValueError):
            get_optimizer(simple_model, optimizer_name="rmsprop", lr=1e-3)

    def test_lr_is_set(self, simple_model):
        lr = 5e-4
        opt = get_optimizer(simple_model, optimizer_name="adam", lr=lr)
        assert opt.param_groups[0]["lr"] == lr


class TestGetScheduler:
    """Tests for get_scheduler."""

    def test_returns_steplr(self, simple_model):
        opt = get_optimizer(simple_model, optimizer_name="adam", lr=1e-3)
        sched = get_scheduler(opt, scheduler_name="steplr", step_size=5)
        assert sched is not None

    def test_returns_cosine(self, simple_model):
        opt = get_optimizer(simple_model, optimizer_name="adam", lr=1e-3)
        sched = get_scheduler(opt, scheduler_name="cosine", T_max=10)
        assert sched is not None

    def test_none_scheduler_returns_none(self, simple_model):
        opt = get_optimizer(simple_model, optimizer_name="adam", lr=1e-3)
        sched = get_scheduler(opt, scheduler_name=None)
        assert sched is None


class TestTrainOneEpoch:
    """Tests for train_one_epoch."""

    def test_returns_float_loss(self, simple_model, tiny_loader):
        opt = get_optimizer(simple_model, optimizer_name="adam", lr=1e-3)
        loss = train_one_epoch(simple_model, tiny_loader, opt, device="cpu")
        assert isinstance(loss, float)

    def test_loss_is_positive(self, simple_model, tiny_loader):
        opt = get_optimizer(simple_model, optimizer_name="adam", lr=1e-3)
        loss = train_one_epoch(simple_model, tiny_loader, opt, device="cpu")
        assert loss > 0

    def test_model_params_change_after_train(self, simple_model, tiny_loader):
        opt = get_optimizer(simple_model, optimizer_name="adam", lr=1e-3)
        params_before = [p.clone() for p in simple_model.parameters()]
        train_one_epoch(simple_model, tiny_loader, opt, device="cpu")
        params_after = list(simple_model.parameters())
        changed = any(
            not torch.equal(b, a) for b, a in zip(params_before, params_after)
        )
        assert changed


class TestValidate:
    """Tests for validate."""

    def test_returns_loss_and_accuracy(self, simple_model, tiny_loader):
        result = validate(simple_model, tiny_loader, device="cpu")
        assert "loss" in result
        assert "accuracy" in result

    def test_accuracy_between_0_and_1(self, simple_model, tiny_loader):
        result = validate(simple_model, tiny_loader, device="cpu")
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_loss_is_positive(self, simple_model, tiny_loader):
        result = validate(simple_model, tiny_loader, device="cpu")
        assert result["loss"] > 0


class TestCheckpointing:
    """Tests for save_checkpoint and load_checkpoint."""

    def test_save_and_load_checkpoint(self, simple_model):
        opt = get_optimizer(simple_model, optimizer_name="adam", lr=1e-3)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ckpt.pth")
            save_checkpoint(simple_model, opt, epoch=1, path=path)
            assert os.path.exists(path)
            new_model = cats_model(num_classes=2, backbone="resnet18", pretrained=False)
            new_opt = get_optimizer(new_model, optimizer_name="adam", lr=1e-3)
            epoch = load_checkpoint(new_model, new_opt, path=path)
            assert epoch == 1

    def test_checkpoint_file_created(self, simple_model):
        opt = get_optimizer(simple_model, optimizer_name="adam", lr=1e-3)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pth")
            save_checkpoint(simple_model, opt, epoch=3, path=path)
            assert os.path.isfile(path)
            assert os.path.getsize(path) > 0
