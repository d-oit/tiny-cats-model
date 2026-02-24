"""Tests for the training module."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model import cats_model
from train import train_one_epoch, validate


@pytest.fixture
def logger():
    """Create a test logger."""
    logger = logging.getLogger("test_logger")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    return logger


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


class TestTrainOneEpoch:
    """Tests for train_one_epoch."""

    def test_returns_loss_and_accuracy(self, simple_model, tiny_loader, logger):
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

    def test_loss_is_positive(self, simple_model, tiny_loader, logger):
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

    def test_model_params_change_after_train(self, simple_model, tiny_loader, logger):
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


class TestValidate:
    """Tests for validate."""

    def test_returns_loss_and_accuracy(self, simple_model, tiny_loader, logger):
        loss_fn = nn.CrossEntropyLoss()
        device = torch.device("cpu")
        val_loss, val_acc = validate(simple_model, tiny_loader, loss_fn, device, logger)
        assert isinstance(val_loss, float)
        assert isinstance(val_acc, float)

    def test_accuracy_between_0_and_1(self, simple_model, tiny_loader, logger):
        loss_fn = nn.CrossEntropyLoss()
        device = torch.device("cpu")
        _, val_acc = validate(simple_model, tiny_loader, loss_fn, device, logger)
        assert 0.0 <= val_acc <= 1.0

    def test_loss_is_positive(self, simple_model, tiny_loader, logger):
        loss_fn = nn.CrossEntropyLoss()
        device = torch.device("cpu")
        val_loss, _ = validate(simple_model, tiny_loader, loss_fn, device, logger)
        assert val_loss > 0
