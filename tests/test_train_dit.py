"""tests/test_train_dit.py

Regression coverage for the DiT training-loop fixes:

- ``--config`` YAML actually applies (it was silently a no-op because the old
  code only overwrote attributes that were ``None``, which argparse defaults
  never are).
- ``--min-lr`` floors the cosine decay (it was parsed and never read), and the
  schedule no longer rebounds upward past its horizon.
- Checkpoints carry the early-stopping state (``best_loss``, ``patience_counter``,
  validation losses) and the LR horizon so a resumed/hub-resumed slice keeps
  them instead of restarting from scratch.
- The generator gets a held-out validation loader plus a deterministic,
  EMA-aware held-out loss.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch.optim import AdamW

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dataset import CAT_BREEDS, create_train_val_dataloaders
from dit import TinyDiT
from dit_validation import evaluate_flow_loss, evaluate_model_and_ema
from flow_matching import EMA
from train_dit import (
    build_lr_lambda,
    load_checkpoint,
    parse_args,
    save_checkpoint,
)

YAML_CONFIG = """
model:
  image_size: 64
  patch_size: 8
  embed_dim: 32
  depth: 2
  num_heads: 4

training:
  steps: 1234
  batch_size: 16
  lr: 5e-5
  min_lr: 1e-6
  warmup_steps: 111

early_stopping:
  patience: 7
  min_delta: 0.002

ema:
  beta: 0.99

augmentation:
  level: medium

logging:
  log_interval: 25
"""


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    path = tmp_path / "dit_config.yaml"
    path.write_text(YAML_CONFIG)
    return path


@pytest.fixture
def tiny_model() -> TinyDiT:
    return TinyDiT(
        image_size=8,
        patch_size=4,
        embed_dim=16,
        depth=1,
        num_heads=2,
        num_classes=3,
    )


@pytest.fixture
def tiny_generator_dataset(tmp_path: Path) -> Path:
    """Minimal cat/other dataset with breed-encoded filenames."""
    cat_dir = tmp_path / "cat"
    other_dir = tmp_path / "other"
    cat_dir.mkdir()
    other_dir.mkdir()
    Image.new("RGB", (16, 16)).save(cat_dir / f"{CAT_BREEDS[0]}_1.jpg")
    Image.new("RGB", (16, 16)).save(cat_dir / f"{CAT_BREEDS[0]}_2.jpg")
    for index in range(6):
        Image.new("RGB", (16, 16)).save(other_dir / f"beagle_{index}.jpg")
    return tmp_path


class TestConfigLoading:
    """--config must reach the parsed arguments."""

    def test_yaml_values_are_applied(self, config_file: Path) -> None:
        args = parse_args(["--data-dir", "data/cats", "--config", str(config_file)])

        assert args.steps == 1234
        assert args.batch_size == 16
        assert args.image_size == 64
        assert args.warmup_steps == 111
        assert args.log_interval == 25

    def test_yaml_aliases_map_to_argument_destinations(self, config_file: Path) -> None:
        args = parse_args(["--data-dir", "data/cats", "--config", str(config_file)])

        assert args.early_stopping_patience == 7
        assert args.early_stopping_min_delta == pytest.approx(0.002)
        assert args.ema_beta == pytest.approx(0.99)
        assert args.augmentation_level == "medium"

    def test_cli_flags_override_yaml(self, config_file: Path) -> None:
        args = parse_args(
            ["--data-dir", "data/cats", "--config", str(config_file), "--steps", "5"]
        )

        assert args.steps == 5

    def test_unknown_keys_are_ignored(self, tmp_path: Path) -> None:
        path = tmp_path / "cfg.yaml"
        path.write_text("training:\n  steps: 10\n  not_an_option: 3\n")

        args = parse_args(["--data-dir", "data/cats", "--config", str(path)])

        assert args.steps == 10
        assert not hasattr(args, "not_an_option")

    def test_defaults_hold_without_config(self) -> None:
        args = parse_args(["--data-dir", "data/cats"])

        assert args.val_split == pytest.approx(0.05)
        assert args.min_lr == pytest.approx(1e-6)


class TestLrSchedule:
    """Warmup, cosine decay, and the min-lr floor."""

    def test_warmup_starts_near_zero_and_reaches_full_lr(self) -> None:
        lr_lambda = build_lr_lambda(warmup_steps=10, steps=100, min_lr_ratio=0.05)

        assert lr_lambda(0) == pytest.approx(0.01)
        assert lr_lambda(10) == pytest.approx(1.0)
        assert 0.01 < lr_lambda(5) < 1.0

    def test_cosine_halves_midway_and_hits_the_floor(self) -> None:
        lr_lambda = build_lr_lambda(warmup_steps=10, steps=110, min_lr_ratio=0.05)

        assert lr_lambda(60) == pytest.approx(0.5, abs=0.02)
        assert lr_lambda(110) == pytest.approx(0.05)

    def test_schedule_does_not_rebound_past_the_horizon(self) -> None:
        """Resuming a slice with a smaller --steps used to raise the LR."""
        lr_lambda = build_lr_lambda(warmup_steps=10, steps=100, min_lr_ratio=0.0)

        at_end = lr_lambda(100)
        past_end = lr_lambda(150)

        assert at_end == pytest.approx(0.0)
        assert past_end == pytest.approx(0.0)


class TestCheckpointState:
    """Early-stopping state and the LR horizon round-trip through checkpoints."""

    def test_save_and_load_round_trip(
        self, tmp_path: Path, tiny_model: TinyDiT
    ) -> None:
        optimizer = AdamW(tiny_model.parameters(), lr=1e-3)
        ema = EMA(beta=0.9)
        ema.init(tiny_model)
        path = tmp_path / "dit_model.pt"

        save_checkpoint(
            model=tiny_model,
            optimizer=optimizer,
            ema=ema,
            step=42,
            loss=0.5,
            path=path,
            logger=logging.getLogger("test_train_dit"),
            best_loss=0.4,
            patience_counter=2,
            val_loss=0.45,
            val_loss_ema=0.43,
            steps=100,
            warmup_steps=10,
        )

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        assert checkpoint["config"]["num_classes"] == 3
        assert checkpoint["best_loss"] == pytest.approx(0.4)

        fresh_model = TinyDiT(
            image_size=8,
            patch_size=4,
            embed_dim=16,
            depth=1,
            num_heads=2,
            num_classes=3,
        )
        fresh_optimizer = AdamW(fresh_model.parameters(), lr=1e-3)
        fresh_ema = EMA(beta=0.9)
        fresh_ema.init(fresh_model)
        state: dict = {}

        _, _, _, start_step = load_checkpoint(
            path,
            fresh_model,
            fresh_optimizer,
            fresh_ema,
            logging.getLogger("test_train_dit"),
            state=state,
        )

        assert start_step == 43
        assert state["best_loss"] == pytest.approx(0.4)
        assert state["patience_counter"] == 2
        assert state["val_loss"] == pytest.approx(0.45)
        assert state["val_loss_ema"] == pytest.approx(0.43)
        assert state["steps"] == 100
        assert state["warmup_steps"] == 10

        for name, param in tiny_model.state_dict().items():
            assert torch.equal(param, fresh_model.state_dict()[name])


class TestTrainValSplit:
    """The generator must have a held-out, augmentation-free validation set."""

    def test_split_is_disjoint_and_uses_eval_transforms(
        self, tiny_generator_dataset: Path
    ) -> None:
        train_loader, val_loader = create_train_val_dataloaders(
            tiny_generator_dataset,
            batch_size=2,
            image_size=8,
            num_workers=0,
            val_split=0.5,
            seed=1,
        )

        assert val_loader is not None
        total = len(train_loader.dataset) + len(val_loader.dataset)
        assert total == 8
        assert len(val_loader.dataset) == 4
        assert set(train_loader.dataset.indices).isdisjoint(
            set(val_loader.dataset.indices)
        )

        val_transform = val_loader.dataset.dataset.transform
        train_transform = train_loader.dataset.dataset.transform
        assert len(val_transform.transforms) < len(train_transform.transforms)

    def test_zero_split_disables_validation(self, tiny_generator_dataset: Path) -> None:
        train_loader, val_loader = create_train_val_dataloaders(
            tiny_generator_dataset,
            batch_size=2,
            image_size=8,
            num_workers=0,
            val_split=0.0,
            seed=1,
        )

        assert val_loader is None
        assert len(train_loader.dataset) == 8


class TestHeldOutLoss:
    """The validation metric must be deterministic and EMA-aware."""

    @staticmethod
    def _loader() -> list[tuple[torch.Tensor, torch.Tensor]]:
        torch.manual_seed(0)
        return [
            (torch.rand(4, 3, 8, 8) * 2 - 1, torch.tensor([0, 1, 2, 0])),
            (torch.rand(4, 3, 8, 8) * 2 - 1, torch.tensor([1, 1, 2, 0])),
        ]

    def test_evaluate_flow_loss_is_repeatable(self, tiny_model: TinyDiT) -> None:
        loader = self._loader()
        tiny_model.train()

        first = evaluate_flow_loss(tiny_model, loader, torch.device("cpu"), 2, 7)
        second = evaluate_flow_loss(tiny_model, loader, torch.device("cpu"), 2, 7)

        assert first == pytest.approx(second)
        assert tiny_model.training is True

    def test_num_batches_limits_the_window(self, tiny_model: TinyDiT) -> None:
        loader = self._loader()

        assert evaluate_flow_loss(
            tiny_model, loader, torch.device("cpu"), 2, 7
        ) != pytest.approx(
            evaluate_flow_loss(tiny_model, loader, torch.device("cpu"), 1, 7)
        )

    def test_empty_loader_returns_nan(self, tiny_model: TinyDiT) -> None:
        loss = evaluate_flow_loss(tiny_model, [], torch.device("cpu"), 2, 7)

        assert loss != loss  # NaN

    def test_ema_loss_uses_ema_weights_and_restores_model(
        self, tiny_model: TinyDiT
    ) -> None:
        loader = self._loader()
        ema = EMA(beta=0.9)
        ema.init(tiny_model)

        with torch.no_grad():
            for param in tiny_model.parameters():
                param.add_(0.1)
        snapshot = {
            name: param.detach().clone()
            for name, param in tiny_model.named_parameters()
        }

        raw_loss, ema_loss = evaluate_model_and_ema(
            tiny_model, loader, torch.device("cpu"), 2, 7, ema
        )

        assert ema_loss is not None
        assert raw_loss != pytest.approx(ema_loss)
        for name, param in tiny_model.named_parameters():
            assert torch.equal(param, snapshot[name])

    def test_no_ema_returns_none(self, tiny_model: TinyDiT) -> None:
        _, ema_loss = evaluate_model_and_ema(
            tiny_model, self._loader(), torch.device("cpu"), 2, 7, None
        )

        assert ema_loss is None
