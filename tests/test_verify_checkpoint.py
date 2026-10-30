"""tests/test_verify_checkpoint.py

Issue #163 WP7/WP8: ``verify_checkpoint.py`` is the publication gate used by
``train.yml``. Its default contract is "ONNX problems are a warning", and
``--require-onnx`` upgrades that to a hard failure so a broken export can never
be published as a verified final model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import torch

import verify_checkpoint as vc
from dit import TinyDiT

CONFIG = {
    "image_size": 4,
    "patch_size": 2,
    "embed_dim": 8,
    "depth": 1,
    "num_heads": 2,
    "num_classes": 13,
}


def _save_checkpoint(
    path: Path, config: dict | None = None, state: dict | None = None
) -> None:
    model = TinyDiT(**(config or CONFIG))
    torch.save(
        {
            "config": config or CONFIG,
            "model_state_dict": model.state_dict() if state is None else state,
        },
        path,
    )


@pytest.fixture
def valid_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip the (heavy) real checkpoint load: report a valid checkpoint."""
    monkeypatch.setattr(
        vc,
        "verify_checkpoint",
        lambda _path: {"valid": True, "path": "ckpt.pt"},
    )


def test_invalid_checkpoint_exits_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        vc, "verify_checkpoint", lambda _path: {"valid": False, "error": "corrupt"}
    )
    with pytest.raises(SystemExit) as excinfo:
        vc.main(["--checkpoint", "ckpt.pt", "--skip-onnx"])
    assert excinfo.value.code == 1


def test_onnx_failure_is_a_warning_by_default(
    valid_checkpoint: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        vc, "verify_onnx_inference", lambda _path: {"valid": False, "error": "boom"}
    )
    # No SystemExit: the checkpoint is valid and ONNX is advisory here.
    vc.main(["--checkpoint", "ckpt.pt", "--onnx", "model.onnx"])


def test_require_onnx_fails_when_onnx_invalid(
    valid_checkpoint: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        vc, "verify_onnx_inference", lambda _path: {"valid": False, "error": "boom"}
    )
    with pytest.raises(SystemExit) as excinfo:
        vc.main(["--checkpoint", "ckpt.pt", "--onnx", "model.onnx", "--require-onnx"])
    assert excinfo.value.code == 1


def test_require_onnx_passes_when_onnx_valid(
    valid_checkpoint: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        vc, "verify_onnx_inference", lambda _path: {"valid": True, "error": None}
    )
    vc.main(["--checkpoint", "ckpt.pt", "--onnx", "model.onnx", "--require-onnx"])


def test_skip_onnx_never_calls_onnx_verification(
    valid_checkpoint: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(_path: str) -> dict:
        raise AssertionError("ONNX verification must not run with --skip-onnx")

    monkeypatch.setattr(vc, "verify_onnx_inference", boom)
    vc.main(["--checkpoint", "ckpt.pt", "--skip-onnx"])


def test_valid_architecture_exercises_all_batch_and_breed_cases(tmp_path: Path) -> None:
    checkpoint = tmp_path / "valid.pt"
    _save_checkpoint(checkpoint)
    report = vc.verify_checkpoint(checkpoint)
    assert report["valid"]
    assert report["config_valid"]
    assert report["parameter_count"] > 0
    assert {
        (case["batch_size"], case["breed_index"]) for case in report["inference_tests"]
    } == {(batch, breed) for batch in (1, 2) for breed in (0, 12, 13)}
    assert all(case["finite"] for case in report["inference_tests"])


def test_invalid_architecture_config_fails(tmp_path: Path) -> None:
    checkpoint = tmp_path / "invalid.pt"
    invalid = {**CONFIG, "image_size": 5}
    _save_checkpoint(checkpoint, invalid)
    report = vc.verify_checkpoint(checkpoint)
    assert not report["valid"]
    assert not report["config_valid"]
    assert "divisible" in report["error"]


@pytest.mark.parametrize(
    ("change", "field"),
    [
        ("missing", "missing_keys"),
        ("unexpected", "unexpected_keys"),
        ("shape", "shape_mismatches"),
        ("dtype", "dtype_mismatches"),
    ],
)
def test_tensor_mismatches_are_reported_and_fail(
    tmp_path: Path, change: str, field: str
) -> None:
    model = TinyDiT(**CONFIG)
    state = dict(model.state_dict())
    first_key = next(iter(state))
    if change == "missing":
        state.pop(first_key)
    elif change == "unexpected":
        state["unexpected.weight"] = torch.zeros(1)
    elif change == "shape":
        state[first_key] = torch.zeros(1)
    else:
        state[first_key] = state[first_key].double()
    checkpoint = tmp_path / f"{change}.pt"
    _save_checkpoint(checkpoint, state=state)
    report = vc.verify_checkpoint(checkpoint)
    assert not report["valid"]
    assert report[field]


def test_checked_load_does_not_copy_any_parameters_on_failure() -> None:
    model = TinyDiT(**CONFIG)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    state = dict(before)
    state[next(iter(state))] = torch.zeros(1)
    with pytest.raises(ValueError):
        vc.load_state_dict_checked(model, state)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key])


def _write_onnx(
    path: Path,
    *,
    bad_dtype: bool = False,
    bad_dims: bool = False,
    bad_names: bool = False,
) -> None:
    import onnx
    from onnx import TensorProto, helper

    batch = "batch"
    noise_shape = [batch, 3, 128 if not bad_dims else 64, 128]
    inputs = [
        helper.make_tensor_value_info(
            "noise",
            TensorProto.DOUBLE if bad_dtype else TensorProto.FLOAT,
            noise_shape,
        ),
        helper.make_tensor_value_info(
            "step" if bad_names else "timestep", TensorProto.FLOAT, [batch]
        ),
        helper.make_tensor_value_info("breed", TensorProto.INT64, [batch]),
    ]
    output = helper.make_tensor_value_info(
        "velocity", TensorProto.FLOAT, [batch, 3, 128, 128]
    )
    graph = helper.make_graph(
        [helper.make_node("Cast", ["noise"], ["velocity"], to=TensorProto.FLOAT)],
        "tiny-test",
        inputs,
        [output],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, path)


def test_onnx_node_names_dtypes_and_dimensions_are_validated(tmp_path: Path) -> None:
    pytest.importorskip("onnx")
    good = tmp_path / "good.onnx"
    _write_onnx(good)
    assert vc.verify_onnx_inference(good)["valid"]

    bad_dtype = tmp_path / "bad-dtype.onnx"
    _write_onnx(bad_dtype, bad_dtype=True)
    dtype_report = vc.verify_onnx_inference(bad_dtype)
    assert not dtype_report["valid"]
    assert any("dtype" in error for error in dtype_report["node_validation"])

    bad_dims = tmp_path / "bad-dims.onnx"
    _write_onnx(bad_dims, bad_dims=True)
    dims_report = vc.verify_onnx_inference(bad_dims)
    assert not dims_report["valid"]
    assert any("dimensions" in error for error in dims_report["node_validation"])

    bad_names = tmp_path / "bad-names.onnx"
    _write_onnx(bad_names, bad_names=True)
    names_report = vc.verify_onnx_inference(bad_names)
    assert not names_report["valid"]
    assert any("names" in error for error in names_report["node_validation"])
