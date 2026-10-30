"""Packaging metadata and console entry-point contract tests."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

tomllib = importlib.import_module("tomllib" if sys.version_info >= (3, 11) else "tomli")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def _project_config() -> dict:
    with (ROOT / "pyproject.toml").open("rb") as pyproject_file:
        return tomllib.load(pyproject_file)


def test_build_backend_and_source_package_mapping() -> None:
    config = _project_config()
    assert config["build-system"]["build-backend"] == "setuptools.build_meta"
    setuptools_config = config["tool"]["setuptools"]
    assert setuptools_config["package-dir"] == {"": "src"}
    modules = setuptools_config["py-modules"]
    assert set(modules) == {path.stem for path in SRC.glob("*.py")}
    for module in modules:
        assert (SRC / f"{module}.py").is_file(), module


def test_console_scripts_resolve_to_callable_entries() -> None:
    config = _project_config()
    scripts = config["project"]["scripts"]
    sys.path.insert(0, str(SRC))
    try:
        for entrypoint in scripts.values():
            module_name, separator, attribute = entrypoint.partition(":")
            assert separator and module_name and attribute, entrypoint
            module = importlib.import_module(module_name)
            assert callable(getattr(module, attribute)), entrypoint
    finally:
        sys.path.remove(str(SRC))


def test_no_dangling_evaluate_or_predict_modules() -> None:
    config = _project_config()
    scripts = config["project"]["scripts"]
    modules = config["tool"]["setuptools"]["py-modules"]
    assert scripts["evaluate-cats"] == "eval:main"
    assert "predict-cats" not in scripts
    assert "evaluate" not in modules
    assert "predict" not in modules


def test_dependency_groups_match_supported_installs() -> None:
    config = _project_config()
    project = config["project"]
    dependencies = {
        dependency.lower().split("[", 1)[0].split(">", 1)[0]
        for dependency in project["dependencies"]
    }
    assert {
        "torch",
        "torchvision",
        "pillow",
        "pyyaml",
        "tqdm",
        "numpy",
    } <= dependencies
    extras = project["optional-dependencies"]
    assert {"training", "onnx", "web-demo", "dev"} <= extras.keys()
    training = {
        dependency.lower().split("[", 1)[0].split(">", 1)[0]
        for dependency in extras["training"]
    }
    assert {"modal", "huggingface_hub", "safetensors", "mlflow", "requests"} <= training
    onnx = {
        dependency.lower().split("[", 1)[0].split(">", 1)[0]
        for dependency in extras["onnx"]
    }
    assert {"onnx", "onnxruntime", "onnxruntime-tools", "onnxscript"} <= onnx
    assert {"gradio", "numpy", "pillow"} <= {
        dependency.lower().split("[", 1)[0].split(">", 1)[0]
        for dependency in extras["web-demo"]
    }
