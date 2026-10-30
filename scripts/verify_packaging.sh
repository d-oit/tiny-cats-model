#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT

python -m build --outdir "$tmp_dir/dist" "$repo_root"
venv_args=()
pip_args=()
if [[ "${VERIFY_PACKAGING_FAST:-0}" == "1" ]]; then
  # Reuse dependencies already installed by CI/local development; still install
  # the built wheel and exercise its actual console entry points.
  venv_args+=(--system-site-packages)
  pip_args+=(--no-deps)
fi
python -m venv "${venv_args[@]}" "$tmp_dir/venv"
venv_python="$tmp_dir/venv/bin/python"
wheel_file=( "$tmp_dir"/dist/*.whl )
"$venv_python" -m pip install "${pip_args[@]}" "${wheel_file[0]}[training]"

if [[ "${SKIP_CLI:-0}" != "1" ]]; then
  "$tmp_dir/venv/bin/train-cats" --help
  "$tmp_dir/venv/bin/evaluate-cats" --help
  "$tmp_dir/venv/bin/verify-cats" --help
fi

"$venv_python" - "$repo_root" <<'PY'
import importlib
import sys
import tomllib
from pathlib import Path

root = Path(sys.argv[1])
with (root / "pyproject.toml").open("rb") as pyproject_file:
    modules = tomllib.load(pyproject_file)["tool"]["setuptools"]["py-modules"]

optional_dependencies = {
    "gradio",
    "huggingface_hub",
    "ipykernel",
    "matplotlib",
    "mlflow",
    "modal",
    "onnx",
    "onnxruntime",
    "onnxruntime_tools",
    "onnxscript",
    "pytest",
    "requests",
    "safetensors",
    "seaborn",
}
for name in modules:
    try:
        importlib.import_module(name)
    except ModuleNotFoundError as error:
        if error.name not in optional_dependencies:
            raise
        print(f"Skipping {name}: optional dependency {error.name!r} is not installed")
    except ImportError as error:
        if name != "test_onnx_inference" or importlib.util.find_spec("onnxruntime"):
            raise
        print(f"Skipping {name}: optional dependency 'onnxruntime' is not installed")
print(f"Checked imports for {len(modules)} packaged modules")
PY
