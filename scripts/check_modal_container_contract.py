#!/usr/bin/env python3
"""scripts/check_modal_container_contract.py

Guardrail sensor for the container import contract (ADR-068).

`modal run <script>` mounts the entrypoint at ``/root`` and imports it during
container hydration, before any function body runs. The image copies modules
into ``/app``. A module-level local import therefore needs ``/app`` on
``sys.path`` before the import line, and every module reachable from a mounted
module must be mounted too.

The Kaggle adapter has a different transport (a zipless workspace archive) but
the same failure mode: a module that the slice imports must be in the pack.

Checks per Modal entrypoint
---------------------------
1. mounts      - module-level local imports AND their transitive local imports
                 are mounted into the image.
2. pythonpath  - every mounted module directory is on ``PYTHONPATH``.
3. bootstrap   - the ``sys.path`` bootstrap precedes the first module-level
                 local import (skipped when there is none).
4. foreign cwd - the module imports from a directory with no sibling modules
                 (skipped when a third-party dependency is absent).

Checks for the Kaggle pack
--------------------------
5. kaggle-pack - the workspace copies every ``src/*.py`` and the extra kernel
                 files, so every local import of the slice resolves.

Exit codes: 0 contract holds, 1 contract violated.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

MODAL_ENTRYPOINTS = ("src/train_dit.py", "src/train.py")
KAGGLE_PROVIDER = "src/kaggle_provider.py"
KAGGLE_SLICE_MODULES = ("scripts/train_kaggle.py", "src/train_dit.py")

TOP_IMPORT_RE = re.compile(r"^from ([A-Za-z_][A-Za-z0-9_]*) import", re.MULTILINE)
ANY_IMPORT_RE = re.compile(r"^\s*from ([A-Za-z_][A-Za-z0-9_]*) import", re.MULTILINE)
MOUNT_RE = re.compile(r'add_local_file\(\s*"([^"]+)"\s*,\s*"([^"]+)"\s*\)')
FULL_SRC_MOUNT_RE = re.compile(r'add_local_dir\(\s*"src"\s*,\s*"/app"')
PYTHONPATH_RE = re.compile(r'"PYTHONPATH":\s*"([^"]+)"')
MISSING_MODULE_RE = re.compile(r"No module named '([^']+)'")
EXTRA_KERNEL_FILES_RE = re.compile(r"EXTRA_KERNEL_FILES\s*=\s*\(([^)]*)\)", re.DOTALL)


class ContractError(Exception):
    """A violation of the container import contract."""


def _is_local_module(name: str) -> bool:
    return (SRC / f"{name}.py").is_file()


def _local_imports(source: str, pattern: re.Pattern[str]) -> dict[str, int]:
    """Map local module name -> first line number where it is imported."""
    imports: dict[str, int] = {}
    for match in pattern.finditer(source):
        name = match.group(1)
        if _is_local_module(name):
            imports.setdefault(name, source[: match.start()].count("\n") + 1)
    return imports


def _mounts(source: str) -> dict[str, str]:
    """Map local repo-relative path -> container path."""
    return dict(MOUNT_RE.findall(source))


def check_mounts(entry: str, source: str) -> str:
    """Every local module reachable from the entrypoint must be mounted."""
    if FULL_SRC_MOUNT_RE.search(source):
        return "whole src/ mounted"
    mounted = {
        Path(local).name for local in _mounts(source) if local.startswith("src/")
    }
    missing: list[str] = []
    seen: set[str] = set()
    stack = list(_local_imports(source, TOP_IMPORT_RE))
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        if f"{name}.py" not in mounted:
            missing.append(f"src/{name}.py")
            continue
        stack.extend(_local_imports((SRC / f"{name}.py").read_text(), ANY_IMPORT_RE))
    if missing:
        raise ContractError(
            f"{entry}: image does not mount " + ", ".join(sorted(set(missing)))
        )
    return f"{len(seen)} local modules covered"


def check_pythonpath(entry: str, source: str) -> None:
    """Every mounted module's container directory must be importable."""
    path_match = PYTHONPATH_RE.search(source)
    if not path_match:
        raise ContractError(
            f"{entry}: image does not set PYTHONPATH — add "
            '\'"PYTHONPATH": "/app"\' so /root hydration can import image copies'
        )
    search_path = [part for part in path_match.group(1).split(":") if part]
    unreachable = sorted(
        {
            str(Path(remote).parent)
            for local, remote in _mounts(source).items()
            if local.startswith("src/") and str(Path(remote).parent) not in search_path
        }
    )
    if unreachable:
        raise ContractError(
            f"{entry}: mounted module directories are not on PYTHONPATH: "
            + ", ".join(unreachable)
        )


def check_bootstrap_order(entry: str, source: str) -> None:
    """The sys.path bootstrap must precede the first local module import."""
    imported = _local_imports(source, TOP_IMPORT_RE)
    if not imported:
        return
    first_import = min(imported.values())
    bootstrap = [
        number
        for number, line in enumerate(source.splitlines(), start=1)
        if "sys.path.insert" in line
    ]
    if not bootstrap or min(bootstrap) > first_import:
        raise ContractError(
            f"{entry}: module-level local imports run before any sys.path "
            f"bootstrap (first local import at line {first_import}) — the "
            "/root-mounted entrypoint will fail container hydration"
        )


def check_foreign_cwd_import(entry: str) -> str:
    """Import the entrypoint from a directory that has no sibling modules."""
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / Path(entry).name
        shutil.copy(ROOT / entry, target)
        env = {**os.environ, "PYTHONPATH": str(SRC)}
        env.pop("PYTHONHOME", None)
        result = subprocess.run(
            [sys.executable, "-c", f"import {target.stem}; print('ok')"],
            cwd=tmp,
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
    if result.returncode == 0:
        return "ok"
    missing = MISSING_MODULE_RE.search(result.stderr)
    if missing and _is_local_module(missing.group(1)):
        raise ContractError(
            f"{entry}: does not import from a foreign cwd — missing local module "
            f"'{missing.group(1)}'\n{result.stderr.strip()}"
        )
    if missing:
        return f"skipped (third-party '{missing.group(1)}' not installed)"
    raise ContractError(
        f"{entry}: does not import from a foreign cwd:\n{result.stderr.strip()}"
    )


def check_kaggle_pack() -> None:
    """The Kaggle workspace must ship every module the slice imports."""
    provider = (ROOT / KAGGLE_PROVIDER).read_text()
    extras_match = EXTRA_KERNEL_FILES_RE.search(provider)
    if not extras_match:
        raise ContractError(
            f"{KAGGLE_PROVIDER}: EXTRA_KERNEL_FILES is missing or unparsable"
        )
    extras = set(re.findall(r'"([^"]+)"', extras_match.group(1)))
    if 'glob("*.py")' not in provider:
        raise ContractError(
            f"{KAGGLE_PROVIDER}: workspace no longer copies every src/*.py; the "
            "pack is a whitelist and its coverage cannot be proven"
        )
    packed = set(extras) | {f"src/{path.name}" for path in SRC.glob("*.py")}
    missing = [
        f"src/{name}.py (imported by {module})"
        for module in KAGGLE_SLICE_MODULES
        for name in _local_imports((ROOT / module).read_text(), ANY_IMPORT_RE)
        if f"src/{name}.py" not in packed
    ]
    missing.extend(
        f"{required} (missing from EXTRA_KERNEL_FILES)"
        for required in ("scripts/train_kaggle.py", "data/download.py")
        if required not in extras
    )
    if missing:
        raise ContractError("Kaggle pack is incomplete: " + ", ".join(sorted(missing)))


def _run(label: str, check) -> int:
    try:
        detail = check()
    except ContractError as error:
        print(f"FAIL {label}: {error}")
        return 1
    suffix = f" ({detail})" if isinstance(detail, str) else ""
    print(f"ok   {label}{suffix}")
    return 0


def main() -> int:
    """Run every check and report the contract result."""
    failures = 0
    for entry in MODAL_ENTRYPOINTS:
        source = (ROOT / entry).read_text()
        failures += _run(
            f"{entry}:mounts", lambda e=entry, s=source: check_mounts(e, s)
        )
        failures += _run(
            f"{entry}:pythonpath", lambda e=entry, s=source: check_pythonpath(e, s)
        )
        failures += _run(
            f"{entry}:bootstrap-order",
            lambda e=entry, s=source: check_bootstrap_order(e, s),
        )
        failures += _run(
            f"{entry}:foreign-cwd-import", lambda e=entry: check_foreign_cwd_import(e)
        )
    failures += _run("kaggle:pack", check_kaggle_pack)
    if failures:
        print(f"\n{failures} contract check(s) failed — see ADR-068")
        return 1
    print("\nContainer import contract holds")
    return 0


if __name__ == "__main__":
    sys.exit(main())
