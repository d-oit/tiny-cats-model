"""Guardrail for the Modal container import contract (ADR-068)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SENSOR = ROOT / "scripts" / "check_modal_container_contract.py"


def test_modal_container_contract_holds() -> None:
    """The entrypoint must hydrate inside the container, not just locally.

    The 2026-11-02 Modal app failed at startup with
    ``ModuleNotFoundError: No module named 'safe_checkpoint'`` because the
    /root-mounted entrypoint imported a sibling before /app was on sys.path.
    """
    result = subprocess.run(
        [sys.executable, str(SENSOR)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
