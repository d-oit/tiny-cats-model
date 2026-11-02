#!/usr/bin/env python3
"""Assert the canonical pool pointer advanced after a bounded slice.

``checkpoints/pool/<experiment-id>/latest/manifest.json`` is flipped last on
every Hub push (ADR-064), so it is the only truthful record of campaign
progress. A slice can exit 0, package its artifacts and still leave the pointer
where it was — that happened on 2026-11-02, when the end-of-run push used the
pre-training ``completed_steps`` and HuggingFace skipped an "empty" commit.

Usage:
    # before the slice
    python scripts/check_pool_pointer.py --record verify/pointer_before.json

    # after the slice
    python scripts/check_pool_pointer.py --record-file verify/pointer_before.json \
        --expect-advanced --target 555100

Exit codes: 0 = ok (progress, or a no-op slice), 1 = the pointer did not move
although the target required it / the pointer cannot be read.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gpu_pool import fetch_pool_pointer

SCHEMA_VERSION = 1


def read_pointer(
    hub_repo: str, experiment_id: str, token: str | None
) -> dict[str, Any] | None:
    """Fetch the pointer document; raises on transient failures."""
    return fetch_pool_pointer(
        hub_repo=hub_repo, experiment_id=experiment_id, token=token
    )


def pointer_step(pointer: dict[str, Any] | None) -> int:
    """Completed global step recorded by a pointer (0 when absent)."""
    if not isinstance(pointer, dict):
        return 0
    value = pointer.get("completed_steps", 0)
    return int(value) if isinstance(value, int) else 0


def record(args: argparse.Namespace, token: str | None) -> int:
    pointer = read_pointer(args.hub_repo, args.experiment_id, token)
    document = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": args.experiment_id,
        "completed_steps": pointer_step(pointer),
        "pointer_found": isinstance(pointer, dict),
        "updated_at": pointer.get("updated_at") if isinstance(pointer, dict) else None,
    }
    path = Path(args.record)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(
        f"pointer before slice: step {document['completed_steps']:,} "
        f"(found={document['pointer_found']}) -> {path}"
    )
    return 0


def expect_advanced(args: argparse.Namespace, token: str | None) -> int:
    before_path = Path(args.record_file)
    try:
        before = json.loads(before_path.read_text())
    except (FileNotFoundError, OSError, json.JSONDecodeError) as exc:
        print(f"::error::cannot read the recorded pointer ({before_path}): {exc}")
        return 1
    if before.get("recorded_failure"):
        print(
            "::warning::the pre-slice pointer could not be recorded; "
            "skipping the advancement assertion"
        )
        return 0
    recorded = int(before.get("completed_steps", 0))
    target = int(args.target)
    pointer = read_pointer(args.hub_repo, args.experiment_id, token)
    current = pointer_step(pointer)

    if target <= recorded:
        print(
            f"SKIP: target {target:,} <= recorded {recorded:,} — this slice was a "
            "successful no-op, so no pointer advancement is expected"
        )
        return 0
    if pointer is None:
        print(
            f"::error::no canonical pointer found at "
            f"{args.experiment_id}/latest (the slice published nothing)"
        )
        return 1
    if current <= recorded:
        print(
            f"::error::canonical pointer did not advance: recorded {recorded:,}, "
            f"now {current:,}, target {target:,}. The slice ran but its "
            "checkpoint never reached the Hub — check the end-of-run push."
        )
        return 1
    print(
        f"OK: canonical pointer advanced {recorded:,} -> {current:,} "
        f"(target {target:,})"
    )
    if args.github_output:
        with open(os.environ.get("GITHUB_OUTPUT", args.github_output), "a") as fh:
            fh.write(f"pointer_before={recorded}\n")
            fh.write(f"pointer_after={current}\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hub-repo", default="d4oit/tiny-cats-model")
    parser.add_argument("--experiment-id", default="dit-breed-conditioned-v4")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--record", help="Write the current pointer to this path")
    mode.add_argument(
        "--record-file",
        help="Compare the current pointer against this recorded document",
    )
    parser.add_argument(
        "--expect-advanced",
        action="store_true",
        help="Require the pointer to have moved past the recorded step",
    )
    parser.add_argument("--target", type=int, help="Global target of the slice")
    parser.add_argument("--github-output", help="Optional $GITHUB_OUTPUT file")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if args.record:
        try:
            return record(args, token)
        except Exception as exc:  # network/auth: never fail the pre-step
            print(f"::warning::could not record the pool pointer: {exc}")
            path = Path(args.record)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "experiment_id": args.experiment_id,
                        "completed_steps": 0,
                        "pointer_found": False,
                        "recorded_failure": str(exc),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
            return 0
    if args.target is None:
        print("::error::--expect-advanced requires --target")
        return 1
    return expect_advanced(args, token)


if __name__ == "__main__":
    raise SystemExit(main())
