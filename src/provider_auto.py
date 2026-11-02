#!/usr/bin/env python3
"""Provider fallback dispatcher for the GPU pool (ADR-067).

Runs the first launchable provider in the chain and falls back to the next
one **only** when an attempt fails on a launch precondition — concretely, a
provider-side billing/quota block that never reached training. A slice that
actually started and then failed is *not* retried on another provider: the
failure stays visible (the stall tracker opens an issue) instead of being
masked by a provider switch.

Usage (printed by ``providers.py launch --provider auto ...``)::

    python src/provider_auto.py --target 540000 --kernel-id owner/slug --hub-resume

The provider that actually ran is written to ``--provider-used-file`` so the
workflow's report/verify steps can fetch artifacts from the right source.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from providers import (
    UnsupportedProviderError,
    build_launch_command,
)

logger = logging.getLogger("provider-auto")

DEFAULT_PROVIDERS = "modal,kaggle"

# Launch-precondition failures that justify falling back: the provider never
# started training, so another provider can run the same slice unchanged.
TERMINAL_LAUNCH_PATTERNS = (
    "has exceeded its spend limit",
    "exceeded your spend limit",
)

EXIT_CONFIG = 5


def run_provider(command: list[str]) -> tuple[int, bool]:
    """Run one provider command, streaming its output.

    Returns:
        ``(exit_code, blocked)`` where ``blocked`` is True when a launch
        precondition pattern appeared in the output (the run never trained).
    """
    blocked = False
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        lowered = line.lower()
        if any(pattern in lowered for pattern in TERMINAL_LAUNCH_PATTERNS):
            blocked = True
    return process.wait(), blocked


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Try each launchable provider in order (ADR-067)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--target", type=int, required=True, help="Global step target")
    parser.add_argument("--providers", default=DEFAULT_PROVIDERS)
    parser.add_argument("--hub-repo", default="d4oit/tiny-cats-model")
    parser.add_argument("--experiment-id", default="dit-breed-conditioned-v4")
    parser.add_argument("--batch-size", default="32")
    parser.add_argument("--lr", default="5e-5")
    parser.add_argument("--save-interval", default="2500")
    parser.add_argument("--hub-push-interval", default="5000")
    parser.add_argument("--warmup-steps", default=None)
    parser.add_argument("--min-lr", default=None)
    parser.add_argument("--schedule-steps", default=None)
    parser.add_argument(
        "--kernel-id",
        default=os.environ.get("KAGGLE_KERNEL_ID"),
        help=(
            "owner/slug of the Kaggle kernel (defaults to $KAGGLE_KERNEL_ID or "
            "the authenticated account's tiny-cats-dit-slice)"
        ),
    )
    parser.add_argument(
        "--hf-token-dataset", default=os.environ.get("KAGGLE_HF_TOKEN_DATASET")
    )
    parser.add_argument("--hub-resume", action="store_true")
    parser.add_argument("--no-hub-push", action="store_true")
    parser.add_argument("--allow-experiment-mismatch", action="store_true")
    parser.add_argument(
        "--provider-used-file",
        default=".provider-used",
        help="File recording the provider that actually ran",
    )
    return parser.parse_args(argv)


def main(
    argv: list[str] | None = None,
    *,
    runner: Callable[[list[str]], tuple[int, bool]] | None = None,
) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )
    run = runner or run_provider

    chain = [part.strip() for part in args.providers.split(",") if part.strip()]
    if not chain:
        logger.error("no providers configured (--providers is empty)")
        return EXIT_CONFIG

    launch_kwargs = {
        "batch_size": args.batch_size,
        "lr": args.lr,
        "save_interval": args.save_interval,
        "hub_push_interval": args.hub_push_interval,
        "hub_resume": args.hub_resume,
        "no_hub_push": args.no_hub_push,
        "warmup_steps": args.warmup_steps,
        "min_lr": args.min_lr,
        "schedule_steps": args.schedule_steps,
        "allow_experiment_mismatch": args.allow_experiment_mismatch,
        "experiment_id": args.experiment_id,
        "hub_repo": args.hub_repo,
        "kernel_id": args.kernel_id,
        "hf_token_dataset": args.hf_token_dataset,
    }

    used_file = Path(args.provider_used_file)
    unavailable: list[str] = []
    for index, provider in enumerate(chain):
        try:
            command = build_launch_command(provider, args.target, **launch_kwargs)
        except (ValueError, UnsupportedProviderError) as exc:
            unavailable.append(f"{provider}: {exc}")
            logger.warning("skipping %s: %s", provider, exc)
            continue

        used_file.parent.mkdir(parents=True, exist_ok=True)
        used_file.write_text(provider + "\n")
        logger.info("launching %s: %s", provider, " ".join(command))
        exit_code, blocked = run(command)

        summary = {
            "provider": provider,
            "exit_code": exit_code,
            "fallback": index > 0,
            "blocked": blocked,
        }
        if exit_code == 0:
            print("PROVIDER_AUTO_JSON=" + json.dumps(summary, sort_keys=True))
            return 0
        if blocked and index + 1 < len(chain):
            logger.warning(
                "%s is blocked by a launch precondition; falling back to %s",
                provider,
                chain[index + 1],
            )
            continue
        if blocked:
            logger.error(
                "%s is blocked by a launch precondition and no provider is "
                "left in the chain (%s)",
                provider,
                ", ".join(chain),
            )
        else:
            logger.error(
                "%s failed at %s with no fallback: only launch-precondition "
                "failures switch providers, training failures stay visible",
                provider,
                exit_code,
            )
        print("PROVIDER_AUTO_JSON=" + json.dumps(summary, sort_keys=True))
        return exit_code

    for reason in unavailable:
        logger.error("provider unavailable: %s", reason)
    print(
        "PROVIDER_AUTO_JSON="
        + json.dumps({"provider": None, "exit_code": EXIT_CONFIG}, sort_keys=True)
    )
    return EXIT_CONFIG


if __name__ == "__main__":
    sys.exit(main())
