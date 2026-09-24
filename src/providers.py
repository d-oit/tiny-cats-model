"""GitHub Actions control plane for GPU-pool training (issue #163 WP5/WP6).

GitHub Actions orchestrates; provider adapters launch/continue *real* GPU
jobs. Unsupported providers fail clearly instead of silently executing CPU
training. HF Hub remains the state transport (see gpu_pool.py WP3 layout),
and every provider execution writes a machine-readable report:

    provider, GPU model, VRAM, provider job/session ID, start time,
    end time, completed global step, checkpoint URI, exit reason

Training slices (WP6): ``--steps`` is a global target; the control plane
plans bounded per-session slices (e.g. 0 → 60k → 120k → ... → 400k) and
runs them as separate provider sessions, each resuming exactly from the
Hub checkpoint (see ADR-065).

CLI (used by .github/workflows/train-pool.yml):

    python src/providers.py plan   --steps 400000 --slice-size 60000
    python src/providers.py gate   --provider lightning --strict
    python src/providers.py launch --provider modal --target 60000 ...
    python src/providers.py report --provider modal --job-id ... --target 60000
"""

from __future__ import annotations

import argparse
import itertools
import json
import shlex
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Canonical Hub pool layout (must match gpu_pool.py POOL_ROOT, WP3).
POOL_ROOT = "checkpoints/pool"
DEFAULT_EXPERIMENT_ID = "dit-breed-conditioned-v4"
DEFAULT_HUB_REPO = "d4oit/tiny-cats-model"

# VRAM in GB for the GPU models our providers offer.
VRAM_GB: dict[str, int] = {"T4": 16, "L4": 24, "P100": 16, "A10G": 24, "CPU": 0}

EXIT_REASONS = frozenset(
    {"completed", "partial", "interrupted", "failed", "unsupported"}
)


class UnsupportedProviderError(RuntimeError):
    """Raised when a provider cannot run real GPU training from the control plane.

    Deliberately loud: issue #163 WP5 requires unsupported providers to fail
    clearly rather than silently execute a CPU simulation.
    """


@dataclass(frozen=True)
class ProviderAdapter:
    """Describes how (and whether) a provider can be driven remotely."""

    name: str
    supported: bool
    gpu_models: list[str] = field(default_factory=list)
    session_limit_minutes: int = 0
    launch_hint: str = ""  # how to run it when supported=False

    @property
    def gpu_model(self) -> str:
        return self.gpu_models[0] if self.gpu_models else "CPU"

    @property
    def vram_gb(self) -> int:
        return VRAM_GB.get(self.gpu_model, 0)


ADAPTERS: dict[str, ProviderAdapter] = {
    "modal": ProviderAdapter(
        name="modal",
        supported=True,
        gpu_models=["T4", "L4"],
        session_limit_minutes=1440,
    ),
    "lightning": ProviderAdapter(
        name="lightning",
        supported=False,
        gpu_models=["T4"],
        session_limit_minutes=720,
        launch_hint=(
            "run manually on a Lightning studio: "
            "python scripts/train_lightning.py --hub-resume"
        ),
    ),
    "colab": ProviderAdapter(
        name="colab",
        supported=False,
        gpu_models=["T4"],
        session_limit_minutes=720,
        launch_hint=(
            "Colab has no supported headless automation; run the training "
            "notebook manually with --hub-resume"
        ),
    ),
    "kaggle": ProviderAdapter(
        name="kaggle",
        supported=False,
        gpu_models=["T4", "P100"],
        session_limit_minutes=540,
        launch_hint=(
            "run manually as a Kaggle notebook/kernel: "
            "python scripts/train_kaggle.py --hub-resume"
        ),
    ),
    "hf_spaces": ProviderAdapter(
        name="hf_spaces",
        supported=False,
        gpu_models=["T4"],
        session_limit_minutes=480,
        launch_hint=(
            "HF Spaces has no supported headless GPU-job API for this repo; "
            "start a Space manually with --hub-resume"
        ),
    ),
    "local": ProviderAdapter(
        name="local",
        supported=False,
        launch_hint=(
            "local training is manual and outside the pool: "
            "python src/train_dit.py --data-dir data/cats --steps N"
        ),
    ),
}


def get_adapter(provider: str) -> ProviderAdapter:
    """Return the adapter for ``provider``; raises for anything not launchable.

    Known-but-unsupported providers raise :class:`UnsupportedProviderError`
    with an actionable message (exit code 2 via the ``gate`` CLI).
    """
    key = provider.strip().lower()
    if key == "all":
        raise UnsupportedProviderError("'all' is not a single provider")
    adapter = ADAPTERS.get(key)
    if adapter is None:
        valid = ", ".join(sorted(ADAPTERS))
        raise UnsupportedProviderError(
            f"Unknown provider {provider!r}. Valid providers: {valid}. "
            "Refusing to guess (no CPU fallback)."
        )
    if not adapter.supported:
        raise UnsupportedProviderError(
            f"Provider {adapter.name!r} is not supported by the GitHub Actions "
            f"control plane yet — refusing to run CPU-simulated training. "
            f"To continue this provider, {adapter.launch_hint}. "
            "See plans/ADR-065-control-plane-and-training-slices.md."
        )
    return adapter


def parse_slice_targets(
    steps: int,
    slice_size: int | None = None,
    targets: list[int] | None = None,
    completed: int = 0,
) -> list[int]:
    """Plan bounded slice targets on the path to the global target ``steps``.

    - ``targets``: explicit strictly-increasing targets; ``steps`` is appended
      when missing so the contract always ends at the global target.
    - ``slice_size``: grid of multiples of the slice size ending at ``steps``
      (e.g. 60000 → [60000, 120000, ..., 400000]).

    Targets at or below ``completed`` are dropped; when everything is already
    complete a single no-op slice ``[steps]`` is returned so the session still
    reports a machine-readable status.
    """
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")

    if targets is not None:
        if not targets:
            raise ValueError("explicit targets list is empty")
        if any(t <= 0 for t in targets):
            raise ValueError(f"targets must be positive: {targets}")
        if any(b <= a for a, b in itertools.pairwise(targets)):
            raise ValueError(f"targets must be strictly increasing: {targets}")
        if any(t > steps for t in targets):
            raise ValueError(
                f"targets must not exceed global target {steps}: {targets}"
            )
        plan = list(targets)
        if plan[-1] != steps:
            plan.append(steps)
    else:
        if slice_size is None or slice_size <= 0:
            raise ValueError(f"slice_size must be positive, got {slice_size}")
        plan = list(range(slice_size, steps + 1, slice_size))
        if not plan or plan[-1] != steps:
            plan.append(steps)

    remaining = [t for t in plan if t > completed]
    return remaining or [steps]


def build_launch_command(
    provider: str,
    target: int,
    *,
    batch_size: str = "32",
    lr: str = "5e-5",
    save_interval: str = "500",
    hub_push_interval: str = "5000",
    hub_resume: bool = False,
    no_hub_push: bool = False,
    experiment_id: str = DEFAULT_EXPERIMENT_ID,
) -> list[str]:
    """Build the exact command that launches one bounded provider session.

    Currently only the Modal adapter can be launched headlessly; every other
    provider raises :class:`UnsupportedProviderError` (WP5: fail clearly).
    """
    adapter = get_adapter(provider)
    if adapter.name != "modal":
        raise UnsupportedProviderError(
            f"No launch recipe for provider {adapter.name!r} yet. {adapter.launch_hint}"
        )
    if target <= 0:
        raise ValueError(f"target must be positive, got {target}")

    command = [
        "modal",
        "run",
        "src/train_dit.py",
        "--data-dir",
        "/data/cats",
        "--output",
        "/outputs/checkpoints/pool/dit_model.pt",
        "--ema-output",
        "/outputs/checkpoints/pool/dit_model_ema.pt",
        "--steps",
        str(target),
        "--batch-size",
        str(batch_size),
        "--lr",
        str(lr),
        "--save-interval",
        str(save_interval),
        "--hub-push-interval",
        str(hub_push_interval),
        "--experiment-id",
        experiment_id,
    ]
    if hub_resume:
        command.append("--hub-resume")
    if no_hub_push:
        command.append("--no-hub-push")
    return command


def fetch_remote_completed(
    hub_repo: str = DEFAULT_HUB_REPO,
    experiment_id: str = DEFAULT_EXPERIMENT_ID,
    token: str | None = None,
) -> int | None:
    """Read ``completed_steps`` from the Hub ``latest/manifest.json`` pointer.

    Returns None when the pointer does not exist yet (first run). Network or
    Hub errors propagate so callers can decide (the ``plan`` CLI treats them
    as "unknown" and warns).
    """
    from huggingface_hub import hf_hub_download  # local import: optional dep

    filename = f"{POOL_ROOT}/{experiment_id}/latest/manifest.json"
    path = hf_hub_download(
        repo_id=hub_repo,
        filename=filename,
        repo_type="model",
        token=token,
    )
    document = json.loads(Path(path).read_text())
    if not isinstance(document, dict):
        raise ValueError(f"Hub pointer {filename} is not a JSON object")
    completed = document.get("completed_steps")
    return int(completed) if completed is not None else None


@dataclass
class ProviderReport:
    """Machine-readable result of one provider session (issue #163 WP5)."""

    provider: str
    gpu_model: str
    vram_gb: int
    job_id: str
    started_at: str
    ended_at: str
    exit_reason: str
    completed_steps: int | None = None
    target_steps: int | None = None
    checkpoint_uri: str | None = None
    experiment_id: str | None = None

    def __post_init__(self) -> None:
        if self.exit_reason not in EXIT_REASONS:
            raise ValueError(
                f"exit_reason must be one of {sorted(EXIT_REASONS)}, "
                f"got {self.exit_reason!r}"
            )
        for name in ("provider", "gpu_model", "job_id", "started_at", "ended_at"):
            if not getattr(self, name):
                raise ValueError(f"ProviderReport.{name} must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, path: str | Path) -> Path:
        """Atomically write the report (tmp file + rename, WP1 style)."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        tmp = destination.with_name(destination.name + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
        tmp.replace(destination)
        return destination

    @classmethod
    def read(cls, path: str | Path) -> ProviderReport:
        return cls(**json.loads(Path(path).read_text()))


def resolve_exit_reason(
    outcome: str, completed_steps: int | None, target_steps: int | None
) -> str:
    """Map a session outcome to a WP5 exit reason.

    - cancelled → interrupted (a usable checkpoint must survive; WP6.8)
    - failure   → failed
    - success   → completed when the target was reached, otherwise partial
      (including the odd case of success with no readable training state).
    """
    if outcome == "cancelled":
        return "interrupted"
    if outcome != "success":
        return "failed"
    if (
        completed_steps is not None
        and target_steps is not None
        and completed_steps >= target_steps
    ):
        return "completed"
    return "partial"


def _read_state(state_file: str | None) -> dict[str, Any] | None:
    if not state_file:
        return None
    path = Path(state_file)
    if not path.exists():
        return None
    from training_state import read_training_state  # src is on sys.path (CLI)

    return read_training_state(path)


def _cmd_plan(args: argparse.Namespace) -> int:
    completed = args.completed
    if args.remote_completed:
        try:
            remote = fetch_remote_completed(
                hub_repo=args.hub_repo,
                experiment_id=args.experiment_id,
                token=args.token,
            )
        except Exception as exc:  # plan must degrade, not die
            print(f"warning: could not read Hub pointer: {exc}", file=sys.stderr)
        else:
            completed = max(completed, remote or 0)

    targets = None
    if args.targets:
        targets = [int(t) for t in args.targets.split(",") if t.strip()]
    plan = parse_slice_targets(
        steps=args.steps,
        slice_size=args.slice_size,
        targets=targets,
        completed=completed,
    )
    print(json.dumps({"target": plan}))
    return 0


def _cmd_gate(args: argparse.Namespace) -> int:
    requested = (
        sorted(ADAPTERS) if args.provider == "all" else [args.provider.strip().lower()]
    )
    unsupported: list[str] = []
    for name in requested:
        try:
            adapter = get_adapter(name)
        except UnsupportedProviderError as exc:
            if args.provider == "all" and not args.strict:
                unsupported.append(name)
                print(f"unsupported: {exc}")
                continue
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(
            f"supported: {adapter.name} gpu={adapter.gpu_model} "
            f"vram={adapter.vram_gb}GB "
            f"session_limit={adapter.session_limit_minutes}min"
        )
    if unsupported:
        print("note: no CPU fallback will be run for: " + ", ".join(unsupported))
    return 0


def _cmd_launch(args: argparse.Namespace) -> int:
    command = build_launch_command(
        args.provider,
        args.target,
        batch_size=args.batch_size,
        lr=args.lr,
        save_interval=args.save_interval,
        hub_push_interval=args.hub_push_interval,
        hub_resume=args.hub_resume,
        no_hub_push=args.no_hub_push,
        experiment_id=args.experiment_id,
    )
    print(shlex.join(command))
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    state = _read_state(args.state_file)
    completed = int(state["completed_steps"]) if state else None
    target = (
        int(state["target_steps"])
        if state and state.get("target_steps")
        else args.target
    )
    exit_reason = resolve_exit_reason(args.outcome, completed, target)

    if args.gpu_model:
        gpu_model = args.gpu_model
        vram = VRAM_GB.get(gpu_model, 0)
    else:
        gpu_model = get_adapter(args.provider).gpu_model
        vram = VRAM_GB.get(gpu_model, 0)

    report = ProviderReport(
        provider=args.provider,
        gpu_model=gpu_model,
        vram_gb=vram,
        job_id=args.job_id,
        started_at=args.started_at,
        ended_at=args.ended_at or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        exit_reason=exit_reason,
        completed_steps=completed,
        target_steps=target,
        checkpoint_uri=args.checkpoint_uri,
        experiment_id=(state or {}).get("experiment_id"),
    )
    report.write(args.out)
    # Machine-readable status line for the session log (WP6.10).
    print(f"PROVIDER_REPORT_JSON={json.dumps(report.to_dict(), sort_keys=True)}")
    return 0 if exit_reason in {"completed", "partial"} else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="providers.py",
        description="GitHub Actions control plane for GPU-pool training (WP5/WP6).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="print a GH Actions matrix of slice targets")
    plan.add_argument("--steps", type=int, required=True, help="global target")
    plan.add_argument("--slice-size", type=int, default=None)
    plan.add_argument("--targets", type=str, default=None, help="explicit a,b,c list")
    plan.add_argument("--completed", type=int, default=0)
    plan.add_argument("--experiment-id", default=DEFAULT_EXPERIMENT_ID)
    plan.add_argument("--hub-repo", default=DEFAULT_HUB_REPO)
    plan.add_argument("--token", default=None)
    plan.add_argument(
        "--remote-completed",
        action="store_true",
        help="read completed steps from the Hub latest/manifest.json pointer",
    )
    plan.set_defaults(func=_cmd_plan)

    gate = sub.add_parser("gate", help="fail clearly for unsupported providers")
    gate.add_argument("--provider", required=True)
    gate.add_argument(
        "--strict",
        action="store_true",
        help="exit 2 when the (single) provider is unsupported",
    )
    gate.set_defaults(func=_cmd_gate)

    launch = sub.add_parser("launch", help="print the provider launch command")
    launch.add_argument("--provider", required=True)
    launch.add_argument("--target", type=int, required=True)
    launch.add_argument("--batch-size", default="32")
    launch.add_argument("--lr", default="5e-5")
    launch.add_argument("--save-interval", default="500")
    launch.add_argument("--hub-push-interval", default="5000")
    launch.add_argument("--hub-resume", action="store_true")
    launch.add_argument("--no-hub-push", action="store_true")
    launch.add_argument("--experiment-id", default=DEFAULT_EXPERIMENT_ID)
    launch.set_defaults(func=_cmd_launch)

    report = sub.add_parser("report", help="write the machine-readable session report")
    report.add_argument("--provider", required=True)
    report.add_argument("--gpu-model", default=None)
    report.add_argument("--job-id", required=True)
    report.add_argument("--started-at", required=True)
    report.add_argument("--ended-at", default=None)
    report.add_argument("--target", type=int, required=True)
    report.add_argument("--state-file", default=None)
    report.add_argument("--checkpoint-uri", default=None)
    report.add_argument(
        "--outcome",
        required=True,
        choices=["success", "failure", "cancelled", "skipped"],
        help="GH step outcome of the training step",
    )
    report.add_argument("--out", required=True)
    report.set_defaults(func=_cmd_report)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except UnsupportedProviderError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
