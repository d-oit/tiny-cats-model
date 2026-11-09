"""Shared canonical-Hub slice runtime for control-plane training providers.

These two helpers implement the campaign slice contract (ADR-063/ADR-064/ADR-067)
that every provider adapter must share:

* ``pull_resume_checkpoint`` resolves the experiment-aware ``latest`` pointer
  instead of the legacy flat ``pool/dit_model_ema.pt`` file, so a slice never
  resumes from stale flat state.
* ``push_progress_backstop`` is the canonical safety push for a run that ended
  between the trainer's interval-boundary pushes.

They were moved here from ``scripts/train_kaggle.py`` so Lightning (and any
future provider) shares the exact working implementations rather than a copy.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger("slice_runtime")


def push_progress_backstop(
    *,
    checkpoint_dir: Path,
    hub_repo: str,
    experiment_id: str,
    hub_push_interval: int,
    token: str | None = None,
) -> None:
    """Canonical Hub safety push for a run that ended between intervals.

    ``train_dit_local`` pushes the canonical layout (immutable ``step-XXXXXX``
    snapshots + the ``latest/manifest.json`` pointer, ADR-064) every
    ``--hub-push-interval`` steps. A crash or early stop *between* boundaries
    would otherwise leave the newest checkpoint local-only; pushing on a
    boundary again would only re-upload an identical snapshot, so this skips
    that case.
    """
    if hub_push_interval <= 0:
        return
    try:
        from gpu_pool import push_checkpoint_to_hub
        from training_state import TRAINING_STATE_FILENAME, read_training_state
    except Exception as exc:  # pragma: no cover - import environment issue
        logger.warning(f"Canonical backstop push unavailable: {exc}")
        return

    state_path = checkpoint_dir / TRAINING_STATE_FILENAME
    state = read_training_state(state_path)
    completed = state.get("completed_steps") if state else None
    if not isinstance(completed, int) or completed <= 0:
        logger.warning("No completed_steps in the training state; backstop skipped")
        return
    if completed % hub_push_interval == 0:
        logger.info(
            f"Step {completed:,} sits on a Hub push boundary; no backstop needed"
        )
        return

    pushed = 0
    for name in ("dit_model.pt", "dit_model_ema.pt"):
        path = checkpoint_dir / name
        if not path.is_file():
            continue
        try:
            push_checkpoint_to_hub(
                checkpoint_path=path,
                hub_repo=hub_repo,
                checkpoint_name=name,
                token=token,
                experiment_id=experiment_id,
                completed_steps=completed,
                training_state_path=state_path,
            )
            pushed += 1
        except Exception as exc:
            logger.warning(f"Backstop push failed for {name}: {exc}")
    if pushed:
        logger.info(f"Backstop push complete at step {completed:,}")


def pull_resume_checkpoint(
    *,
    hub_repo: str,
    checkpoint_dir: Path,
    experiment_id: str,
    token: str | None = None,
) -> Path | None:
    """Pull the newest committed snapshot for a campaign slice.

    Resolves the canonical ``latest`` pointer (issue #163 WP3) instead of the
    legacy flat ``pool/dit_model_ema.pt`` file. That flat file is a pre-WP3
    leftover: resuming a 2026-10-31 Kaggle slice from it replayed steps
    60k->87.5k while the Hub already held 531,238, so the stale-push guard
    discarded the whole session.
    """
    from gpu_pool import pull_checkpoint_from_hub

    pulled = pull_checkpoint_from_hub(
        hub_repo=hub_repo,
        checkpoint_name="dit_model_ema.pt",
        output_dir=str(checkpoint_dir),
        token=token,
        experiment_id=experiment_id,
    )
    return Path(pulled) if pulled else None
