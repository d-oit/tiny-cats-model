# ADR-058: TinyDiT training — switch to L40S (non-spot) and tighten save_interval

- **Status:** Accepted
- **Date:** 2026-07-07
- **Deciders:** tiny-cats-model maintainers

## Context

TinyDiT (the diffusion-transformer generator for cats) trains on Modal GPUs via
`src/train_dit.py` + `.github/workflows/train.yml`. After the latest merge to
`main`, a live 100k-step run on the **A10G** GPU (the choice recorded in
ADR-023) was preempted at step 568 — Modal reassigned the spot-tier A10G back
to a higher-priority tenant. Because `save_interval` defaulted to **10,000
steps**, no checkpoint was on the volume before the preempt, so 568 steps of
compute were lost and the next `gh workflow run` would have hit the same fate.

This ADR formalizes the fix implemented in PR #107.

## Decision

1. **Switch the DiT training GPU from A10G (spot) to L40S (non-spot).**
   - `src/train_dit.py`: `gpu="A10G"  # Better for transformer training (ADR-023)`
     → `gpu="L40S"  # Non-spot GPU: avoids preemptions on A10G (ADR-057/058)`.
   - L40S is non-spot on Modal, so it is reclaimed only on container-idle
     timeout (currently 24h on the function), not on tenant priority shifts.
2. **Tighten `save_interval` to 1,000 steps for DiT.**
   - `src/train_dit.py`: default remains `10_000` (conservative for the local
     CLI).
   - `.github/workflows/train.yml`: a new `save_interval` `workflow_dispatch`
     input (default `"1000"`) is wired through to
     `modal run src/train_dit.py --save-interval ...`.
3. **Document the explicit `ADR-058` cross-reference in code comments.**
   - The `gpu="L40S"` decorator comment names both this ADR and ADR-057.
   - Future engineers touching these lines have one obvious place to look.

## Rationale

- **Root cause vs. mitigation.** The bug is *not* that the user asked for a
  10,000-step cadence; it's that A10G is spot-tier and can be revoked at
  any wall-clock moment. Switching to L40S addresses the root cause. The
  tightened `save_interval` is a belt-and-braces mitigation so that even
  if Modal's non-spot policy for L40S ever regresses or we later switch
  back to spot for cost, `load_checkpoint()` can resume within ~1 minute of
  progress loss instead of ~3 hours.
- **Cost.** L40S is non-spot, so it's priced higher than A10G and cannot
  be preempted for capacity reasons. The trade-off is intentional: we
  would rather pay a small per-hour premium than watch training silently
  fail and re-loop every three hours. Empirically the previous run
  wasted most of its wall-clock on cold-start/image-pull/dataset-download
  overhead (see ADR-057 "Speed vs. wall-clock" section) — the GPU work
  itself was only a few minutes, so the per-hour cost delta is small
  relative to the wasted CI minutes and engineering time.
- **Cross-reference.** ADR-023 picked A10G when training was shorter and
  the cost was the dominant constraint. ADR-057 audited Modal and
  flagged preemption as the dominant operational risk. ADR-058 combines
  both — it formally relocates the GPU class for DiT, and the workflow
  surface (save_interval input) is the user-visible knob an operator can
  tune if the L40S economics ever change again.

## Consequences

### Positive

- DiT training completes deterministically given the 24h function timeout.
- Loss of progress on any future preemption is bounded to ~1 minute instead
  of ~3 hours.
- One workflow_dispatch input (`save_interval`) is added; other inputs are
  unchanged.

### Negative

- ~3–5× GPU-hour cost for DiT runs (L40S vs. A10G). Acceptable given the
  training is event-driven (manually triggered) rather than continuous,
  and the previous A10G runs were effectively 100% wasted on preemption
  cycles.

### Neutral / followup

- The classifier (`src/train.py`) still uses A10G in ADR-023. Classifier
  training is much shorter (~20 epochs, minutes), so preemptions there are
  cheaper and the A10G choice remains reasonable. No change required for
  this ADR.
- If we ever move DiT back to a budget-tier spot GPU, we should still keep
  `save_interval` ≥ 1,000 so resume is fast.

## Implementation

| File | Change |
|---|---|
| `src/train_dit.py` | `gpu="A10G"` → `gpu="L40S"` (line ~461) |
| `.github/workflows/train.yml` | New `save_interval` `workflow_dispatch` input (default `"1000"`) and `--save-interval ${{ github.event.inputs.save_interval || '1000' }}` appended to the `modal run src/train_dit.py` invocation |

PR: **#107** — `chore(dit): switch to L40S GPU + save_interval=1k (ADR-058)`.
Merged 2026-07-07. Validation: `bash scripts/quality-gate.sh` (ruff format,
ruff lint, actionlint, yamllint, mypy, pytest, 17 agent-skill validator), code
review (Nit Pick Nick) — all green.

## References

- **ADR-023** — Modal GPU retry strategy (chose A10G originally).
- **ADR-025** — Modal cold-start optimization (kept; relevant for L40S spin-up time).
- **ADR-057** — Modal CLI verification & best-practice audit (flagged preempt as
  primary risk; recommended L40S/A100).
- **PR #107** — Implementation.
