# ADR-058: TinyDiT training — GPU selection and cost optimization

- **Status:** Accepted (updated 2026-07-07)
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

This ADR formalizes the GPU selection strategy and cost optimization for DiT training.

## Decision

1. **Switch the DiT training GPU from A10G (spot) to L40S (non-spot), then to T4/L4 (cost-optimized).**
   - `src/train_dit.py`: `gpu="A10G"` → `gpu="L40S"` → `gpu=["T4", "L4"]`
   - L40S was chosen to avoid preemptions, but proved too expensive ($1.95/hr).
   - T4 ($0.59/hr) with L4 fallback ($0.80/hr) provides 3x cost reduction.
   - GPU fallbacks allow Modal to pick cheapest available.

2. **Tighten `save_interval` to 1,000 steps for DiT.**
   - `src/train_dit.py`: default remains `10_000` (conservative for the local
     CLI).
   - `.github/workflows/train.yml`: a new `save_interval` `workflow_dispatch`
     input (default `"1000"`) is wired through to
     `modal run src/train_dit.py --save-interval ...`.

3. **Document the explicit `ADR-058` cross-reference in code comments.**
   - The `gpu=["T4", "L4"]` decorator comment names this ADR.
   - Future engineers touching these lines have one obvious place to look.

## Rationale

- **Root cause vs. mitigation.** The bug is *not* that the user asked for a
  10,000-step cadence; it's that A10G is spot-tier and can be revoked at
  any wall-clock moment. Switching to L40S addressed preemption, but
  introduced cost issues. T4/L4 provides cost optimization while accepting
  preemption risk (mitigated by frequent checkpoints).
- **Cost.** L40S ($1.95/hr) proved too expensive for $10 credit budget.
  T4 ($0.59/hr) provides 17 hours of training vs 5 hours on L40S.
  L4 ($0.80/hr) provides fallback with 12.5 hours of training.
- **Preemption vs. cost trade-off.** T4/L4 are spot-tier and can be
  preempted, but with `save_interval=1000`, resume is fast (~1 minute
  progress loss). The 3x cost reduction justifies accepting preemption risk.
- **Cross-reference.** ADR-023 picked A10G when training was shorter and
  the cost was the dominant constraint. ADR-057 audited Modal and
  flagged preemption as the dominant operational risk. ADR-058 combines
  both — it formally relocates the GPU class for DiT, and the workflow
  surface (save_interval input) is the user-visible knob an operator can
  tune if the economics ever change again.

## Consequences

### Positive

- DiT training completes deterministically given the 24h function timeout.
- Loss of progress on any future preemption is bounded to ~1 minute instead
  of ~3 hours.
- 3x cost reduction: T4 ($0.59/hr) vs L40S ($1.95/hr).
- One workflow_dispatch input (`save_interval`) is added; other inputs are
  unchanged.

### Negative

- T4/L4 are spot-tier and can be preempted (unlike L40S). Mitigated by
  frequent checkpoints and Modal retry logic.

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
| `src/train_dit.py` | `gpu="A10G"` → `gpu="L40S"` → `gpu=["T4", "L4"]` (line ~493) |
| `.github/workflows/train.yml` | New `save_interval` `workflow_dispatch` input (default `"1000"`) and `--save-interval ${{ github.event.inputs.save_interval || '1000' }}` appended to the `modal run src/train_dit.py` invocation |

PR: **#107** — `chore(dit): switch to L40S GPU + save_interval=1k (ADR-058)`.
Merged 2026-07-07. Validation: `bash scripts/quality-gate.sh` (ruff format,
ruff lint, actionlint, yamllint, mypy, pytest, 17 agent-skill validator), code
review (Nit Pick Nick) — all green.

Updated 2026-07-07: Switched from L40S to T4/L4 for cost optimization.

## References

- **ADR-023** — Modal GPU retry strategy (chose A10G originally).
- **ADR-025** — Modal cold-start optimization (kept; relevant for T4/L4 spin-up time).
- **ADR-057** — Modal CLI verification & best-practice audit (flagged preempt as
  primary risk; recommended L40S/A100).
- **PR #107** — Implementation.
- **Modal Pricing** — https://modal.com/pricing (T4: $0.59/hr, L4: $0.80/hr, L40S: $1.95/hr)
