# ADR-065: GitHub Actions as control plane; bounded training slices

- **Status:** Accepted (2026-09-23)
- **Date:** 2026-09-23
- **Deciders:** tiny-cats-model maintainers
- **Related:** issue #163 (WP5, WP6), `src/providers.py`, `src/gpu_pool.py`,
  `.github/workflows/train-pool.yml`, `tests/test_providers.py`,
  ADR-063 (exact resume), ADR-064 (canonical layout + Hub transport)

## Context

Issue #163 WP5 called out that `.github/workflows/train-pool.yml` was
misleading: the non-Modal `train-pool-runner` job installed
`torch ... --index-url .../whl/cpu` and therefore ran a **CPU simulation**
while presenting itself as Lightning/Colab/Kaggle training. WP6 additionally
required bounded, resumable training slices instead of one enormous job.

## Decision

### 1. GitHub Actions is the control plane only

`train-pool.yml` never trains. It:

1. **plans** bounded slice targets toward one global target (`plan-slices`),
2. **gates** the requested provider (`provider-gate`),
3. **launches** one real GPU provider session per slice (`train-slice`),
4. **summarizes** (`pool-summary`).

All launch/plan/report logic lives in `src/providers.py` (unit-tested Python),
not shell.

### 2. Provider adapters fail clearly (no silent CPU training)

`providers.ADAPTERS` declares which providers have a real headless adapter.
Today only **Modal** is launchable (`modal run src/train_dit.py`).
Lightning/Colab/Kaggle/HF Spaces/local are `supported=False`:

- `python src/providers.py gate --provider kaggle --strict` exits **2** with
  an actionable message (how to run that provider manually) instead of
  silently CPU-training.
- `gate --provider all` lists supported/unsupported and never launches a CPU
  fallback.
- Manual per-provider scripts (`scripts/train_lightning.py`, …) remain for
  humans who explicitly run them; the control plane never invokes them.

### 3. Bounded/resumable slices (WP6 contract)

- `--steps` is the **global target** (ADR-063); `slice_size` bounds each
  provider session. `plan` emits a GH matrix, e.g. for
  `steps=400000, slice_size=60000`:
  `[60000, 120000, 180000, 240000, 300000, 360000, 400000]` — exactly the
  issue's example layout — filtered against the Hub pointer's
  `completed_steps` so finished slices are skipped; when everything is done
  a single no-op slice runs and reports `exit_reason: completed`.
- `train-slice` runs with `strategy.max-parallel: 1` and `fail-fast: true`:
  slices are sequential (each resumes from the previous slice's checkpoint)
  and a failed slice cancels the rest rather than running on stale state.
- Each slice is its own Modal provider session with its own
  `timeout-minutes: 360`; the default `slice_size=25000` fits the 6-hour
  cron window at the measured T4 throughput (~1.16 steps/s at batch 32,
  ADR-057) — on faster providers pass `-f slice_size=60000`.
- Interruption safety: `hub_push_interval` (default 5000) syncs mid-run, and
  SIGTERM/SIGINT handling (ADR-061/063) leaves a usable checkpoint; the next
  scheduled run's `plan` picks up from the pointer.

### 4. Machine-readable provider report (WP5)

Every slice session writes `provider_report_<target>.json` (uploaded as a
`provider-report-<target>` artifact) containing:

```
provider, gpu_model, vram_gb, job_id, started_at, ended_at,
completed_steps, target_steps, checkpoint_uri, exit_reason
```

`exit_reason` ∈ `completed | partial | interrupted | failed | unsupported`,
derived from the GH step outcome plus the session's `training_state.json`.
The same JSON is echoed to the log as `PROVIDER_REPORT_JSON=...`.
HF Hub remains the only cross-provider state transport (ADR-064).

## Consequences

- The `train-pool-runner` CPU-simulation job is **removed**; requesting an
  unsupported provider now fails the workflow at the gate with instructions.
- Scheduled pool runs plan toward the 400k global target in bounded slices
  instead of a single 60k target.
- Adding a provider = adding an `ProviderAdapter` + launch recipe in
  `src/providers.py` (gate/plan/report behavior is already tested).
- Slice observability moves from raw logs to per-slice report artifacts.
