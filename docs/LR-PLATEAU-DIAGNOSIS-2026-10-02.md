# Training Plateau Root Cause: LR Pinned at the Floor — 2026-10-02

Why the `dit-breed-conditioned-v4` run stopped improving long before its step
target: the cosine schedule has effectively been at its `--min-lr` floor since
~480k steps, i.e. **2% of the nominal learning rate**.

## Evidence

Checkpoint fields (`checkpoints/pool/dit_model_ema.pt`, pulled from the volume):

| Field | Value | Meaning |
|-------|-------|---------|
| `step` | 480000 | steps actually completed |
| `steps` | 525000 | recorded as "LR-schedule horizon this run is using" |
| `manifest.target_steps` | 480000 | this slice's target |

525,000 is a **stale slice target** — it belongs to a slice of the 2026-10-01
run that was killed (and whose 6h-timeout slices never advanced training
meaningfully). The resume path continues whatever horizon is recorded:

```python
recorded_steps = resume_state.get("steps")
if recorded_steps:
    schedule_steps = max(int(recorded_steps), steps)
```

`save_checkpoint` stores the *session target* under `steps`, so every slice
inherits the previous slice's target as the LR horizon, and
`max(recorded, target)` faithfully continues a wrong curve. The real campaign
horizon (600k) is recorded nowhere.

Plugging the numbers through `build_lr_lambda` (`warmup=2000`,
`base_lr=5e-5`, `min_lr_ratio = min_lr/base_lr = 0.02`):

| Horizon | Step | Effective LR | % of nominal |
|---------|------|--------------|--------------|
| 525000 (recorded) | 450000 | 2.49e-6 | 5.0% |
| 525000 (recorded) | 502500 | **1.00e-6** | **2.0%** |
| 600000 (true target) | 502500 | 3.21e-6 | 6.4% |
| 400000 (original run) | 502500 | 1.00e-6 | 2.0% |

At 2% of nominal LR with AdamW, plus EMA beta 0.9999 (a ~10k-step averaging
window), the model is effectively frozen — matching the measured plateau
(FID 263.31 → 261.72 across 44k steps) and the unchanged artifact profile.

## Remediation

**Immediate, no code change to the trainer:** `--min-lr` is *not* part of the
immutable experiment manifest (manifest carries `learning_rate`, `scheduler`,
`warmup_steps`), and the resume path reads the value from the CLI. Raising the
floor therefore lifts the LR without tripping the manifest gate. It is plumbed
through the control plane as an opt-in flag, enabled by a repository variable
(a `workflow_dispatch` input was not possible — the event caps at 10 inputs and
this workflow already uses all 10):

```bash
gh variable set MIN_LR -b 2e-5   # unset = trainer default 1e-6, no-op today
```

`--min-lr 1e-5` puts the remaining run at ~20% of nominal LR (a standard
stage-2 constant-LR extension); `2e-5` is more aggressive. Verify by watching
`val_loss_ema` and the next FID measurement rather than the training loss,
which is augmented-batch only.

**Proper fix (after the campaign, or with an explicit manifest migration):**

1. Persist the *experiment* horizon separately from the session target
   (e.g. `schedule_steps` in the checkpoint, seeded from the campaign goal)
   and let `max()` continue that instead of the last slice's target.
2. Consider EMA beta 0.999 for stage-2 work: 0.9999 lags a small-LR phase by
   ~10k steps.
3. Add a guard: if the cosine decay term is below `min_lr_ratio` at resume,
   log a warning that the run is running at the LR floor — the current logs
   never say this, which is why the plateau went unnoticed.

## Note on scope

The `steps`-as-horizon semantics arrive with #164 (2026-09-23), which predates
the 400k release tag (2026-09-25). This is therefore not specific to the
600k stretch: any run that reached its horizon through slices is suspect, and
the late-stage 400k slices most likely sat at the floor too (unverified —
that would need the 400k-era checkpoints, which were superseded).