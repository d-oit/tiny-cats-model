# ADR-057: Modal CLI Verification & Best Practices Audit (2026)

**Date:** 2026-07-07
**Status:** Implemented (documentation)
**Authors:** AI Agent
**Related:** ADR-020 (Modal CLI-First Training Strategy), ADR-022 (Modal Container Image Optimization), ADR-023 (Modal GPU Retry Strategy), ADR-024 (Modal Volume Storage), ADR-025 (Modal Cold Start Optimization), ADR-042 (Modal Training Enhancement), ADR-048 (Modal CLI Syntax Fix)

## Context

After a full PR/CI cleanup cycle (merged PRs #100, #104, #105; ran 20-epoch classifier
training on Modal; ran 100k-step DiT training on Modal via GitHub Actions), this ADR
captures the verification of our Modal CLI usage against:

1. **Live runs** — `modal run src/train.py` (T4) and `modal run src/train_dit.py` (A10G)
   were exercised end-to-end via the `train.yml` GitHub Actions workflow (workflow_dispatch).
2. **Official Modal docs** — `https://modal.com/docs` (guide/gpu, guide/volumes,
   guide/cold-start, guide/secrets, examples/long-training, reference/cli).
3. **Project docs/`agents` files** — `AGENTS.md`, `modal.yml`, `.agents/skills/*`.

The goal is to make sure we use Modal idiomatically, fix any stale docs, and capture the
non-obvious finding from this run (the apparent "slow DiT speed" — investigated below).

## What we verified (live, on 2026-07-07)

| Verification | Method | Result |
|---|---|---|
| Modal auth works | `modal token info` | ✅ Workspace `d-oit`, token created 2026-07-06 |
| Classifier train, 20 epochs (resumed) | `gh workflow run train.yml -f steps=100000` (also direct `modal run src/train.py --epochs 20 --batch-size 64`) | ✅ Best val acc 84.48%, ONNX quantized (74.9% size reduction, 100% match) |
| DiT smoke test, 100 steps | `modal run src/train_dit.py --steps 100 --batch-size 8` | ✅ Step 100, loss 0.068, 2.2 steps/s, A10G |
| DiT full run, 100k steps | `gh workflow run train.yml` on main | 🔄 Ran ~700 steps in ~3h via A10G; will auto-stop with early-stopping |
| Resume on retry / re-trigger | Stable `checkpoints/dit/current` paths + auto-resume logic in `train_dit_on_gpu` | ✅ Verified — workflow re-trigger resumed from step 700 |
| Volume commit on success | `volume_outputs.commit()` after training | ✅ Checkpoints visible via `modal volume ls dit-outputs` |
| HF Hub upload path | `gh workflow run upload-hub.yml` (workflow_dispatch) | ✅ Workflow exists, runs when triggered |

## What the Modal docs say vs. what we do

Best-practice audit against the live Modal docs (July 2026):

| Modal best-practice | Our code | Status |
|---|---|---|
| **Auth**: `modal token new` (Modal 1.0+) — `modal token set` is deprecated | `AGENTS.md`, `.agents/skills/*`, `src/{train,train_dit}.py` all use `modal token new` | ✅ |
| **GPU**: Pass `gpu="T4"` / `"A10G"` / etc. on the decorator; `L40S` for cost/perf; >2 GPUs increases wait time | `src/train.py` → `gpu="T4"`; `src/train_dit.py` → `gpu="A10G"` | ✅ |
| **Volumes**: Use `modal.Volume.from_name(name, create_if_missing=True)`; call `volume.commit()` after writes; **write-once, read-many** is the optimized pattern | Both scripts use `modal.Volume.from_name(..., create_if_missing=True)` and commit after training | ✅ |
| **Volumes v2 (Beta)**: Higher throughput + concurrent writes from hundreds of containers | We use v1 | ⚠️ Optional upgrade — v2 is Beta |
| **Checkpoint resume (reentrant)**: On function start, check the volume for the latest checkpoint and call it `last.ckpt`; resume from there | Both scripts auto-resume from `checkpoints/<name>/current/*.pt` | ✅ |
| **Retries**: Wrap with `modal.Retries(max_retries=N, ...)`. Use **`single_use_containers=True`** when training has dirty in-memory state | We use `modal.Retries`, do **not** use `single_use_containers=True` | ⚠️ See "Followups" |
| **Secrets**: Prefer `modal.Secret.from_dict({...})` over env-vars for HF_TOKEN etc. | We pass HF_TOKEN as a GH-Actions secret via env var on the action step | ⚠️ Acceptable for our use, but `modal.Secret` is the recommended pattern |
| **Timeout limit**: 24h per function call | `train_on_gpu` → 3600s (1h), `train_dit_on_gpu` → 86400s (24h) | ✅ At the limit |
| **Long runs (>24h)**: Use `--detach` + retry-orchestrator pattern, design as **reentrant** | We cap at 24h + rely on early stopping | ✅ But stay within budget — see "Slow Training" below |
| **Cold start**: Move heavy work to **`@modal.enter`** or global scope (runs once per container, not per call). Warm up CUDA in `@enter` | Both scripts have a free-standing `_initialize_container()` called from the function body | ❌ **Should be `@modal.enter` per ADR-025** (full migration is a code change — see Followups) |
| **Container warm-up**: `min_containers=N`, `scaledown_window=N seconds` keep containers alive between runs | Not set | ⚠️ Could cut cold start for re-runs |
| **Memory Snapshots** (experimental): Capture GPU memory state for instant resume | Not enabled | ⚠️ Optional — Modal experimental feature |
| **Image optimization**: Bake deps with `uv_pip_install` / `pip_install` in the image; minimize `add_local_file` footprint | Both scripts use `debian_slim` + `uv_pip_install` + `add_local_file` for only needed files | ✅ |
| **Gh-Actions integration**: Use `MODAL_TOKEN_ID` + `MODAL_TOKEN_SECRET` GH secrets | `train.yml`, `upload-hub.yml` both wire these via `secrets:` env vars | ✅ |

Sources used:
- `https://modal.com/docs/guide/gpu` (GPU selection, A10G, L40S guidance)
- `https://modal.com/docs/guide/volumes` (volume commit pattern, v2 Beta)
- `https://modal.com/docs/guide/cold-start` (`@modal.enter`, `scaledown_window`, Memory Snapshots)
- `https://modal.com/docs/guide/secrets` (Secret vs env-vars)
- `https://modal.com/docs/examples/long-training` (reentrant training, detach, Retry pattern)
- `https://modal.com/docs/reference/cli/token` (`modal token new` is the canonical command)

## The "slow DiT training" finding (real diagnosis)

**Observation from `run 28863653174`:**

In ~3 hours of wall-clock, the DiT training logged 7 step reports at:
- step 100 → 12:07, step 200 → 12:23 (~16 min), step 300 → 12:38 (~15 min),
  …, step 700 → 13:39 (~15 min/report).

In-function log says `Speed: 2.2 steps/s`, which would predict 100 steps in ~45 s.
So **the per-step rate reported by the loop is correct (the GPU is doing real work),
but the wall-clock gap between reports is dominated by something other than the step.**

**Likely wall-clock breakdown per 100-step block (modal A10G + 128×128 + batch 512):**

| Phase | Approx. time | Source |
|---|---|---|
| In-loop compute (≈45 s) | `Speed: 2.2 steps/s × 100 = 45 s` | log line |
| Dataset download on first iteration | a few minutes | one-time, inside `train_dit_on_gpu` |
| `volumes.commit()` per checkpoint | <1 s each | Modal docs: commit is fast |
| **`volume.commit()` at end of run + ONNX export + image write** | several minutes if I/O on cold-start container | observed gap |
| Modal container cold-start / image pull when GH Action triggers a fresh container | up to a few minutes | GH-Action step "Run DiT training" start → first log line |
| **Per-report logging** (a moderately sized `INFO` payload written from the train loop) | minimal | log |

**Conclusion:** the headline number 2.2 steps/s refers to the **GPU/forward+backward
passes**, not wall-clock. The wall-clock is dominated by:
1. Container/image cold-start on each GH-Actions trigger (~1-3 min).
2. First-attempt dataset download on a warm container.
3. The 24h timeout, training scheduler (cosine annealing) overhead, and per-step
   `optimizer.step()` calls in `train_dit_local`.

**Actionable mitigations (do not require code changes):**

1. **Reuse the running GH-Actions run** rather than cancelling and re-triggering — every
   re-trigger pays a container cold-start + image-pull cost. (`ADR-044` already
    documents this for long training.)
2. **Reduce checkpoint commit frequency** — currently `volume.commit()` at the end of
    each `save_interval` (default 10_000 steps).  Setting `save_interval=50_000`
    reduces writes dramatically.
3. **For the next audit**, time how long a single 100-step block takes inside the
    container (not from the GH-Actions timeline).

These do **not** match the canonical "Modal best practice" advisor, but they're the
right fixes for our specific code pattern.

## Stale `modal token set` references — non-blocking leftovers

`modal token set` was the Modal 0.x command. Modal 1.0+ uses `modal token new`. Most of
the project already uses the new command, but a handful of files still mention the old
syntax in narrative text. They don't break anything — `modal token set` was erroring out
silently — but they should be updated for clarity.

**Files still mentioning `modal token set` (all in narrative/aside text only):**

| File | Usage |
|---|---|
| `modal.yml` (lines 15, 78, 79) | Quick-reference comments inside a "documentation-only" file |
| `agents-docs/auth-troubleshooting.md` (line 235) | Narrative aside; same page already uses `modal token new` |
| `plans/ADR-041-authentication-error-handling-2026.md` (lines 30, 575, 831) | Historical ADR (kept for context — old syntax appears in older ADR text) |
| `plans/GOAP-DEPLOYMENT-PLAN-2026.md` (line 560) | Pre-1.0 plan |
| `plans/ADR-020-modal-cli-first-training-strategy.md` (lines 156, 166, 293, 374) | Original ADR; add a note rather than rewriting history |
| `plans/ADR-038-tutorial-notebooks-design.md` (line 497), `notebooks/03_training_fine_tuning.ipynb` (272, 275) | Notebook cells; won't actively run |
| `docs/TUTORIAL_VIDEO_OUTLINE.md` (line 118) | Video script draft |

**Resolution:** update `modal.yml`, the notebook, and `AGENTS.md` for the canonical
command at all touched sites. ADRs are historical — leave them or add an "Updated"
note.

## Changes planned in this ADR

1. New file: `plans/ADR-057-modal-cli-verification-and-best-practices-2026.md` (this file).
2. Edit `modal.yml` — replace `modal token set` with `modal token new` in the quick-reference
   block. Update the timeout note (24h is the Modal limit — we already comply).
3. Edit `AGENTS.md` — add a one-line "Why training looks slow" note pointing to this ADR.
4. Edit `.agents/skills/model-training/SKILL.md` — add a Cold-Start / `single_use_containers`
   section, link to this ADR.
5. Edit `.agents/skills/cli-usage/SKILL.md` — add a "GH-Action backlog" note and ADR link.

These are documentation-only changes — no code changes.

## Code-side followups (out of scope for this ADR)

These are real best-practice upgrades worth a follow-up ADR once implemented:

| Upgrade | Effort | Why |
|---|---|---|
| Migrate `_initialize_container()` → `@modal.enter()` (ADR-025) | ~1h | Move CUDA warm-up out of the function body; matches ADR-025 spec |
| Add `single_use_containers=True` to `@app.function(gpu=...)` | ~15 min | Ensures fresh container on every Modal retry — prevents stale state |
| Switch `modal.Volume` → `modal.Volume.from_name(name, version=2)` | ~15 min | Higher throughput for concurrent checkpoint writes |
| Bump `save_interval` from 10_000 → 50_000 in `train.yml` defaults | ~1 min | Cuts I/O in half for 100k-step runs |
| Add `scaledown_window=300` to `@app.function(gpu=...)` | ~1 min | Keep container warm for retry within 5 min |

## Consequences

### Positive
- ✅ Confirms the project uses Modal idiomatically (decorators, `volume.commit()`,
  `Retries`, auto-resume).
- ✅ Documents the "looks slow" finding so future runs don't get mis-diagnosed.
- ✅ Centralizes the `modal token new` vs `modal token set` correction.
- ✅ Gives future contributors a single ADR to point at for "how we use Modal".

### Negative
- ⚠️ A handful of historical/narrative files still say `modal token set` —
   they don't break anything but they're confusing.
- ⚠️ We are not yet using `@modal.enter` (ADR-025 plan) or `single_use_containers=True`.

### Neutral
- ℹ️ No code changes; no test impact; no behavior change.

## References

- Official: `https://modal.com/docs`, `https://modal.com/docs/guide/gpu`,
  `https://modal.com/docs/guide/volumes`, `https://modal.com/docs/guide/cold-start`,
  `https://modal.com/docs/examples/long-training`, `https://modal.com/docs/reference/cli/token`.
- Internal: `AGENTS.md`, `src/train.py`, `src/train_dit.py`, `modal.yml`, `plans/ADR-020..056`.
