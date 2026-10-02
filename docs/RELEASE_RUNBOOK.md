# Checkpoint Release Runbook

How to verify, publish, evaluate and tag a pool checkpoint once a global
target is reached. Precedent: the `checkpoint-400k` tag (2026-09-25), whose
annotated message is the canonical evidence trail for a release.

## What happens automatically

The slice that reaches the target handles the completion chain itself
(ADR-064). When `providers.py verify` reports `reached_target=true`, the slice
job: downloads `artifacts/` from the `dit-outputs` volume; requires
`generator/model.pt`, `model.onnx`, `model_quantized.onnx`, `manifest.json`;
runs `verify_checkpoint.py --require-onnx`; generates
`evaluation_report.json` + `benchmark_report.json`; and publishes the verified
package to HF Hub (`push_to_hub` input, default true — HF publishing only
happens in CI, where the `HF_TOKEN` secret exists).

Manual work is therefore: confirm the chain, run the statistical evaluation,
record the numbers, and tag.

## Steps (example: the 600k milestone)

### 1. Confirm the milestone

```bash
python src/providers.py verify \
  --state-file checkpoints/pool/training_state.json \
  --checkpoint checkpoints/pool/dit_model.pt \
  --target 600000
```

Expect `completed_steps=600000`, `reached_target=true`, `checkpoint_valid=true`
(`converged` true only if early stopping fired — `completed_steps` always
reflects the step actually reached, never the target).

### 2. Confirm the completion chain ran and published

```bash
gh run list --workflow=train-pool.yml --limit 3
gh run download <run-id> -n provider-report-600000 -D /tmp/pr600
```

The target slice must be `completed/success` with
`"exit_reason": "completed"` and `"completed_steps": 600000`, and the package
must appear on `huggingface.co/d4oit/tiny-cats-model`.

If training reached the target but the chain did not publish (e.g. the slice
died between training and packaging), re-run it — an already-complete target
is a successful no-op that never overwrites the checkpoint:

```bash
gh workflow run train.yml -f steps=600000 -f batch_size=32
```

### 3. Statistical evaluation (~40 min on CPU)

```bash
bash scripts/eval_checkpoint.sh
```

988 generated samples (13 breeds × 76, 30 steps, CFG 1.5, seed 42) vs all
`data/cats/cat` photos; prints FID / Inception Score / Precision / Recall
against the previous milestone baseline (436k: FID 263.31, IS 1.60, P 0.50,
R 0.48 — see `docs/EVAL-436K-CHECKPOINT-2026-10-02.md`). Record the result in
a dated `docs/EVAL-<steps>-CHECKPOINT-<date>.md` before tagging.

### 4. Annotated tag

```bash
git tag -a checkpoint-600k -m "600k production training complete: <evidence>"
git push origin checkpoint-600k
```

The message should cite the run IDs and gate results, like `checkpoint-400k`:
`providers.py verify` output, the slice job's success, artifact package
verification (ONNX + quantized), evaluation/benchmark report generation, and
Hub publication.

## Release checklist

- [ ] `providers.py verify`: `reached_target=true`, `checkpoint_valid=true`
- [ ] Target slice job `completed/success`, provider report `exit_reason: completed`
- [ ] HF package published to `d4oit/tiny-cats-model`
- [ ] Evaluation recorded in a dated `docs/EVAL-*.md` with delta vs previous milestone
- [ ] Annotated tag pushed