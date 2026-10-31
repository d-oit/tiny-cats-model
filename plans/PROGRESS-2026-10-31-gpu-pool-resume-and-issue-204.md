# Progress 2026-10-31 — GPU pool resume + issue #204

- **Branch:** `feat/open-issues-contract-hardening`
- **Related:** ADR-063 (exact global-step resume), ADR-064 (canonical Hub
  layout), ADR-065 (control plane), ADR-066 (bounded slices), ADR-067
  (Kaggle adapter + auto fallback), issue #204

## GPU pool: why the campaign stopped

Three scheduled `Train (GPU Pool)` runs failed on 2026-10-31
(runs 41221001996, 41197538754, 41165937975). Every one failed inside
`Launch slice on the selected provider`:

```
launching modal: modal run src/train_dit.py ... --steps 540000 ...
Error: Workspace ac-yWabHo1HEkyjRi6HzZAzxH has exceeded its spend limit
modal is blocked by a launch precondition; falling back to kaggle
skipping kaggle: the kaggle provider needs --kernel-id owner/slug
provider unavailable: kaggle: ...
PROVIDER_AUTO_JSON={"exit_code": 5, "provider": null}
```

So the Modal spend limit was handled correctly (ADR-067 fallback), but the
fallback itself was not launchable: `KAGGLE_KERNEL_ID` was never set as a
repository variable (only `MIN_LR` exists), and `build_launch_command` treated
a missing kernel id as a hard configuration error. `KAGGLE_API_TOKEN` *is*
configured. The pool therefore made no progress despite a valid second
provider, and the run-page annotation that exists to surface this was skipped.

### Changes

1. **Kaggle kernel id auto-resolution** (`src/kaggle_provider.py`)
   - `resolve_kernel_id(api, kernel_id)`: explicit flag → `KAGGLE_KERNEL_ID`
     env → `<authenticated-account>/tiny-cats-dit-slice`.
   - `--kernel-id` is no longer `required=True`; `main()` authenticates early
     only when it must resolve the owner, then reuses that client for push/poll.
   - `--hf-token-dataset` also falls back to `KAGGLE_HF_TOKEN_DATASET`.
   - Kaggle creates the kernel on first push, so a fresh repository now needs
     only `KAGGLE_API_TOKEN`.
2. **Control plane** (`src/providers.py`, `src/provider_auto.py`)
   - `build_launch_command` reads `KAGGLE_KERNEL_ID` /
     `KAGGLE_HF_TOKEN_DATASET` from the environment and no longer rejects a
     Kaggle launch that omits the flag; the fallback dispatcher can attempt
     Kaggle instead of pre-skipping it.
3. **Workflow** (`.github/workflows/train-pool.yml`)
   - Failure annotation now uses `if: always() && steps.train.outcome ==
     'failure'`. Without `always()` it inherited "all previous steps
     succeeded" and was skipped in exactly the failing runs it exists for.
   - `KAGGLE_KERNEL_ID` / `KAGGLE_HF_TOKEN_DATASET` are forwarded to the launch
     step (optional).
   - The provider-report step no longer double-fails a failed slice: the report
     file is still written and uploaded, the non-zero exit becomes a warning.

### Expected next scheduled run

`plan-slices` reads the Hub pointer and emits 540000 → 600000. Modal is tried
first; on the spend-limit error the dispatcher now launches Kaggle as
`<account>/tiny-cats-dit-slice`, which resumes from the Hub checkpoint at
540000 and reports through the same provider-report/verify pipeline.

## Issue #204: pin browser artifacts and verify integrity

The model contract/manifest and digest verification landed with #195–#207
(`frontend/src/modelAssets.ts`, `frontend/src/generated/modelManifest.json`,
Web Crypto check before session creation, and the e2e test
`@runtime rejects a stale bundled fallback digest`). Remaining gaps closed
here:

- `scripts/generate_frontend_metadata.py --package-manifest
  artifacts/manifest.json` generates/validates the release manifest from the
  *verified* artifact package: a bundled model whose `(sha256, bytes)` is not
  listed in the package manifest is refused.
- Tests (`tests/test_frontend_metadata.py`) cover digest/size mismatch,
  mutable revisions (`main`, tags) being rejected, 40-hex pinning, unverified
  package bytes, and `--check` detecting a stale artifact or revision.
- `hubRevision` stays `null` until a release uploads the exact ONNX bytes and
  records the resulting Hub commit SHA; the browser then makes no request to a
  mutable `main` ref and uses digest-verified bundled files.

## Verification

```bash
bash .agents/skills/token_safe_exec.sh \
  "python -m pytest tests/test_providers.py tests/test_kaggle_provider.py tests/test_frontend_metadata.py -q"
python .agents/skills/smart_lint.py \
  "python -m ruff check src/providers.py src/kaggle_provider.py src/provider_auto.py scripts/generate_frontend_metadata.py tests/test_frontend_metadata.py tests/test_providers.py tests/test_kaggle_provider.py"
```

## Follow-up execution status (2026-10-31)

- **Blocked-Modal → Kaggle end to end:** done. `tests/test_pool_fallback_e2e.py`
  runs the real dispatcher and kernel packaging with only the Kaggle client
  faked: the blocked Modal command falls back, the Kaggle launch carries no
  `--kernel-id`, the generated `kernel-metadata.json` resolves to
  `<authenticated-account>/tiny-cats-dit-slice`, the slice keeps
  `--steps 540000` / `--schedule-steps 600000` / `--hub-resume`, and the
  provider register records `kaggle`. A second test asserts the workflow's
  failure annotation uses `always() &&`.
- **Live pool dispatch:** not run from this session. The workflow changes are
  uncommitted and `feat/open-issues-contract-hardening` does not exist on
  `origin` (GitHub API: 404), so `gh workflow run --ref` would execute the
  pre-fix workflow and stall at the same kernel-id gate. Dispatching also
  launches a real ~6 h GPU slice, so it needs an explicit commit/push decision.
- **Manifest pinning:** blocked locally — `huggingface_hub.get_token()` is
  empty and the model repo is private (401 anonymously), so no immutable
  commit SHA can be resolved. Once the exact bundled ONNX files are uploaded,
  pin with `python scripts/generate_frontend_metadata.py --revision <40-hex>`
  followed by `--check`; until then the frontend makes no mutable `main`
  request and uses digest-verified bundled bytes.

## Follow-ups

- After the next Kaggle slice succeeds, set the `KAGGLE_KERNEL_ID` repository
  variable only if a non-default kernel name is wanted; it is now optional.
- Pin the production manifest (`--revision <40-hex>`) as part of the next model
  release so remote artifacts are traceable independently of frontend builds.
