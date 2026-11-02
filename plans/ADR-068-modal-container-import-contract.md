# ADR-068: Modal Container Import Contract

- **Status:** Accepted
- **Date:** 2026-11-02
- **Related:** ADR-022 (image optimization), ADR-025 (`@modal.enter` cold start),
  ADR-042 (deferred imports), ADR-057 (CLI verification), ADR-065 (slices),
  issue #163

## Context

`modal run src/train_dit.py` mounts the entrypoint at `/root/train_dit.py`
inside the container and imports it during **container hydration**. That import
happens before any function body runs, before `@modal.enter` executes, and
before `os.chdir("/app")`. The image carries a second copy of every module under
`/app` via `add_local_file`.

`src/train_dit.py` has a module-level import:

```python
from safe_checkpoint import UnsafeCheckpointError, load_torch_checkpoint
```

Locally this resolves because the repo's `src/` is on `sys.path`. In the
container, hydration imports `/root/train_dit.py`, whose directory contains no
siblings, and `/app` was added to `sys.path` only later — inside the function
body. Result:

```text
Function fu-yP8oJ7IVQVD4F7C4KHqoqk failed during startup.
ModuleNotFoundError: No module named 'safe_checkpoint'
  File "/root/train_dit.py", line 54, in <module>
```

The same class had already burned two 6-hour slices on 2026-10-31 (slices
555000 and 570000). The prior fix added the missing `safe_checkpoint.py` mount
and a test that every mounted module is imported, but it did not test *when*
imports resolve. The image mount was correct; the import order was not. The
failure therefore recurred on the 2026-11-02 app (`ap-22eiZTSjSQ8chxvTD5VGA8`).

## Decision

1. **Bootstrap `sys.path` before the first module-level local import** in
   `src/train_dit.py`, covering both the module's own directory and `/app`.
2. **Set `PYTHONPATH=/app` in the image env** so image copies resolve even for
   imports that bypass the bootstrap.
3. **Mount the whole `src/` package** in both Modal images
   (`add_local_dir("src", "/app", ignore=[...])`) instead of a hand-maintained
   `add_local_file` whitelist. The whitelist drifted twice — `safe_checkpoint`
   (startup, 2026-10-31) and `dataset_manifest` (first training step,
   2026-11-02) — and each gap killed a slice that had already launched.
4. **Apply the same contract to every container entrypoint** — `src/train.py`
   (classifier) mounts `model.py`, which imports `safe_checkpoint` at module
   level, so the classifier image gets the same `safe_checkpoint.py` mount and
   `PYTHONPATH=/app`. The Kaggle slice pack is checked too: it must copy every
   `src/*.py` plus the extra kernel files the slice imports.
5. **Enforce it with a computational sensor**
   (`scripts/check_modal_container_contract.py`, wired into `do-harness.toml`
   and `tests/test_modal_container_contract.py`) that checks:
   - every module-level local import is mounted into the image;
   - every mounted `src/` module's container directory is on `PYTHONPATH`;
   - the `sys.path` bootstrap precedes the first local import;
   - the entrypoint imports from a foreign cwd with only the image path;
   - every local import reachable from a mounted module is itself mounted;
   - the Kaggle workspace pack contains the slice's local imports.

   The sensor runs in `do-harness verify`, the pre-commit hook, the CI test
   job, and `tests/test_modal_container_contract.py`.

## Consequences

- Container startup failures of this class fail locally in seconds instead of
  inside a dispatched slice.
- Adding a module that `train_dit.py` imports now requires a mount, a
  `PYTHONPATH`-reachable location, and an import below the bootstrap. The
  sensor names the missing piece.
- `@modal.enter` remains responsible for heavy runtime setup (CUDA warm-up,
  chdir); it is no longer load-bearing for import resolution.
- The checks are static plus one subprocess import, so they run in the quality
  gate and pre-commit without a GPU.
