---
name: modal-container-contract
description: >
  Diagnose and prevent Modal container startup failures where the entrypoint
  module is imported before the image paths exist. Use when a Modal run fails
  with ModuleNotFoundError at container startup, when adding or renaming a
  module imported by src/train_dit.py, or before launching a GPU slice.
  Triggers: "ModuleNotFoundError", "failed during startup", "modal run fails",
  "container hydration", "slice died immediately".
license: MIT
metadata:
  version: "1.0.0"
  tags: modal containers startup imports guardrail
---

# Modal Container Contract

## The failure class

`modal run src/train_dit.py` mounts the entrypoint at `/root/train_dit.py` and
imports it during **container hydration**, before any function body, before
`@modal.enter`, and before any `sys.path` setup inside a function. The image
carries separate copies in `/app`. A module-level local import therefore
succeeds locally (repo cwd) and fails in the container.

Observed signature (2026-11-01/02, slices 555000/570000 and the 04:39 app):

```text
Function fu-... failed during startup.
ModuleNotFoundError: No module named 'safe_checkpoint'
  File "/root/train_dit.py", line 54, in <module>
```

Cost: the slice launches, burns its scheduling window, and reports nothing.
Retries repeat the same startup failure.

## Diagnosis in 60 seconds

```bash
# 1. Read the container's own error, not the workflow's summary
modal app list
modal app logs <app-id> | tail -40

# 2. Prove it is a path problem, not a missing file
ls src/safe_checkpoint.py                      # exists in the repo?
grep -n 'add_local_file("src/safe_checkpoint.py"' src/train_dit.py

# 3. Reproduce hydration locally: no sibling modules beside the module
mkdir -p /tmp/modal-sim && cp src/train_dit.py /tmp/modal-sim/
(cd /tmp/modal-sim && python -c "import train_dit")
```

A `ModuleNotFoundError` in step 3 that does not occur from the repo root is
this contract violation.

## Fix

Two parts, both required:

1. Bootstrap `sys.path` **above** the first module-level local import, covering
   the module's own directory and the image mount:

   ```python
   _HERE = Path(__file__).resolve().parent
   for _import_root in (_HERE, Path("/app")):
       if _import_root.is_dir() and str(_import_root) not in sys.path:
           sys.path.insert(0, str(_import_root))

   from safe_checkpoint import UnsafeCheckpointError, load_torch_checkpoint
   ```

2. Add `"PYTHONPATH": "/app"` to the image `.env({...})` so any import that
   skips the bootstrap still resolves.
3. Mount the whole package — `add_local_dir("src", "/app", ignore=["__pycache__", "*.pyc"])` —
   instead of listing files. A whitelist drifted twice in this repo
   (`safe_checkpoint`, then `dataset_manifest`) and each gap killed a
   dispatched slice.

Do not "fix" this by removing the import or wrapping it in `try/except`: a
missing checkpoint verifier must fail loudly, not silently downgrade.

## Other entrypoints share the contract

- **Classifier (`src/train.py`)** mounts `model.py`, which imports
  `safe_checkpoint` at module level. The same missing mount breaks classifier
  slices; keep the mount and `PYTHONPATH=/app` in that image too.
- **Kaggle (`src/kaggle_provider.py`)** ships a workspace archive rather than
  an image: every `src/*.py` plus `EXTRA_KERNEL_FILES` must be in the pack, or
  the kernel dies after the queue wait.

The sensor covers all three, including local imports reachable transitively
from any mounted module.

## Sensor

```bash
python scripts/check_modal_container_contract.py
npx -y do-harness verify --set verification --changed
```

The sensor checks mounts (including transitive local imports), `PYTHONPATH`,
bootstrap ordering, a foreign-cwd import, and the Kaggle pack.
`tests/test_modal_container_contract.py` and the CI test job run it.

## Rules for future changes

- Keep `add_local_dir("src", ...)` in both images. Do not reintroduce a
  per-file whitelist: it drifts silently and the failure surfaces only after
  dispatch.
- Any import above the bootstrap must be stdlib or third-party; local modules
  go below it.
- Prefer lazy imports inside functions for modules only one code path needs.
- Never rely on `os.chdir` or `@modal.enter` to make module-level imports work.
- Verify with the sensor before dispatching a GPU slice; a startup failure
  costs more than any local test loop.
