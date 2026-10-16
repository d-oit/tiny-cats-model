# ADR-067: Kaggle as a second launchable provider (kernel launcher + auto fallback)

- **Status:** Accepted (2026-10-16)
- **Date:** 2026-10-16
- **Related:** ADR-063 (exact global-step resume), ADR-064 (canonical layout +
  Hub transport), ADR-065 (control plane provider gate), `src/kaggle_provider.py`,
  `src/provider_auto.py`, `scripts/train_kaggle.py`,
  `.github/workflows/train-pool.yml`

## Context

The 2026-10-03 → 2026-10-16 stall: every scheduled pool run died inside
`modal run` with `Workspace … has exceeded its spend limit`, and the pool had no
second provider to fall back to. ADR-065 deliberately made non-Modal providers
unlaunchable ("fail clearly, never CPU-simulate"), so the campaign simply
stopped. The training contract was already provider-agnostic (ADR-063 exact
resume, ADR-064 HF Hub transport); what was missing was a headless
launch/poll/attest adapter.

## Decision

### 1. Kaggle becomes a launchable provider

`providers.ADAPTERS["kaggle"].supported = True`, and
`providers.py launch --provider kaggle --kernel-id owner/slug …` emits a
`python src/kaggle_provider.py …` command that packages one slice as a
**private Kaggle script kernel**:

- the workspace is self-contained — every `src/*.py`, `data/download.py` and
  `scripts/train_kaggle.py` plus a generated `train.py` entrypoint with the
  slice arguments baked in, so the pushed kernel version pins exactly what ran
  (no repo clone inside the kernel);
- `kernel-metadata.json` requests GPU + internet, private visibility, and
  optionally the private dataset carrying `HF_TOKEN`;
- the launcher pushes, polls to a terminal `KernelWorkerStatus`, and mirrors
  the kernel's `*.log` output into `kaggle-training.log` for the workflow's
  log/annotation steps.

Polling trusts a terminal status only once the **pushed version** is live: a
kernel keeps reporting its previous version's status until the new one starts,
so an immediate `COMPLETE` would otherwise return before training ever ran.

### 2. Provider-agnostic evidence

`providers.py fetch-artifacts --provider <p> --name <file> --dest <dir>`
materialises the report/verify inputs: Modal reads its `dit-outputs` volume,
every Hub-backed provider resolves the canonical `latest/manifest.json` pointer
(ADR-064). The workflow no longer calls `modal volume get` inline.

### 3. `--provider auto`: fall back only on launch preconditions

`src/provider_auto.py` walks a chain (default `modal,kaggle`), streaming each
attempt. It switches providers **only** when the attempt failed before training
— matched launch-precondition patterns such as the spend-limit message. A slice
that started and then failed stays visible on its own provider instead of being
silently retried elsewhere (which would mask real failures and duplicate spend).
The provider that actually ran is recorded in `.provider-used` for the artifact
steps. Scheduled runs default to `auto`; dispatches keep their explicit
provider.

### 4. Constraints this encodes

- **One slice must fit one GitHub Actions job (360 min).** Poll deadline
  defaults to 5.5 h, kernel session timeout to 6 h; the 15k-step slice
  (~3.6 h on a T4) fits with headroom.
- **Kaggle cannot attach secrets at push time** (kaggle-api#582). `HF_TOKEN` is
  resolved inside the kernel from, in order: the environment, a private dataset
  mount (`/kaggle/input/<dataset>/HF_TOKEN` — the documented workaround), or
  Kaggle Secrets when a human attached them in the UI.
- **Manifest parity.** The kernel runs the same global `--steps` contract,
  experiment id, LR floor/horizon and batch size as the Modal slices, and
  materialises the dataset with `data/download.py` (`--force-download`) so the
  `relpath:size` dataset fingerprint matches — a repacked dataset would fail
  the resume manifest gate (ADR-063) even with identical images.
- **Hub-sync parity.** `scripts/train_kaggle.py` no longer writes the legacy
  flat layout; canonical interval pushes come from `train_dit_local`, with a
  backstop push when a run ends between push intervals.

## Setup (one-off, operator)

1. `gh secret set KAGGLE_API_TOKEN` — Kaggle API token from
   kaggle.com → Settings → API (kaggle-api 2.x reads `KAGGLE_API_TOKEN`
   directly; `~/.kaggle/access_token` works locally).
2. Create the private dataset carrying the Hub token: a Kaggle dataset (e.g.
   `d-oit/tiny-cats-hf-token`) whose contents include a file `HF_TOKEN` with
   the HuggingFace token.
3. `gh variable set KAGGLE_KERNEL_ID -b d-oit/tiny-cats-dit-slice` (any free
   slug; the first push creates the kernel) and
   `gh variable set KAGGLE_HF_TOKEN_DATASET -b d-oit/tiny-cats-hf-token`.
4. Nothing else — scheduled runs try Modal first and fall back to Kaggle.

## Verification

- Unit tests cover the kernel packaging (self-contained workspace, metadata
  fields, sentinel-token non-embedding), the slice-argument contract, the
  stale-version polling guard, push rejection paths, log mirroring, the
  dispatcher fallback matrix (blocked → fallback, training failure → no
  fallback, unconfigured provider → skip, nothing runnable → config error),
  launch-command construction for `kaggle`/`auto`, and `fetch-artifacts` for
  both sources.
- A live Kaggle push/poll is not exercised by CI tests (no credentials in the
  sandbox). The first scheduled `auto` run on a blocked Modal is the smoke
  test; the stall-tracker alerting surfaces a failure instead of staying quiet.

## Consequences

- The pool survives a Modal billing/quota block without operator action.
- Two providers must keep manifest parity: the Kaggle kernel pins the repo
  revision it packaged, and `--force-download` keeps the dataset fingerprint
  identical to the Modal runs'.
- Kaggle sessions (~9 h cap, 30 h/week quota) are shorter than Modal's
  container limit; the slice default already fits, and the poll deadline
  enforces it at the job level.
