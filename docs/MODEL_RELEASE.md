# Browser Model Releases: Promotion, Verification and Rollback

Issue #204: the frontend must never run a model it cannot trace to an
immutable release. Every browser model is described by
`frontend/src/generated/modelManifest.json`, generated from verified artifact
bytes:

| Field | Meaning |
|-------|---------|
| `hub.revision` | Full 40-character Hub commit SHA, or `null` for a bundled-only release |
| `hub.immutable` | `true` only when `hub.revision` is a full commit SHA |
| `models.<key>.sha256` | SHA-256 of the exact artifact bytes |
| `models.<key>.size` | Exact byte size |
| `models.<key>.contract` | Labels/conditioning, preprocessing and architecture |
| `models.<key>.contractHash` | SHA-256 of the canonical contract JSON |
| `models.<key>.onnx` | ONNX inputs, outputs and opset when the graph is readable |
| `policy` | `allowMutableRevision: false`, mandatory SHA-256/size, digest-matched local fallback |

`frontend/src/modelAssets.ts` validates that policy at module load, then
verifies size and SHA-256 with Web Crypto before either worker creates an ONNX
session. The Hub URL is built only from a full commit SHA (`resolve/<sha>`);
the bundled file is a fallback only when it matches the same digest and size.
A stale bundle is a hard error, never a silent downgrade.

## Promoting a release

1. **Package the exact artifacts.** Package the verified package (the
   generator plus, for a browser release, the classifier ONNX files):

   ```bash
   python - <<'PY'
   from artifacts import package_final_artifacts
   package_final_artifacts(
       "outputs",
       raw_checkpoint="outputs/checkpoints/pool/dit_model.pt",
       onnx="outputs/export/model.onnx",
       quantized="outputs/export/generator_quantized.onnx",
       classifier_onnx="outputs/export/cats_classifier.onnx",
       classifier_quantized="outputs/export/cats_quantized.onnx",
       require_completed=400000,
   )
   PY
   ```

   This writes `outputs/artifacts/manifest.json`, the
   **verified artifact package manifest**. `package_final_artifacts` refuses to
   publish below the required step count.

2. **Upload those exact bytes** to `d4oit/tiny-cats-model` and capture the
   returned immutable commit SHA (`src/upload_to_hub.py` and
   `src/upload_to_huggingface.py` return it as `revision`). Never use a branch
   name, a tag, or a short SHA.

3. **Generate the pinned manifest**, sourcing digests/sizes from the package
   manifest and checking the uploaded bytes:

   ```bash
   python scripts/generate_frontend_metadata.py \
     --package-manifest outputs/artifacts/manifest.json \
     --revision <40-character-commit-sha> \
     --verify-hub
   ```

   `--verify-hub` downloads every artifact from
   `https://huggingface.co/d4oit/tiny-cats-model/resolve/<sha>/...` and compares
   SHA-256 and size before writing; `HF_TOKEN` is used when the repository is
   private. Without `--verify-hub` a pinned revision is refused, unless the
   offline escape hatch `--no-verify-hub` is passed explicitly. The generated
   manifest is also compared against the bundled bytes, so a stale
   `frontend/public/models/` file cannot be promoted.

   The release is appended to `frontend/src/generated/modelReleases.json`
   (revision, per-model digests/sizes, package manifest digest, timestamp) so
   model history is traceable independently of frontend releases.

4. **Commit the artifacts, contract, manifest and release history together**
   and let CI prove they are consistent:

   ```bash
   python scripts/generate_frontend_metadata.py --check
   ```

   `--check` re-hashes the bundled artifacts against the committed manifest,
   verifies the contract matches `src/model_contract.py`, and fails on drift.
   CI runs it on every PR (see `.github/workflows/ci.yml`).

## Rolling back

Rollback re-pins a previously recorded release instead of hand-editing a SHA:

```bash
# Inspect recorded releases
python -c "import json; print(json.dumps(json.load(open('frontend/src/generated/modelReleases.json')), indent=2))"

# Restore the matching model files, then re-pin the recorded revision
python scripts/generate_frontend_metadata.py --rollback <40-character-commit-sha>
# or the latest recorded release
python scripts/generate_frontend_metadata.py --rollback
```

The rollback fails when the bundled artifacts do not match the recorded
digests and sizes, so a rollback can never pair new bytes with an old
revision. Run `--check` afterwards and commit the restored artifacts with the
regenerated manifest.

## Failure modes and their guarantees

| Situation | Behaviour |
|-----------|-----------|
| No revision pinned (`revision: null`) | No Hub request at all; bundled bytes are digest-verified |
| Hub bytes differ from the manifest | Hub fetch rejected, then bundled fallback verified; both failing is a hard error |
| Bundled bytes stale/different | Hard error (`digest`/`size` mismatch) before any ONNX session |
| Manifest allows mutable revisions or omits digests | `validateModelManifest` throws at module load |
| Hand-edited manifest | `--check` fails in CI; runtime digest verification still rejects the bytes |
| Upload result without a commit SHA | Upload logs a warning; no release is recorded |

## See also

- `frontend/public/models/README.md` — artifact layout and runtime notes
- `docs/RELEASE_RUNBOOK.md` — checkpoint publication runbook (training side)
- `scripts/generate_frontend_metadata.py` — manifest/release generator
- `src/model_release.py` — revision validation, package verification, history
