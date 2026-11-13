# 600k DiT release: evaluation evidence

## Release identity

- Workflow: [Release Pool Checkpoint, run 43444751713](https://github.com/d-oit/tiny-cats-model/actions/runs/43444751713), completed successfully on 2026-11-13.
- Target: 600,000 steps; experiment `dit-breed-conditioned-v4`; source pointer `step-600000`.
- Source Hub revision: `6d2a2ca12496bfba6a89f7c1f5bb34712758eeee`.
- Published immutable Hub revision: `e204fdd0b905de7cdd6807c27487896c0159413d`.
- Packaged raw checkpoint SHA-256: `30a7bfa439a670140be47c2307c33c8f192fcd4f438107a1556e69022a568dae`.
- Canonical package-manifest SHA-256 (as recorded in release evidence): `c16a45a5370bdc63b1df42f0994f2817b9ac9843f61aa19e5b74ad4208b87211`.
- Downloaded Actions artifact `release-600000` SHA-256: `e3a6fa59389aa8df6477d67afec494a0da330e93d5b67c902d89e250c1bab08f`.
- Bundled generator: `frontend/public/models/generator_quantized.onnx`, 34,161,922 bytes, SHA-256 `692c9779cd7871283ae66b87026083576a0a44f5d5d88d925e8bf597e74d2dd4`.
- Bundled classifier: `frontend/public/models/cats_quantized.onnx`, 11,232,526 bytes, SHA-256 `6fa0ca5d0242d8ec1532811db2183f208118b695a161567f428df51e5f977df3` (unchanged from the existing bundle).
- The release manifest pins both browser models to the published revision and verifies size and SHA-256 before creating an ONNX session. The workflow also verified the pool pointer bytes at the published revision; the pointer remains at `step-600000`.

The release exporter uses the raw `dit_model.pt` for the generator export. The package also contains the EMA checkpoint, but this does not identify which weights produced either historical report below.

## Historical statistical results

The historical reports are `eval_out/samples-531k/report.json` and `eval_out/samples-600k-baseline/report.json`. The original reports and generated samples were left intact; the 600k report is copied here verbatim:

```json
{
  "fid": 263.56637437215534,
  "inception_score": {
    "mean": 1.5683199167251587,
    "std": 0.05481218174099922
  },
  "precision": 0.5,
  "recall": 0.43916666666666665,
  "metadata": {
    "step": 600000,
    "loss": 0.5879713714122772,
    "timestamp": "2026-11-03T04:40:53.212863",
    "config": {
      "image_size": 128,
      "patch_size": 16,
      "embed_dim": 384,
      "depth": 12,
      "num_heads": 6,
      "num_classes": 13
    },
    "num_samples_generated": 988
  }
}
```

The 531,238-step reference report, also copied verbatim:

```json
{
  "fid": 262.9977929759247,
  "inception_score": {
    "mean": 1.5941845178604126,
    "std": 0.061940617859363556
  },
  "precision": 0.5,
  "recall": 0.46791666666666665,
  "metadata": {
    "step": 531238,
    "loss": 0.5876103854179382,
    "timestamp": "2026-10-03T00:07:24.184567",
    "config": {
      "image_size": 128,
      "patch_size": 16,
      "embed_dim": 384,
      "depth": 12,
      "num_heads": 6,
      "num_classes": 13
    },
    "num_samples_generated": 988
  }
}
```

| Metric | 531,238 steps | 600,000 steps | Change (600k − 531k) |
|---|---:|---:|---:|
| FID (lower is better) | 262.9978 | 263.5664 | +0.5686 |
| Inception Score mean | 1.5942 | 1.5683 | −0.0259 |
| Inception Score std | 0.0619 | 0.0548 | −0.0071 |
| Precision | 0.5000 | 0.5000 | 0.0000 |
| Recall | 0.4679 | 0.4392 | −0.0288 |

These results are broadly flat, with unchanged precision and lower recall at 600k; FID is slightly higher and Inception Score mean slightly lower. The reports provide no significance test or confidence interval for FID, precision, or recall. No improvement or regression claim is warranted.

Both reports record 988 generated samples and the same 128×128, 13-class architecture. The documented sampling protocol was 76 samples for each of 13 conditions, 30 Euler steps, CFG 1.5, and seed 42. The report JSON itself does not store the sampler, per-condition counts, step count, CFG scale, or seed, so these settings cannot be reconstructed from the reports alone. Neither historical report records a checkpoint filename or hash, nor whether raw or EMA weights were evaluated; the statistical results therefore cannot be attributed to the exact checkpoint in this release.

## Separate release smoke and timing results

The successful release workflow generated one CPU sample for each of the 13 breed conditions. All 13 PNGs are present in `outputs/artifacts/evaluation/samples/`; `evaluation_report.json` records `num_samples_generated: 13` and checkpoint metadata, but contains no FID, Inception Score, precision, or recall. This smoke output is not comparable to the historical 988-sample statistical runs and is not included in the table above.

The local browser smoke requested both models from the pinned Hub revision; both requests returned HTTP 401, then the UI reported `Verified model · local · WASM`, confirming digest-checked bundled fallback. Generation with breed index 0 (Abyssinian), seed 42, 30 steps, and CFG 1.5 completed with a 128×128 PNG, and the Download control produced a PNG file.

Visual quality is a release limitation: the browser image was noise-like with no recognizable cat. The inspected CPU smoke image and historical 600k Abyssinian sample are likewise noise-like. The browser smoke proves the verified inference path and UI complete, not useful image quality; no quality improvement is claimed.

The separate classifier smoke used `Sphynx_56.jpg`; the bundled classifier classified it as Cat with 84.7% confidence and displayed finite probabilities of 84.7% Cat and 15.3% Not a Cat. This verifies the local classifier fallback and output path, not classifier accuracy.

The separate CPU PyTorch inference benchmark (`outputs/artifacts/evaluation/benchmark_report.json`, 5 warmups and 20 measured runs) recorded:

| Measurement | Result |
|---|---:|
| Latency p50 | 38.168 ms |
| Latency p95 | 44.903 ms |
| Latency p99 | 48.798 ms |
| Mean latency | 39.154 ms |
| Throughput, batch 1 | 25.790 images/s |
| Throughput, batch 4 | 39.286 images/s |
| Throughput, batch 8 | 43.222 images/s |

These are CPU PyTorch timings, not browser ONNX or end-to-end generation timings, and are kept separate from the statistical comparison.

## Artifacts

The release artifact contains `outputs/release-evidence.json`, `outputs/source-pointer.json`, and the package under `outputs/artifacts/`. The release evidence records the source and published revisions, 600,000 completed steps, experiment ID, raw checkpoint SHA-256, and canonical package-manifest SHA-256. The downloaded artifact digest and every package file size/SHA-256 were independently verified before copying the browser bundle files.
