#!/usr/bin/env python3
"""scripts/benchmark_estimates.py

Benchmark and calibrate the GPU hour estimation against real training data.

Reads training logs and ADRs to compare estimated vs actual training times,
then provides calibration recommendations for estimate_gpu_hours().

Usage:
    python scripts/benchmark_estimates.py          # Show full benchmark report
    python scripts/benchmark_estimates.py --tune   # Print tuned constants
    python scripts/benchmark_estimates.py --steps 50000  # Estimate for N steps
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gpu_pool import (
    T4_STEPS_PER_SECOND,
    estimate_cost,
    estimate_gpu_hours,
)

# ─────────────────────────────────────────────────────────────
# Known training benchmarks (from ADRs, training logs)
# ─────────────────────────────────────────────────────────────

# Each entry: (steps, batch_size, image_size, gpu_type, actual_hours, source)
# actual_hours is the wall-clock time reported in training logs/ADRs.
# Convention: source strings prefixed with "DERIVED:" are scaled/synthetic
# (e.g., A10G→T4 via known speed factors) and never count toward the
# `--tune` real-benchmark threshold.
KNOWN_BENCHMARKS: list[dict[str, Any]] = [
    {
        "steps": 400_000,
        "batch_size": 256,
        "image_size": 128,
        "gpu_type": "A10G/H100",
        "actual_hours": 42,  # midpoint of 36-48h from TRAINING_400K_LOG.md
        "source": "docs/TRAINING_400K_LOG.md",
    },
    {
        "steps": 100_000,
        "batch_size": 512,
        "image_size": 128,
        "gpu_type": "T4",
        "actual_hours": None,  # placeholder — needs real T4 wall-clock measurement
        "source": "ADR-057 (estimated, not yet measured on real T4)",
        "note": "Theoretical: 100k/2.2 = 45,454s ≈ 12.6h. Replace with real run.",
    },
    {
        "steps": 200_000,
        "batch_size": 128,
        "image_size": 128,
        "gpu_type": "T4",
        "actual_hours": None,  # placeholder — fill in from real runs
        "source": "Placeholder for real T4 200k run",
        "note": "Estimate: 200k/(2.2*512/128) = 200k/8.8 = 22,727s ≈ 6.3h",
    },
    # Derived from 400k A10G benchmark using GPU speed factors.
    # A10G is 2.5x T4, so T4 time ~ A10G time * 2.5.
    {
        "steps": 400_000,
        "batch_size": 256,
        "image_size": 128,
        "gpu_type": "T4",
        "actual_hours": 105,  # 42h(A10G) * 2.5 = 105h on T4
        "source": "DERIVED: scaled from TRAINING_400K_LOG.md (A10G->T4, 2.5x)",
    },
]

# Speed factors relative to T4 baseline
GPU_SPEED_FACTORS: dict[str, float] = {
    "T4": 1.0,  # baseline (T4_STEPS_PER_SECOND from gpu_pool)
    "L4": 1.3,  # ~30% faster than T4
    "A10G": 2.5,  # ~2.5x T4
    "H100": 4.0,  # ~4x T4
    "V100": 1.5,  # ~1.5x T4
    "P100": 0.9,  # slightly slower than T4
    "CPU": 0.02,  # 50x slower
}

# Minimum number of REAL (non-derived) benchmarks in KNOWN_BENCHMARKS
# required to recommend a tune. Derived entries (synthesized via
# speed-factor scaling from another entry) don't count toward this
# threshold because they double-count their parent's information.
MIN_REAL_BENCHMARKS_FOR_TUNE: int = 3


# ─────────────────────────────────────────────────────────────
# Calibration
# ─────────────────────────────────────────────────────────────


def calibrate_steps_per_second(
    baseline_steps_per_sec: float | None = None,
) -> dict[str, Any]:
    """Calibrate and report estimation accuracy against known benchmarks.

    Iterates KNOWN_BENCHMARKS once, classifying each entry with a non-None
    ``actual_hours`` as REAL or DERIVED via the source-string heuristic
    (``source.startswith("DERIVED:")``). Mean error / recommended baseline
    are computed from REAL entries only — derived entries are too correlated
    to their parent benchmark to count as independent samples.

    Args:
        baseline_steps_per_sec: Current baseline for T4.
            Defaults to T4_STEPS_PER_SECOND from gpu_pool.

    Returns:
        Calibration report dict with keys:
          - baseline_steps_per_sec (float)
          - recommended_baseline (float)
          - mean_error_pct (float): mean of REAL entries' error
          - num_benchmarks (int): count of REAL entries with actual_hours
            (note: this used to count ALL entries, but as of the
            real/derived split it counts REAL only — derived entries are
            in num_derived)
          - num_derived (int): count of DERIVED entries with actual_hours
          - comparisons (list[dict]): per-benchmark detail records. Each
            has a boolean ``is_derived`` field so downstream consumers
            can filter/recompute as needed.
          - warning (str | None): present when real exports < MIN_REAL_BENCHMARKS_FOR_TUNE.
    """
    report: dict[str, Any] = {
        "baseline_steps_per_sec": baseline_steps_per_sec,
        "comparisons": [],
        "recommended_baseline": baseline_steps_per_sec,
        "mean_error_pct": 0.0,
    }

    # Use current T4_STEPS_PER_SECOND from gpu_pool as default
    if baseline_steps_per_sec is None:
        baseline_steps_per_sec = T4_STEPS_PER_SECOND
        report["baseline_steps_per_sec"] = T4_STEPS_PER_SECOND
        report["recommended_baseline"] = T4_STEPS_PER_SECOND

    # Single-pass over KNOWN_BENCHMARKS. Source-string heuristic
    # (`source.startswith("DERIVED:")`) classifies synthesized entries
    # so they don't pollute the real-benchmark count toward the tune
    # threshold (≥3 REAL entries required for a recommendation).
    real_errors: list[float] = []
    derived_errors: list[float] = []

    for bench in KNOWN_BENCHMARKS:
        if bench["actual_hours"] is None:
            continue

        estimated = estimate_gpu_hours(
            bench["steps"],
            batch_size=bench["batch_size"],
            image_size=bench["image_size"],
        )
        gpu_factor = GPU_SPEED_FACTORS.get(bench["gpu_type"].split("/")[0], 1.0)
        estimated_adjusted = estimated / gpu_factor
        actual = bench["actual_hours"]
        error_pct = ((estimated_adjusted - actual) / actual) * 100

        is_derived = bench.get("source", "").startswith("DERIVED:")
        (derived_errors if is_derived else real_errors).append(error_pct)

        report["comparisons"].append(
            {
                "benchmark": bench,
                "estimated_raw_hours": round(estimated, 1),
                "estimated_adjusted_hours": round(estimated_adjusted, 1),
                "actual_hours": actual,
                "error_pct": round(error_pct, 1),
                "gpu_factor": gpu_factor,
                "is_derived": is_derived,
            }
        )

    if real_errors or derived_errors:
        # Mean over REAL only — derived entries are too correlated to their
        # parent benchmark to count as independent samples.
        real_mean = (
            round(sum(real_errors) / len(real_errors), 1) if real_errors else 0.0
        )
        report["mean_error_pct"] = real_mean
        report["recommended_baseline"] = round(
            baseline_steps_per_sec * (1 + real_mean / 100), 2
        )
        report["num_benchmarks"] = len(real_errors)
        report["num_derived"] = len(derived_errors)
        if len(real_errors) < MIN_REAL_BENCHMARKS_FOR_TUNE:
            report["warning"] = (
                f"Only {len(real_errors)} REAL benchmark(s) available "
                f"(plus {len(derived_errors)} DERIVED) — calibration is "
                "suggestive, not definitive. Add more real T4 wall-clock "
                "measurements to KNOWN_BENCHMARKS."
            )

    return report


def print_calibration_report(report: dict[str, Any]) -> None:
    """Print a human-readable calibration report."""
    print("=" * 70)
    print("GPU HOUR ESTIMATION BENCHMARK")
    print("=" * 70)
    print(f"Baseline: {report['baseline_steps_per_sec']} steps/s (T4)")
    print()

    for comp in report["comparisons"]:
        bench = comp["benchmark"]
        print(f"  {bench['source']}")
        print(f"    GPU: {bench['gpu_type']} (factor: {comp['gpu_factor']}x)")
        print(
            f"    Steps: {bench['steps']:,} @ batch={bench['batch_size']} {bench['image_size']}x{bench['image_size']}"
        )
        print(f"    Raw estimate:  {comp['estimated_raw_hours']:.1f}h (T4 baseline)")
        print(
            f"    Adjusted:      {comp['estimated_adjusted_hours']:.1f}h (for {bench['gpu_type']})"
        )
        print(f"    Actual:        {comp['actual_hours']:.1f}h")
        sign = "+" if comp["error_pct"] >= 0 else ""
        print(f"    Error:         {sign}{comp['error_pct']:.1f}%")
        print()

    if report["comparisons"]:
        print(f"Mean error (real only): {report['mean_error_pct']:+.1f}%")
        real_n = report.get("num_benchmarks", 0)
        deriv_n = report.get("num_derived", 0)
        print(
            f"Benchmarks used: {real_n} real, {deriv_n} derived "
            f"(derived entries are suggestions, not independent measurements)"
        )
        if report.get("warning"):
            print(f"⚠️  {report['warning']}")
        if (
            abs(report["mean_error_pct"]) > 20
            and real_n >= MIN_REAL_BENCHMARKS_FOR_TUNE
        ):
            print(
                f"⚠️  Recommendation: tune baseline to {report['recommended_baseline']} steps/s"
            )
        elif real_n < MIN_REAL_BENCHMARKS_FOR_TUNE:
            print(
                f"⚠️  Not enough real benchmarks (≥{MIN_REAL_BENCHMARKS_FOR_TUNE}) "
                "to recommend a tune"
            )
        else:
            print("✅ Baseline is within acceptable range (≤20% error)")


def print_estimate_table(
    steps: int,
    batch_size: int = 512,
    image_size: int = 128,
) -> None:
    """Print a cost/time estimate table for all providers.

    Args:
        steps: Training steps.
        batch_size: Batch size.
        image_size: Image size.
    """
    costs = estimate_cost(steps, batch_size=batch_size, image_size=image_size)
    gpu_hours = estimate_gpu_hours(steps, batch_size, image_size)

    print(f"\n{'=' * 70}")
    print(
        f"TRAINING ESTIMATE: {steps:,} steps @ batch={batch_size} {image_size}x{image_size}"
    )
    print(f"{'=' * 70}")
    print(f"Estimated GPU hours (T4 baseline): {gpu_hours:.1f}h")
    print()
    print(f"{'Provider':<20s} {'GPU':<6s} {'Hours':>7s} {'Cost':>8s} {'Free?':>6s}")
    print("-" * 52)

    for _name, info in sorted(costs.items(), key=lambda x: x[1]["estimated_cost"]):
        free_flag = "✅" if info["within_free_tier"] else "⚠️ "
        print(
            f"{info['provider']:<20s} "
            f"{info['gpu_type']:<6s} "
            f"{info['gpu_hours']:>6.1f}h "
            f"${info['estimated_cost']:>7.2f} "
            f"{free_flag:>6s}"
        )

    print("-" * 52)
    print()


def print_tuned_constants() -> None:
    """Print recommended tuned constants for estimate_gpu_hours().

    Gating: actionable tune recommendations require ``num_real >= 3`` in
    ``KNOWN_BENCHMARKS``. Derived entries (computed via speed-factor
    scaling from another entry) are tracked separately and never count
    toward the threshold — they double-count their parent's information.
    """
    report = calibrate_steps_per_second()
    num_real = report.get("num_benchmarks", 0)
    num_derived = report.get("num_derived", 0)
    print()
    print("# Calibration status for src/gpu_pool.py:estimate_gpu_hours()")
    print(f"# Real benchmarks: {num_real}    Derived: {num_derived}")
    print(f"# Mean error (real only): {report['mean_error_pct']:+.1f}%")

    current = T4_STEPS_PER_SECOND

    # Gate: require 3+ REAL (non-derived) benchmarks before recommending.
    if num_real < MIN_REAL_BENCHMARKS_FOR_TUNE:
        print()
        print(f"# NOT recommending a tune with only {num_real} REAL benchmark(s).")
        print(
            f"# Calibration requires >= {MIN_REAL_BENCHMARKS_FOR_TUNE} real "
            "T4 wall-clock measurements."
        )
        print(
            f"# The {num_derived} derived entry(ies) are suggestive but not independent."
        )
        print(f"# Keeping STEPS_PER_SECOND_T4 = {current} (do not auto-apply).")
        print()
        return

    print()
    print(f"# Recommended: STEPS_PER_SECOND_T4 = {report['recommended_baseline']}")
    print()
    print("# GPU speed factors (multiply STEPS_PER_SECOND_T4 by these):")
    for gpu, factor in sorted(GPU_SPEED_FACTORS.items()):
        speed = report["recommended_baseline"] * factor
        print(f"#   {gpu:<6s}: {factor:4.1f}x → {speed:.1f} steps/s")
    print()


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark and calibrate GPU hour estimation"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Estimate for specific step count (shows provider table)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size for estimate",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=128,
        help="Image size for estimate",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Print recommended tuned constants",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output calibration report as JSON",
    )
    args = parser.parse_args()

    if args.json:
        report = calibrate_steps_per_second()
        print(json.dumps(report, indent=2, default=str))
        return

    if args.tune:
        print_tuned_constants()
        return

    if args.steps:
        print_estimate_table(args.steps, args.batch_size, args.image_size)
        return

    # Default: full report
    report = calibrate_steps_per_second()
    print_calibration_report(report)
    # Only show tuned constants if we have useful data
    if report.get("num_benchmarks", 0) > 0:
        print_tuned_constants()


if __name__ == "__main__":
    main()
