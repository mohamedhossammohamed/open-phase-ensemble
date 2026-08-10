#!/usr/bin/env python3
"""Run industry-standard benchmarks for open-phase-ensemble.

Usage:
    python scripts/run_benchmarks.py --benchmark TSB-AD-U --split eval
    python scripts/run_benchmarks.py --benchmark TSB-AD-U --split eval --max-series 10
    python scripts/run_benchmarks.py --benchmark TSB-AD-U --split eval --tsb-ad-metrics
    python scripts/run_benchmarks.py --all

Scientific honesty protocol:
1. Hyperparameter tuning is performed ONLY on the 'train' (tuning) split.
2. Final results are reported on the 'eval' split.
3. Primary metric is VUS-PR (NeurIPS 2024 / PVLDB 2022).
4. A chronological warm-up is applied to the evaluation split.
5. Results are saved with full provenance (checksums, config, timestamps).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure src is on the path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))

from tsad.benchmarks import (
    BenchmarkConfig,
    BenchmarkRunner,
    TSADPipelineWrapper,
    get_benchmark,
    list_benchmarks,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run industry-standard TSAD benchmarks"
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default=None,
        choices=list_benchmarks(),
        help="Benchmark to run (default: TSB-AD-U)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all downloaded benchmarks",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="eval",
        choices=["eval", "train"],
        help="Split to evaluate on (default: eval). 'train' = tuning split.",
    )
    parser.add_argument(
        "--max-series",
        type=int,
        default=None,
        help="Limit number of series (for smoke testing)",
    )
    parser.add_argument(
        "--warmup-fraction",
        type=float,
        default=0.05,
        help="Fraction of eval split to use as warm-up (default: 0.05)",
    )
    parser.add_argument(
        "--max-buffer",
        type=int,
        default=15,
        help="VUS max buffer / sliding window (default: 15)",
    )
    parser.add_argument(
        "--n-surrogates",
        type=int,
        default=0,
        help="Number of IAAFT surrogates for significance testing (default: 0)",
    )
    parser.add_argument(
        "--tsb-ad-metrics",
        action="store_true",
        help="Compute full TSB-AD metric set (requires TSB-AD installed)",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=10000,
        help="Max points per series before downsampling (default: 10000, 0=disable)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results (default: results/benchmarks)",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Root directory for benchmark data (default: data/benchmarks)",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root) if args.data_root else project_root / "data" / "benchmarks"
    output_dir = Path(args.output_dir) if args.output_dir else project_root / "results" / "benchmarks"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.all:
        names = list_benchmarks()
    elif args.benchmark:
        names = [args.benchmark]
    else:
        names = ["TSB-AD-U"]

    config = BenchmarkConfig(
        warmup_fraction=args.warmup_fraction,
        max_buffer=args.max_buffer,
        n_surrogates=args.n_surrogates,
        compute_tsb_ad_metrics=args.tsb_ad_metrics,
        max_series=args.max_series,
        downsample_max_points=args.downsample if args.downsample > 0 else None,
    )

    all_results = {}
    for name in names:
        print(f"\n{'='*60}")
        print(f"  Benchmark: {name}")
        print(f"  Split: {args.split}")
        print(f"{'='*60}")

        ds = get_benchmark(name, data_root=data_root)
        if not ds.is_downloaded:
            print(f"  SKIP: not downloaded. Run scripts/download_benchmarks.py --benchmark {name}")
            continue

        model = TSADPipelineWrapper()
        runner = BenchmarkRunner(ds, model, config)

        print(f"  Running {model.name} on {name} ({args.split} split)...")
        start = time.perf_counter()
        try:
            result = runner.run()
        except Exception as exc:
            print(f"  ERROR: {type(exc).__name__}: {exc}")
            all_results[name] = {"error": str(exc)}
            continue

        elapsed = time.perf_counter() - start
        n_success = sum(1 for r in result.series_results if r.error is None)
        n_fail = sum(1 for r in result.series_results if r.error is not None)

        print(f"  Completed in {elapsed:.1f}s ({n_success} success, {n_fail} fail)")
        print(f"\n  Aggregate metrics:")
        for k in sorted(result.aggregate.keys()):
            if k.endswith("_mean"):
                print(f"    {k:40s} {result.aggregate[k]:.4f}")

        # Save result
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        result_path = output_dir / f"{name}_{args.split}_{timestamp}.json"
        result.save(result_path)
        print(f"\n  Results saved to {result_path}")
        all_results[name] = {"result_path": str(result_path), "aggregate": result.aggregate}

    # Print summary table
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Benchmark':20s} {'VUS-PR (mean)':>15s} {'VUS-ROC (mean)':>15s}")
    print(f"  {'-'*50}")
    for name, info in all_results.items():
        if "error" in info:
            print(f"  {name:20s} {'ERROR':>15s}")
            continue
        agg = info.get("aggregate", {})
        vus_pr = agg.get("vus_pr_mean", float("nan"))
        vus_roc = agg.get("vus_roc_mean", float("nan"))
        print(f"  {name:20s} {vus_pr:>15.4f} {vus_roc:>15.4f}")

    # Save summary
    summary_path = output_dir / f"summary_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}.json"
    summary_path.write_text(json.dumps(all_results, indent=2, default=float) + "\n", encoding="utf-8")
    print(f"\n  Summary saved to {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
