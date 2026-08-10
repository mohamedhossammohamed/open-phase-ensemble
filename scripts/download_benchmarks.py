#!/usr/bin/env python3
"""Download all supported benchmark datasets.

Usage:
    python scripts/download_benchmarks.py [--benchmark NAME] [--force]

Without --benchmark, downloads all benchmarks that have a public URL.
Yahoo S5 requires manual download (see README).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure src is on the path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))

from tsad.benchmarks import BENCHMARK_REGISTRY, get_benchmark, list_benchmarks


def main() -> int:
    parser = argparse.ArgumentParser(description="Download benchmark datasets")
    parser.add_argument(
        "--benchmark",
        type=str,
        default=None,
        choices=list_benchmarks(),
        help="Download only the named benchmark (default: all)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if data is already present",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Root directory for benchmark data (default: data/benchmarks)",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root) if args.data_root else project_root / "data" / "benchmarks"
    data_root.mkdir(parents=True, exist_ok=True)

    names = [args.benchmark] if args.benchmark else list_benchmarks()

    for name in names:
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")
        try:
            ds = get_benchmark(name, data_root=data_root)
            if ds.is_downloaded and not args.force:
                print(f"  Already downloaded at {ds.dataset_dir}")
                continue
            print(f"  Downloading to {ds.dataset_dir} ...")
            start = time.perf_counter()
            ds.download(force=args.force)
            elapsed = time.perf_counter() - start
            print(f"  Done in {elapsed:.1f}s")
        except NotImplementedError as exc:
            print(f"  SKIP: {exc}")
        except Exception as exc:
            print(f"  ERROR: {type(exc).__name__}: {exc}")
            return 1

    print(f"\n{'='*60}")
    print("  Summary")
    print(f"{'='*60}")
    for name in list_benchmarks():
        ds = get_benchmark(name, data_root=data_root)
        status = "OK" if ds.is_downloaded else "MISSING"
        print(f"  {name:20s} {status}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
