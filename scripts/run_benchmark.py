import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from tsad.datasets import load_npz_dataset
from tsad.evaluation.protocol import evaluate_stream


def run_dataset_benchmark(
    name: str,
    signal: np.ndarray,
    labels: np.ndarray,
    *,
    max_points: int = 5000,
    warmup_fraction: float = 0.2,
    n_surrogates: int = 20,
):
    result = evaluate_stream(
        signal,
        labels,
        max_points=max_points,
        warmup_fraction=warmup_fraction,
        n_surrogates=n_surrogates,
    )
    print(f"\n--- Running Benchmark on {name} ---")
    print(
        f"Points: {result['n_evaluated']} evaluated after "
        f"{result['warmup_points']} warm-up points"
    )
    print(f"System VUS-ROC: {result['system_vus_roc']:.4f}")
    print(f"System VUS-PR:  {result['system_vus_pr']:.4f}")
    print(f"Persistence VUS-ROC: {result['persistence_vus_roc']:.4f}")
    print(
        f"IAAFT VUS-ROC: {result['surrogate_vus_roc_mean']:.4f} "
        f"+/- {result['surrogate_vus_roc_std']:.4f} "
        f"(n={result['n_surrogates']})"
    )
    print(f"Predictive edge: {result['predictive_edge']:+.4f}")
    print(f"IAAFT empirical p-value: {result['surrogate_p_value']:.4f}")
    print("Per-detector VUS-ROC:", result["detector_vus_roc"])
    return result

def main(
    allow_synthetic: bool = False,
    *,
    max_points: int = 5000,
    warmup_fraction: float = 0.2,
    n_surrogates: int = 20,
):
    raw_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/raw"))
    np.random.seed(42)

    datasets = [
        os.path.join(raw_dir, "physionet/100.npz"),
        os.path.join(raw_dir, "cwru/cwru_bearing.npz"),
    ]
    for dataset_file in datasets:
        signal, labels, manifest = load_npz_dataset(
            dataset_file,
            allow_synthetic=allow_synthetic,
        )
        run_dataset_benchmark(
            manifest["name"],
            signal,
            labels,
            max_points=max_points,
            warmup_fraction=warmup_fraction,
            n_surrogates=n_surrogates,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run provenance-checked TSAD benchmarks.")
    parser.add_argument(
        "--allow-synthetic",
        action="store_true",
        help="allow explicitly marked synthetic datasets for smoke testing only",
    )
    parser.add_argument("--max-points", type=int, default=5000)
    parser.add_argument("--warmup-fraction", type=float, default=0.2)
    parser.add_argument("--surrogates", type=int, default=20)
    args = parser.parse_args()
    main(
        allow_synthetic=args.allow_synthetic,
        max_points=args.max_points,
        warmup_fraction=args.warmup_fraction,
        n_surrogates=args.surrogates,
    )
