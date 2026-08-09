import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from tsad.datasets import load_npz_dataset
from tsad.evaluation.protocol import evaluate_stream


def parse_seeds(seeds_arg: str | list[int]) -> list[int]:
    """Parse seeds argument which can be a range ('0-9'), comma-separated ('0,1,2'), or integer list."""
    if isinstance(seeds_arg, list):
        return seeds_arg
    seeds_arg = seeds_arg.strip()
    if "-" in seeds_arg and "," not in seeds_arg:
        start, end = map(int, seeds_arg.split("-"))
        return list(range(start, end + 1))
    return [int(s.strip()) for s in seeds_arg.split(",") if s.strip()]


def _compute_mean_and_95ci(vals: list[float] | np.ndarray) -> dict[str, float]:
    """Compute mean and 95% confidence interval via normal approximation."""
    arr = np.asarray(vals, dtype=np.float64)
    n = len(arr)
    mean_val = float(np.mean(arr))
    if n > 1:
        std_val = float(np.std(arr, ddof=1))
        se = std_val / np.sqrt(n)
        ci_margin = float(1.96 * se)
    else:
        std_val = 0.0
        se = 0.0
        ci_margin = 0.0
    return {
        "mean": mean_val,
        "std": std_val,
        "n": n,
        "ci_margin": ci_margin,
        "ci_lower": mean_val - ci_margin,
        "ci_upper": mean_val + ci_margin,
    }


def _make_json_serializable(obj: Any) -> Any:
    """Recursively convert numpy types into standard Python types for JSON serialization."""
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int8)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return [_make_json_serializable(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): _make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_serializable(x) for x in obj]
    return obj


def run_single_benchmark(
    signal: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int = 42,
    max_points: int = 5000,
    warmup_fraction: float = 0.2,
    n_surrogates: int = 20,
) -> dict[str, Any]:
    return evaluate_stream(
        signal,
        labels,
        max_points=max_points,
        warmup_fraction=warmup_fraction,
        n_surrogates=n_surrogates,
        seed=seed,
    )


def run_cwru_stratified_benchmark(
    raw_dir: str,
    seeds: list[int] | None = None,
    allow_synthetic: bool = False,
    max_points: int = 5000,
    warmup_fraction: float = 0.2,
    n_surrogates: int = 20,
) -> dict[str, Any] | None:
    """Run stratified benchmark across all generated CWRU fault-onset stream files."""
    if seeds is None:
        seeds = list(range(10))

    cwru_dir = os.path.join(raw_dir, "cwru")
    dataset_files = sorted(glob.glob(os.path.join(cwru_dir, "cwru_*.npz")))

    if not dataset_files:
        print(f"No CWRU .npz datasets found in {cwru_dir}.")
        print("Please run `python data/download.py --cwru-all` first.")
        return None

    print(f"\nFound {len(dataset_files)} CWRU stream file(s): {[os.path.basename(f) for f in dataset_files]}")
    print(f"Running CWRU benchmark across seeds: {seeds} ({len(seeds)} seed(s) per condition)\n")

    full_results: dict[str, Any] = {
        "method_ci": "normal_approximation_95_percent",
        "seeds": seeds,
        "max_points": max_points,
        "warmup_fraction": warmup_fraction,
        "n_surrogates": n_surrogates,
        "conditions": {},
    }

    condition_edges: dict[str, float] = {}

    for dataset_file in dataset_files:
        cond_name = Path(dataset_file).stem
        signal, labels, manifest = load_npz_dataset(
            dataset_file,
            allow_synthetic=allow_synthetic,
        )

        cond_seed_results = {}
        cond_system_vus_roc = []
        cond_predictive_edge = []
        cond_p_values = []

        print(f"Processing Condition {cond_name} ...", end="", flush=True)
        for s in seeds:
            res = run_single_benchmark(
                signal,
                labels,
                seed=s,
                max_points=max_points,
                warmup_fraction=warmup_fraction,
                n_surrogates=n_surrogates,
            )
            cond_seed_results[str(s)] = res
            cond_system_vus_roc.append(res["system_vus_roc"])
            cond_predictive_edge.append(res["predictive_edge"])
            cond_p_values.append(res["surrogate_p_value"])

        summary_vus = _compute_mean_and_95ci(cond_system_vus_roc)
        summary_edge = _compute_mean_and_95ci(cond_predictive_edge)
        summary_p = _compute_mean_and_95ci(cond_p_values)

        full_results["conditions"][cond_name] = {
            "manifest": manifest,
            "seeds": cond_seed_results,
            "summary": {
                "system_vus_roc": summary_vus,
                "predictive_edge": summary_edge,
                "surrogate_p_value": summary_p,
            },
        }
        condition_edges[cond_name] = summary_edge["mean"]
        print(" Done.")

    header = (
        f"{'Condition':<35} | {'Runs':<5} | {'System VUS-ROC (Mean ± 95% CI)':<32} | "
        f"{'Predictive Edge (Mean ± 95% CI)':<33} | {'p-value (Mean ± 95% CI)':<25}"
    )
    divider = "-" * len(header)

    print("\n" + "=" * len(header))
    print(" CWRU STRATIFIED STREAMING BENCHMARK RESULTS (95% CI via Normal Approx)")
    print("=" * len(header))
    print(header)
    print(divider)

    for cond_name, cdata in full_results["conditions"].items():
        s_vus = cdata["summary"]["system_vus_roc"]
        s_edge = cdata["summary"]["predictive_edge"]
        s_p = cdata["summary"]["surrogate_p_value"]

        vus_str = f"{s_vus['mean']:.4f} ± {s_vus['ci_margin']:.4f}"
        edge_str = f"{s_edge['mean']:+.4f} ± {s_edge['ci_margin']:.4f}"
        p_str = f"{s_p['mean']:.4f} ± {s_p['ci_margin']:.4f}"
        if s_p["mean"] < 0.05:
            p_str += " *"

        print(f"{cond_name:<35} | {s_vus['n']:<5} | {vus_str:<32} | {edge_str:<33} | {p_str:<25}")

    print(divider)

    if condition_edges:
        min_cond = min(condition_edges, key=lambda k: condition_edges[k])
        min_edge_val = condition_edges[min_cond]
        print(f"\nWeakest predictive edge observed: '{min_cond}' with mean predictive edge = {min_edge_val:+.4f}")
        if "ball" in min_cond and "0.007" in min_cond:
            print("FIELD EVALUATION SUMMARY: Confirmed. Ball fault at 0.007\" severity showed the weakest predictive edge, matching field literature.")
        else:
            print(f"FIELD EVALUATION SUMMARY: Weakest predictive edge was observed for condition '{min_cond}' (mean edge: {min_edge_val:+.4f}).")

    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../results"))
    os.makedirs(results_dir, exist_ok=True)
    json_path = os.path.join(results_dir, "cwru_benchmark_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_make_json_serializable(full_results), f, indent=2)
    print(f"Saved CWRU stratified benchmark results to {json_path}\n")

    return full_results


def main():
    parser = argparse.ArgumentParser(description="Run provenance-checked TSAD benchmarks.")
    parser.add_argument(
        "--allow-synthetic",
        action="store_true",
        help="allow explicitly marked synthetic datasets for smoke testing only",
    )
    parser.add_argument("--max-points", type=int, default=5000)
    parser.add_argument("--warmup-fraction", type=float, default=0.2)
    parser.add_argument("--surrogates", type=int, default=20)
    parser.add_argument(
        "--seeds",
        type=str,
        default="0-9",
        help="Seeds to run for IAAFT surrogate generation (e.g. '0-9' or '0,1,2')",
    )
    parser.add_argument(
        "--run-cwru",
        action="store_true",
        help="Optionally run the CWRU benchmark path as well",
    )
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    raw_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/raw"))
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../results"))
    os.makedirs(results_dir, exist_ok=True)

    physio_dir = os.path.join(raw_dir, "physionet")
    dataset_files = sorted(glob.glob(os.path.join(physio_dir, "*.npz")))

    if not dataset_files and not args.run_cwru:
        print(f"No PhysioNet .npz datasets found in {physio_dir}.")
        print("Please run `python data/download.py` first.")
        sys.exit(1)

    if dataset_files:
        print(f"Found {len(dataset_files)} PhysioNet record(s): {[os.path.basename(f) for f in dataset_files]}")
        print(f"Running benchmark across seeds: {seeds} ({len(seeds)} seed(s) per record)\n")

        full_results: dict[str, Any] = {
            "method_ci": "normal_approximation_95_percent",
            "seeds": seeds,
            "max_points": args.max_points,
            "warmup_fraction": args.warmup_fraction,
            "n_surrogates": args.surrogates,
            "records": {},
            "pooled_summary": {},
        }

        all_system_vus_roc = []
        all_predictive_edge = []
        all_p_values = []

        for dataset_file in dataset_files:
            record_name = Path(dataset_file).stem
            signal, labels, manifest = load_npz_dataset(
                dataset_file,
                allow_synthetic=args.allow_synthetic,
            )

            record_seed_results = {}
            record_system_vus_roc = []
            record_predictive_edge = []
            record_p_values = []

            print(f"Processing Record {record_name} ...", end="", flush=True)
            for s in seeds:
                res = run_single_benchmark(
                    signal,
                    labels,
                    seed=s,
                    max_points=args.max_points,
                    warmup_fraction=args.warmup_fraction,
                    n_surrogates=args.surrogates,
                )
                record_seed_results[str(s)] = res
                record_system_vus_roc.append(res["system_vus_roc"])
                record_predictive_edge.append(res["predictive_edge"])
                record_p_values.append(res["surrogate_p_value"])

                all_system_vus_roc.append(res["system_vus_roc"])
                all_predictive_edge.append(res["predictive_edge"])
                all_p_values.append(res["surrogate_p_value"])

            summary_vus = _compute_mean_and_95ci(record_system_vus_roc)
            summary_edge = _compute_mean_and_95ci(record_predictive_edge)
            summary_p = _compute_mean_and_95ci(record_p_values)

            full_results["records"][record_name] = {
                "manifest": manifest,
                "seeds": record_seed_results,
                "summary": {
                    "system_vus_roc": summary_vus,
                    "predictive_edge": summary_edge,
                    "surrogate_p_value": summary_p,
                },
            }
            print(" Done.")

        pooled_vus = _compute_mean_and_95ci(all_system_vus_roc)
        pooled_edge = _compute_mean_and_95ci(all_predictive_edge)
        pooled_p = _compute_mean_and_95ci(all_p_values)

        pooled_edge["excludes_zero"] = bool(
            pooled_edge["ci_lower"] > 0 or pooled_edge["ci_upper"] < 0
        )

        full_results["pooled_summary"] = {
            "n_total_runs": len(all_system_vus_roc),
            "n_records": len(dataset_files),
            "system_vus_roc": pooled_vus,
            "predictive_edge": pooled_edge,
            "surrogate_p_value": pooled_p,
        }

        header = (
            f"{'Record':<10} | {'Runs':<5} | {'System VUS-ROC (Mean ± 95% CI)':<32} | "
            f"{'Predictive Edge (Mean ± 95% CI)':<33} | {'p-value (Mean ± 95% CI)':<25}"
        )
        divider = "-" * len(header)

        print("\n" + "=" * len(header))
        print(" MIT-BIH STREAMING BENCHMARK AGGREGATE RESULTS (95% CI via Normal Approx)")
        print("=" * len(header))
        print(header)
        print(divider)

        sig_records_count = 0
        for record_name, rdata in full_results["records"].items():
            s_vus = rdata["summary"]["system_vus_roc"]
            s_edge = rdata["summary"]["predictive_edge"]
            s_p = rdata["summary"]["surrogate_p_value"]

            vus_str = f"{s_vus['mean']:.4f} ± {s_vus['ci_margin']:.4f}"
            edge_str = f"{s_edge['mean']:+.4f} ± {s_edge['ci_margin']:.4f}"
            p_str = f"{s_p['mean']:.4f} ± {s_p['ci_margin']:.4f}"

            if s_p["mean"] < 0.05:
                sig_records_count += 1
                p_str += " *"

            print(f"{record_name:<10} | {s_vus['n']:<5} | {vus_str:<32} | {edge_str:<33} | {p_str:<25}")

        print(divider)
        p_vus_str = f"{pooled_vus['mean']:.4f} ± {pooled_vus['ci_margin']:.4f}"
        p_edge_str = f"{pooled_edge['mean']:+.4f} ± {pooled_edge['ci_margin']:.4f}"
        p_p_str = f"{pooled_p['mean']:.4f} ± {pooled_p['ci_margin']:.4f}"
        if pooled_p["mean"] < 0.05:
            p_p_str += " *"

        print(
            f"{'POOLED':<10} | {pooled_vus['n']:<5} | {p_vus_str:<32} | "
            f"{p_edge_str:<33} | {p_p_str:<25}"
        )
        print("=" * len(header))
        print(" * indicates mean surrogate p-value < 0.05")
        print(f" Statistically significant predictive edge records (p < 0.05): {sig_records_count} / {len(dataset_files)}")
        print(f" Pooled 95% CI for Predictive Edge: [{pooled_edge['ci_lower']:+.4f}, {pooled_edge['ci_upper']:+.4f}]")
        print(f" Pooled 95% CI excludes zero: {pooled_edge['excludes_zero']}\n")

        json_path = os.path.join(results_dir, "physionet_benchmark_results.json")
        serializable_results = _make_json_serializable(full_results)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(serializable_results, f, indent=2)
        print(f"Saved full raw benchmark results to {json_path}")

    if args.run_cwru:
        run_cwru_stratified_benchmark(
            raw_dir,
            seeds=seeds,
            allow_synthetic=args.allow_synthetic,
            max_points=args.max_points,
            warmup_fraction=args.warmup_fraction,
            n_surrogates=args.surrogates,
        )


if __name__ == "__main__":
    main()
