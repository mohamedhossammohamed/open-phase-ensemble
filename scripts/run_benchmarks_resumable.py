#!/usr/bin/env python3
"""Resumable, incrementally-checkpointed benchmark runner.

Saves progress at two granularities:
  1. INTRA-SERIES: every `checkpoint_interval` points, the partial scores
     array is flushed to a .npy file. On resume, partial scores are reloaded
     and the pipeline continues from where it stopped — no re-scoring of
     already-processed points.
  2. PER-SERIES: once a series fully completes (scores + metrics computed),
     its result is appended to checkpoint.jsonl. On resume, completed series
     are skipped entirely.

Storage layout (non-redundant):
  <output_dir>/
    checkpoint.jsonl          # one line per completed series (append-only)
    partial/                  # partial scores for in-progress series
      <series_name>.npy       # overwritten each checkpoint (latest only)
      <series_name>.meta.json # point count + config for resume validation
    run_log.json              # running stats, updated after each series
    progress.log              # human-readable log (append-only)

Usage:
    python scripts/run_benchmarks_resumable.py --benchmark TSB-AD-U --split train --downsample 10000
    python scripts/run_benchmarks_resumable.py --benchmark TSB-AD-U --split train --resume
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tsad.benchmarks.base import (  # noqa: E402
    BenchmarkConfig,
    BenchmarkRunner,
    _causal_block_downsample,
)
from tsad.benchmarks.tsb_ad import TSB_AD_U  # noqa: E402
from tsad.benchmarks.wrappers import TSADPipelineWrapper  # noqa: E402
from tsad.evaluation.vus import compute_vus_pr, compute_vus_roc  # noqa: E402

# How often (in points) to flush partial scores within a single series.
CHECKPOINT_INTERVAL = 500


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_checkpoint(checkpoint_path: Path) -> dict[str, dict]:
    """Load completed series results from checkpoint.jsonl."""
    completed = {}
    if checkpoint_path.exists():
        for line in checkpoint_path.read_text().strip().split("\n"):
            if not line:
                continue
            obj = json.loads(line)
            completed[obj["name"]] = obj
    return completed


def append_checkpoint(checkpoint_path: Path, result: dict) -> None:
    """Append a completed series result to checkpoint.jsonl."""
    with checkpoint_path.open("a") as f:
        f.write(json.dumps(result, default=float) + "\n")


def save_partial(partial_dir: Path, series_name: str, scores: np.ndarray,
                 point_idx: int, config_hash: str) -> None:
    """Save partial scores for an in-progress series (overwrites previous)."""
    safe_name = series_name.replace("/", "_")
    np.save(partial_dir / f"{safe_name}.npy", scores[:point_idx])
    meta = {"series_name": series_name, "point_idx": point_idx,
            "config_hash": config_hash, "ts": utc_now()}
    (partial_dir / f"{safe_name}.meta.json").write_text(json.dumps(meta) + "\n")


def load_partial(partial_dir: Path, series_name: str, config_hash: str) -> tuple[np.ndarray, int] | None:
    """Load partial scores for resumption. Returns (scores_array, point_idx) or None."""
    safe_name = series_name.replace("/", "_")
    npy_path = partial_dir / f"{safe_name}.npy"
    meta_path = partial_dir / f"{safe_name}.meta.json"
    if not npy_path.exists() or not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    if meta.get("config_hash") != config_hash:
        return None  # stale checkpoint from a different config
    partial_scores = np.load(npy_path)
    return partial_scores, meta["point_idx"]


def clear_partial(partial_dir: Path, series_name: str) -> None:
    """Remove partial files after a series completes."""
    safe_name = series_name.replace("/", "_")
    for p in [partial_dir / f"{safe_name}.npy", partial_dir / f"{safe_name}.meta.json"]:
        if p.exists():
            p.unlink()


def config_hash(config: BenchmarkConfig) -> str:
    import hashlib
    d = {
        "warmup_fraction": config.warmup_fraction,
        "max_buffer": config.max_buffer,
        "n_surrogates": config.n_surrogates,
        "seed": config.seed,
        "compute_tsb_ad_metrics": config.compute_tsb_ad_metrics,
        "downsample_max_points": config.downsample_max_points,
    }
    return hashlib.md5(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]


def persistence_scores(signal: np.ndarray) -> np.ndarray:
    s = np.asarray(signal, dtype=np.float64)
    out = np.zeros(len(s), dtype=np.float64)
    if len(s) > 1:
        out[1:] = np.abs(np.diff(s))
    return out


def primary_metrics(scores: np.ndarray, labels: np.ndarray, max_buffer: int) -> dict[str, float]:
    return {
        "vus_pr": compute_vus_pr(scores, labels, max_buffer=max_buffer),
        "vus_roc": compute_vus_roc(scores, labels, max_buffer=max_buffer),
    }


def tsb_ad_metrics(scores: np.ndarray, labels: np.ndarray, max_buffer: int) -> dict[str, float]:
    try:
        from TSB_AD.evaluation.metrics import get_metrics
        out = get_metrics(scores, labels.astype(int), slidingWindow=max_buffer,
                          pred=None, version="opt", thre=250)
        return {f"tsb_ad_{k.lower().replace(' ', '_')}": float(v) for k, v in out.items()}
    except Exception as e:
        return {"tsb_ad_error": str(e)}


def run_series_with_checkpoint(
    series,
    model: TSADPipelineWrapper,
    config: BenchmarkConfig,
    partial_dir: Path,
    cfg_hash: str,
    progress_log,
    series_idx: int,
    total_series: int,
    series_timeout: float,
) -> dict | None:
    """Run one series with intra-series checkpointing. Returns result dict or None on timeout."""

    name = series.name
    signal = series.signal
    labels = series.labels
    train_split = series.train_split

    # Downsample
    if config.downsample_max_points is not None and len(signal) > config.downsample_max_points:
        signal, labels, stride = _causal_block_downsample(signal, labels, config.downsample_max_points)
        train_split = max(1, int(train_split / stride))

    n_total = len(signal)
    train_signal = signal[:train_split]

    # Check for partial scores to resume from
    partial = load_partial(partial_dir, name, cfg_hash)
    if partial is not None:
        saved_scores, resume_idx = partial
        progress_log(f"  [{series_idx+1}/{total_series}] {name} RESUMING from point {resume_idx}/{n_total}")
        # Re-create pipeline and replay train + already-scored points to restore state
        model.fit(train_signal, None)
        # Need to replay through the pipeline to restore internal state
        # The wrapper resets pipeline on predict, so we need to manually step through
        if model.pipeline is None or model.reset_per_series:
            model.pipeline = model.pipeline_factory()
        # Replay train portion + already-scored portion to restore detector state
        for i in range(min(resume_idx, n_total)):
            model.pipeline.step(float(signal[i]))
        scores = np.zeros(n_total, dtype=np.float64)
        scores[:resume_idx] = saved_scores
        start_idx = resume_idx
    else:
        progress_log(f"  [{series_idx+1}/{total_series}] {name} START n={n_total} pos={int(labels.sum())}")
        model.fit(train_signal, None)
        if model.pipeline is None or model.reset_per_series:
            model.pipeline = model.pipeline_factory()
        scores = np.zeros(n_total, dtype=np.float64)
        start_idx = 0

    # Score point-by-point with periodic checkpointing
    t_series_start = time.perf_counter()
    try:
        for i in range(start_idx, n_total):
            # Check timeout
            if series_timeout > 0 and (time.perf_counter() - t_series_start) > series_timeout:
                elapsed = time.perf_counter() - t_series_start
                progress_log(f"  [{series_idx+1}/{total_series}] {name} TIMEOUT at point {i}/{n_total} ({elapsed:.0f}s)")
                save_partial(partial_dir, name, scores, i, cfg_hash)
                return {
                    "name": name, "error": f"timeout after {elapsed:.0f}s at point {i}/{n_total}",
                    "elapsed_seconds": elapsed, "n_eval": 0, "n_positive": 0,
                    "metrics": {}, "baselines": {}, "ts": utc_now(),
                }

            a_t, _ = model.pipeline.step(float(signal[i]))
            scores[i] = float(a_t)

            # Checkpoint every CHECKPOINT_INTERVAL points
            if (i + 1) % CHECKPOINT_INTERVAL == 0:
                save_partial(partial_dir, name, scores, i + 1, cfg_hash)
                elapsed = time.perf_counter() - t_series_start
                speed = (i + 1 - start_idx) / elapsed if elapsed > 0 else 0
                progress_log(f"  [{series_idx+1}/{total_series}] {name} "
                           f"point {i+1}/{n_total} ({100*(i+1)//n_total}%) "
                           f"speed={speed:.0f} pts/s elapsed={elapsed:.0f}s")

        # Final save of complete scores
        save_partial(partial_dir, name, scores, n_total, cfg_hash)
    except Exception as e:
        elapsed = time.perf_counter() - t_series_start
        # Save what we have
        save_partial(partial_dir, name, scores, i if 'i' in dir() else start_idx, cfg_hash)
        progress_log(f"  [{series_idx+1}/{total_series}] {name} ERROR at point {i}: {e}")
        return {
            "name": name, "error": f"{type(e).__name__}: {e}",
            "elapsed_seconds": elapsed, "n_eval": 0, "n_positive": 0,
            "metrics": {}, "baselines": {}, "ts": utc_now(),
        }

    if not np.isfinite(scores).all():
        elapsed = time.perf_counter() - t_series_start
        progress_log(f"  [{series_idx+1}/{total_series}] {name} ERROR: non-finite scores")
        return {
            "name": name, "error": "non-finite scores",
            "elapsed_seconds": elapsed, "n_eval": 0, "n_positive": 0,
            "metrics": {}, "baselines": {}, "ts": utc_now(),
        }

    # Compute metrics on eval window (same protocol as BenchmarkRunner._run_series)
    eval_signal = signal[train_split:]
    eval_labels = labels[train_split:]
    warmup = max(1, int(len(eval_signal) * config.warmup_fraction))
    if warmup >= len(eval_signal):
        warmup = max(1, len(eval_signal) // 10)
    eval_scores = scores[train_split + warmup:]
    eval_labels_warm = eval_labels[warmup:]

    if len(eval_scores) == 0:
        return {
            "name": name, "error": "warmup consumed eval split",
            "elapsed_seconds": time.perf_counter() - t_series_start,
            "n_eval": 0, "n_positive": 0, "metrics": {}, "baselines": {}, "ts": utc_now(),
        }

    metrics = primary_metrics(eval_scores, eval_labels_warm, config.max_buffer)
    if config.compute_tsb_ad_metrics:
        metrics.update(tsb_ad_metrics(eval_scores, eval_labels_warm, config.max_buffer))

    pers = persistence_scores(eval_signal)[warmup:]
    baselines = {"persistence": primary_metrics(pers, eval_labels_warm, config.max_buffer)}
    if config.compute_tsb_ad_metrics:
        baselines["persistence"].update(tsb_ad_metrics(pers, eval_labels_warm, config.max_buffer))

    elapsed = time.perf_counter() - t_series_start
    result = {
        "name": name,
        "metrics": metrics,
        "baselines": baselines,
        "n_eval": len(eval_labels_warm),
        "n_positive": int(eval_labels_warm.sum()),
        "elapsed_seconds": elapsed,
        "error": None,
        "ts": utc_now(),
    }

    progress_log(f"  [{series_idx+1}/{total_series}] {name} DONE "
               f"VUS-PR={metrics.get('vus_pr', 0):.4f} "
               f"VUS-ROC={metrics.get('vus_roc', 0):.4f} "
               f"elapsed={elapsed:.1f}s")

    # Clear partial files — series is complete
    clear_partial(partial_dir, name)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Resumable benchmark runner")
    parser.add_argument("--benchmark", type=str, default="TSB-AD-U")
    parser.add_argument("--split", type=str, default="eval", choices=["eval", "train"])
    parser.add_argument("--max-series", type=int, default=None)
    parser.add_argument("--warmup-fraction", type=float, default=0.05)
    parser.add_argument("--max-buffer", type=int, default=15)
    parser.add_argument("--n-surrogates", type=int, default=0)
    parser.add_argument("--tsb-ad-metrics", action="store_true")
    parser.add_argument("--downsample", type=int, default=10000)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Resume from checkpoint if it exists (default: True)")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--series-timeout", type=float, default=600,
                        help="Max seconds per series (0=disable, default=600)")
    parser.add_argument("--checkpoint-interval", type=int, default=500,
                        help="Save partial scores every N points (default=500)")
    args = parser.parse_args()

    global CHECKPOINT_INTERVAL
    CHECKPOINT_INTERVAL = args.checkpoint_interval

    data_root = Path(args.data_root) if args.data_root else ROOT / "data" / "benchmarks"
    output_dir = Path(args.output_dir) if args.output_dir else ROOT / "results" / "benchmarks"
    run_dir = output_dir / f"{args.benchmark}_{args.split}_resumable"
    run_dir.mkdir(parents=True, exist_ok=True)
    partial_dir = run_dir / "partial"
    partial_dir.mkdir(exist_ok=True)

    checkpoint_path = run_dir / "checkpoint.jsonl"
    run_log_path = run_dir / "run_log.json"
    progress_path = run_dir / "progress.log"

    def progress_log(msg: str):
        line = f"[{utc_now()}] {msg}"
        print(line, flush=True)
        with progress_path.open("a") as f:
            f.write(line + "\n")

    config = BenchmarkConfig(
        warmup_fraction=args.warmup_fraction,
        max_buffer=args.max_buffer,
        n_surrogates=args.n_surrogates,
        compute_tsb_ad_metrics=args.tsb_ad_metrics,
        max_series=args.max_series,
        downsample_max_points=args.downsample if args.downsample > 0 else None,
    )
    cfg_hash = config_hash(config)

    # Load checkpoint
    completed = load_checkpoint(checkpoint_path) if args.resume else {}
    if completed:
        progress_log(f"RESUME: {len(completed)} series already completed, continuing...")

    # Get dataset and series list
    ds = TSB_AD_U(data_root=data_root)
    if not ds.is_downloaded:
        progress_log(f"ERROR: {args.benchmark} not downloaded at {ds.csv_dir}")
        return 1

    all_series = list(ds.iter_series(args.split))
    if args.max_series:
        all_series = all_series[:args.max_series]
    total = len(all_series)

    progress_log(f"{'='*60}")
    progress_log(f"  Benchmark: {args.benchmark} | Split: {args.split}")
    progress_log(f"  Total series: {total} | Already completed: {len(completed)}")
    progress_log(f"  Remaining: {total - len(completed)}")
    progress_log(f"  Config: warmup={config.warmup_fraction} buffer={config.max_buffer} "
               f"downsample={config.downsample_max_points} tsb_ad={config.compute_tsb_ad_metrics}")
    progress_log(f"  Checkpoint interval: every {CHECKPOINT_INTERVAL} points")
    progress_log(f"  Series timeout: {args.series_timeout}s")
    progress_log(f"{'='*60}")

    model = TSADPipelineWrapper()
    t_run_start = time.perf_counter()
    results = list(completed.values())  # start with already-completed

    for idx, series in enumerate(all_series):
        if series.name in completed:
            continue  # skip already done

        result = run_series_with_checkpoint(
            series, model, config, partial_dir, cfg_hash,
            progress_log, idx, total, args.series_timeout,
        )

        if result is not None:
            append_checkpoint(checkpoint_path, result)
            results.append(result)

            # Update running aggregate
            success = [r for r in results if r.get("error") is None]
            if success:
                mean_pr = np.mean([r["metrics"].get("vus_pr", 0) for r in success])
                mean_roc = np.mean([r["metrics"].get("vus_roc", 0) for r in success])
                n_done = len(results)
                n_ok = len(success)
                n_err = n_done - n_ok
                elapsed_total = time.perf_counter() - t_run_start
                progress_log(f"  >> Running aggregate: [{n_done}/{total}] "
                           f"ok={n_ok} err={n_err} "
                           f"mean VUS-PR={mean_pr:.4f} VUS-ROC={mean_roc:.4f} "
                           f"total elapsed={elapsed_total:.0f}s")

                # Update run_log.json
                run_log = {
                    "ts": utc_now(),
                    "total_series": total,
                    "completed": n_done,
                    "success": n_ok,
                    "errors": n_err,
                    "remaining": total - n_done,
                    "mean_vus_pr": float(mean_pr),
                    "mean_vus_roc": float(mean_roc),
                    "elapsed_seconds": elapsed_total,
                    "config_hash": cfg_hash,
                }
                run_log_path.write_text(json.dumps(run_log, indent=2) + "\n")

    # Final aggregate
    success = [r for r in results if r.get("error") is None]
    elapsed_total = time.perf_counter() - t_run_start

    progress_log(f"\n{'='*60}")
    progress_log(f"  COMPLETED: {len(results)}/{total} "
               f"({len(success)} success, {len(results)-len(success)} errors)")
    progress_log(f"  Total elapsed: {elapsed_total:.0f}s ({elapsed_total/60:.1f} min)")

    if success:
        metric_keys = set()
        for r in success:
            metric_keys.update(r["metrics"].keys())
        aggregate = {}
        for k in sorted(metric_keys):
            vals = [r["metrics"][k] for r in success if k in r["metrics"] and r["metrics"][k] is not None]
            if vals:
                aggregate[f"{k}_mean"] = float(np.mean(vals))
                aggregate[f"{k}_median"] = float(np.median(vals))
                aggregate[f"{k}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                aggregate[f"{k}_n"] = len(vals)

        # Baseline aggregates
        for bname in ["persistence"]:
            b_keys = set()
            for r in success:
                if bname in r.get("baselines", {}):
                    b_keys.update(r["baselines"][bname].keys())
            for k in sorted(b_keys):
                vals = [r["baselines"][bname][k] for r in success
                        if bname in r.get("baselines", {}) and k in r["baselines"][bname]
                        and r["baselines"][bname][k] is not None]
                if vals:
                    aggregate[f"{bname}_{k}_mean"] = float(np.mean(vals))

        progress_log(f"\n  Aggregate metrics ({len(success)} successful series):")
        for k in sorted(aggregate.keys()):
            if k.endswith("_mean"):
                progress_log(f"    {k:45s} {aggregate[k]:.4f}")

        # Save final result
        final_result = {
            "dataset_name": args.benchmark,
            "model_name": model.name,
            "split": args.split,
            "series_results": results,
            "aggregate": aggregate,
            "provenance": {
                "config": {
                    "warmup_fraction": config.warmup_fraction,
                    "max_buffer": config.max_buffer,
                    "n_surrogates": config.n_surrogates,
                    "seed": config.seed,
                    "compute_tsb_ad_metrics": config.compute_tsb_ad_metrics,
                    "downsample_max_points": config.downsample_max_points,
                },
                "config_hash": cfg_hash,
                "total_series": total,
                "n_success": len(success),
                "n_errors": len(results) - len(success),
                "elapsed_seconds": elapsed_total,
                "timestamp_utc": utc_now(),
            },
        }
        final_path = run_dir / f"final_result_{args.split}.json"
        final_path.write_text(json.dumps(final_result, indent=2, default=float) + "\n")
        progress_log(f"\n  Final result saved to {final_path}")

    # Final run log
    run_log = {
        "ts": utc_now(),
        "total_series": total,
        "completed": len(results),
        "success": len(success),
        "errors": len(results) - len(success),
        "elapsed_seconds": elapsed_total,
        "config_hash": cfg_hash,
    }
    run_log_path.write_text(json.dumps(run_log, indent=2) + "\n")
    progress_log(f"  Run log saved to {run_log_path}")
    progress_log(f"  Checkpoint at {checkpoint_path}")
    progress_log(f"{'='*60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
