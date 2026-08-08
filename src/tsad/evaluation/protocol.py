"""Causal, reproducible evaluation protocol for streaming benchmarks."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from tsad.config import SEED
from tsad.evaluation.iaaft import generate_iaaft_surrogate
from tsad.evaluation.vus import compute_vus_pr, compute_vus_roc
from tsad.pipeline import TSADPipeline


def persistence_scores(signal: np.ndarray) -> np.ndarray:
    """Causal persistence baseline using the absolute first difference."""
    signal = np.asarray(signal, dtype=np.float64)
    scores = np.zeros(len(signal), dtype=np.float64)
    if len(signal) > 1:
        scores[1:] = np.abs(np.diff(signal))
    return scores


def _run_pipeline(
    signal: np.ndarray,
    pipeline_factory: Callable[[], Any],
) -> tuple[np.ndarray, np.ndarray]:
    pipeline = pipeline_factory()
    fused_scores = np.zeros(len(signal), dtype=np.float64)
    detector_scores = []
    for index, value in enumerate(signal):
        fused_scores[index], _ = pipeline.step(float(value))
        detector_scores.append(np.asarray(pipeline.last_scores, dtype=np.float64).copy())
    return fused_scores, np.asarray(detector_scores, dtype=np.float64)


def _metric_pair(scores: np.ndarray, labels: np.ndarray, max_buffer: int) -> dict[str, float]:
    return {
        "vus_roc": compute_vus_roc(scores, labels, max_buffer=max_buffer),
        "vus_pr": compute_vus_pr(scores, labels, max_buffer=max_buffer),
    }


def evaluate_stream(
    signal: np.ndarray,
    labels: np.ndarray,
    *,
    warmup_fraction: float = 0.2,
    max_buffer: int = 15,
    max_points: int | None = 5000,
    n_surrogates: int = 20,
    seed: int = SEED,
    pipeline_factory: Callable[[], Any] = TSADPipeline,
) -> dict[str, Any]:
    """Evaluate a stream after chronological warm-up against explicit baselines."""
    if not 0.0 <= warmup_fraction < 1.0:
        raise ValueError("warmup_fraction must be in [0, 1)")
    if n_surrogates < 1:
        raise ValueError("n_surrogates must be positive")

    signal = np.asarray(signal, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int8)
    if signal.ndim != 1 or labels.ndim != 1 or len(signal) != len(labels):
        raise ValueError("signal and labels must be one-dimensional arrays of equal length")

    if max_points is not None and len(signal) > max_points:
        stride = int(np.ceil(len(signal) / max_points))
        signal = signal[::stride]
        labels = labels[::stride]

    warmup_points = int(len(signal) * warmup_fraction)
    evaluation_slice = slice(warmup_points, None)
    evaluation_labels = labels[evaluation_slice]

    system_scores, detector_scores = _run_pipeline(signal, pipeline_factory)
    system_metrics = _metric_pair(system_scores[evaluation_slice], evaluation_labels, max_buffer)
    persistence_metrics = _metric_pair(
        persistence_scores(signal)[evaluation_slice], evaluation_labels, max_buffer
    )

    detector_metrics = {}
    if detector_scores.ndim == 2:
        for index in range(detector_scores.shape[1]):
            detector_metrics[f"detector_{index}"] = compute_vus_roc(
                detector_scores[evaluation_slice, index],
                evaluation_labels,
                max_buffer=max_buffer,
            )

    surrogate_metrics = []
    for surrogate_index in range(n_surrogates):
        surrogate = generate_iaaft_surrogate(signal, seed=seed + surrogate_index)
        surrogate_scores, _ = _run_pipeline(surrogate, pipeline_factory)
        surrogate_metrics.append(
            compute_vus_roc(
                surrogate_scores[evaluation_slice],
                evaluation_labels,
                max_buffer=max_buffer,
            )
        )

    surrogate_array = np.asarray(surrogate_metrics, dtype=np.float64)
    p_value = (1.0 + float(np.sum(surrogate_array >= system_metrics["vus_roc"]))) / (
        n_surrogates + 1.0
    )
    return {
        "n_total": len(signal),
        "n_evaluated": len(signal) - warmup_points,
        "warmup_points": warmup_points,
        "n_surrogates": n_surrogates,
        "system_vus_roc": system_metrics["vus_roc"],
        "system_vus_pr": system_metrics["vus_pr"],
        "persistence_vus_roc": persistence_metrics["vus_roc"],
        "persistence_vus_pr": persistence_metrics["vus_pr"],
        "detector_vus_roc": detector_metrics,
        "surrogate_vus_roc": surrogate_metrics,
        "surrogate_vus_roc_mean": float(np.mean(surrogate_array)),
        "surrogate_vus_roc_std": float(np.std(surrogate_array, ddof=1))
        if n_surrogates > 1
        else 0.0,
        "predictive_edge": system_metrics["vus_roc"] - float(np.mean(surrogate_array)),
        "surrogate_p_value": p_value,
    }
