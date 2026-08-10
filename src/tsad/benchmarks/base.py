"""Base classes and runner for benchmark evaluation."""

from __future__ import annotations

import hashlib
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

from tsad.config import SEED
from tsad.evaluation.iaaft import generate_iaaft_surrogate
from tsad.evaluation.vus import compute_vus_pr, compute_vus_roc


def _sha256_bytes(data: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(data)
    return digest.hexdigest()


def _causal_block_downsample(
    signal: np.ndarray,
    labels: np.ndarray,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Downsample by causal blocks while retaining any label in each block.

    This preserves the streaming semantics: each block is represented by its
    last observed value, and a block is positive if any source annotation
    falls inside it.  This avoids silently discarding sparse anomalies.
    """
    if max_points < 1:
        raise ValueError("max_points must be positive")
    if len(signal) <= max_points:
        return signal, labels, 1

    stride = int(np.ceil(len(signal) / max_points))
    starts = np.arange(0, len(signal), stride, dtype=np.int64)
    ends = np.minimum(starts + stride, len(signal))
    representative_indices = ends - 1
    block_labels = np.maximum.reduceat(labels, starts).astype(np.int8, copy=False)
    return signal[representative_indices], block_labels, stride


def _series_hash(signal: np.ndarray, labels: np.ndarray) -> str:
    """Stable hash of a (signal, labels) pair for provenance."""
    return hashlib.sha256(signal.tobytes() + labels.tobytes()).hexdigest()


@dataclass(frozen=True)
class BenchmarkSeries:
    """A single time series with train/eval split and provenance."""
    name: str
    signal: np.ndarray
    labels: np.ndarray
    train_split: int
    metadata: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.signal.ndim not in (1, 2):
            raise ValueError("signal must be 1-D (univariate) or 2-D (multivariate)")
        if self.labels.ndim != 1:
            raise ValueError("labels must be one-dimensional")
        if len(self.signal) != len(self.labels):
            raise ValueError("signal and labels must have the same length")
        if not (0 <= self.train_split < len(self.signal)):
            # Allow train_split == 0 (no train split) or train_split == len(signal)-1
            if not (self.train_split == 0 or self.train_split == len(self.signal)):
                raise ValueError(f"train_split={self.train_split} out of range [0, {len(self.signal)}]")

    @property
    def eval_slice(self) -> slice:
        """Return the evaluation slice (everything after train_split)."""
        return slice(self.train_split, None)

    @property
    def eval_signal(self) -> np.ndarray:
        return self.signal[self.eval_slice]

    @property
    def eval_labels(self) -> np.ndarray:
        return self.labels[self.eval_slice]

    @property
    def eval_hash(self) -> str:
        return _series_hash(self.eval_signal, self.eval_labels)


class BenchmarkDataset(ABC):
    """Abstract base for an industry-standard benchmark dataset."""

    name: str

    def __init__(self, data_root: str | Path | None = None) -> None:
        if data_root is None:
            project_root = Path(__file__).resolve().parents[3]
            data_root = project_root / "data" / "benchmarks"
        self.data_root = Path(data_root)
        self.dataset_dir = self.data_root / self.name

    @property
    @abstractmethod
    def is_downloaded(self) -> bool:
        """Whether the raw data is present locally."""

    @abstractmethod
    def download(self, *, force: bool = False) -> None:
        """Download and extract the benchmark archive."""

    @abstractmethod
    def iter_series(self, split: str) -> Iterator[BenchmarkSeries]:
        """Iterate over series for a given split ('train' or 'eval').

        For benchmarks with explicit train/eval splits, 'train' yields the
        training portion of every series; 'eval' yields the full series
        (train split index is set so eval can be sliced).  For benchmarks
        without a train split, 'train' is empty and 'eval' contains all
        series.
        """

    @abstractmethod
    def provenance(self) -> dict[str, Any]:
        """Return a provenance manifest for this dataset."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, root={self.dataset_dir})"


class ModelWrapper(ABC):
    """Abstract wrapper that adapts any model to the benchmark runner."""

    name: str = "unnamed_model"

    @abstractmethod
    def fit(self, train_signal: np.ndarray, train_labels: np.ndarray | None = None) -> None:
        """Optionally train/tune on the training split.

        Implementations must not access labels beyond the train split.
        """

    @abstractmethod
    def predict(self, signal: np.ndarray) -> np.ndarray:
        """Return an anomaly score for every point in ``signal``."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


@dataclass(frozen=True)
class SeriesResult:
    """Per-series benchmark result."""
    name: str
    metrics: dict[str, float] = field(default_factory=dict)
    baselines: dict[str, dict[str, float]] = field(default_factory=dict)
    n_eval: int = 0
    n_positive: int = 0
    elapsed_seconds: float = 0.0
    error: str | None = None


@dataclass(frozen=True)
class BenchmarkResult:
    """Aggregate benchmark result with per-series details."""
    dataset_name: str
    model_name: str
    series_results: list[SeriesResult] = field(default_factory=list)
    aggregate: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "model_name": self.model_name,
            "series_results": [
                {
                    "name": r.name,
                    "metrics": r.metrics,
                    "baselines": r.baselines,
                    "n_eval": r.n_eval,
                    "n_positive": r.n_positive,
                    "elapsed_seconds": r.elapsed_seconds,
                    "error": r.error,
                }
                for r in self.series_results
            ],
            "aggregate": self.aggregate,
            "provenance": self.provenance,
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=float) + "\n", encoding="utf-8")
        return path


@dataclass(frozen=True)
class BenchmarkConfig:
    """Reproducible configuration for a benchmark run."""
    warmup_fraction: float = 0.05
    max_buffer: int = 15
    n_surrogates: int = 0
    seed: int = SEED
    compute_tsb_ad_metrics: bool = True
    compute_detectors: bool = False
    downsample_max_points: int | None = None
    max_series: int | None = None  # For quick smoke tests
    n_jobs: int = 1  # Parallelism; default 1 for deterministic streaming

    def __post_init__(self):
        if not 0.0 <= self.warmup_fraction < 1.0:
            raise ValueError("warmup_fraction must be in [0, 1)")
        if self.n_surrogates < 0:
            raise ValueError("n_surrogates must be non-negative")


class BenchmarkRunner:
    """Runs a model on a benchmark dataset with scientifically honest protocol."""

    def __init__(
        self,
        dataset: BenchmarkDataset,
        model: ModelWrapper,
        config: BenchmarkConfig | None = None,
    ) -> None:
        self.dataset = dataset
        self.model = model
        self.config = config or BenchmarkConfig()

    def _run_series(
        self,
        series: BenchmarkSeries,
        metric_fn: Callable[[np.ndarray, np.ndarray], dict[str, float]],
    ) -> SeriesResult:
        start = time.perf_counter()
        try:
            # Apply downsampling for very long series to keep runtime tractable.
            signal = series.signal
            labels = series.labels
            train_split = series.train_split
            downsample_stride = 1

            if self.config.downsample_max_points is not None and len(signal) > self.config.downsample_max_points:
                signal, labels, downsample_stride = _causal_block_downsample(
                    signal, labels, self.config.downsample_max_points
                )
                train_split = max(1, int(train_split / downsample_stride))

            train_signal = signal[:train_split]
            self.model.fit(train_signal, None)

            # Score the full series; evaluation is sliced after warmup.
            scores = self.model.predict(signal)
            if not np.isfinite(scores).all():
                raise ValueError("model returned non-finite anomaly scores")

            eval_signal = signal[train_split:]
            eval_labels = labels[train_split:]
            if len(eval_signal) == 0:
                raise ValueError("empty evaluation split")

            warmup_points = max(1, int(len(eval_signal) * self.config.warmup_fraction))
            if warmup_points >= len(eval_signal):
                warmup_points = max(1, len(eval_signal) // 10)
            eval_scores = scores[train_split + warmup_points:]
            eval_labels_warm = eval_labels[warmup_points:]

            if len(eval_scores) == 0:
                raise ValueError("warmup consumed the entire evaluation split")

            metrics = metric_fn(eval_scores, eval_labels_warm)

            # Baselines computed on the same eval window.
            baselines: dict[str, dict[str, float]] = {}
            persistence = self._persistence_scores(eval_signal)[warmup_points:]
            baselines["persistence"] = metric_fn(persistence, eval_labels_warm)

            if self.config.n_surrogates > 0:
                surrogate_metrics = []
                for i in range(self.config.n_surrogates):
                    surrogate = generate_iaaft_surrogate(eval_signal, seed=self.config.seed + i)
                    surrogate_scores = self.model.predict(
                        np.concatenate([signal[:train_split], surrogate])
                    )[train_split:]
                    surrogate_scores = surrogate_scores[warmup_points:]
                    surrogate_metrics.append(metric_fn(surrogate_scores, eval_labels_warm))
                baselines["surrogate"] = self._aggregate_surrogates(surrogate_metrics)

            elapsed = time.perf_counter() - start
            return SeriesResult(
                name=series.name,
                metrics=metrics,
                baselines=baselines,
                n_eval=len(eval_labels_warm),
                n_positive=int(np.sum(eval_labels_warm)),
                elapsed_seconds=elapsed,
            )
        except Exception as exc:
            elapsed = time.perf_counter() - start
            return SeriesResult(
                name=series.name,
                error=f"{type(exc).__name__}: {exc}",
                elapsed_seconds=elapsed,
            )

    @staticmethod
    def _persistence_scores(signal: np.ndarray) -> np.ndarray:
        s = np.asarray(signal, dtype=np.float64)
        out = np.zeros(len(s), dtype=np.float64)
        if len(s) > 1:
            out[1:] = np.abs(np.diff(s))
        return out

    @staticmethod
    def _aggregate_surrogates(surrogate_metrics: list[dict[str, float]]) -> dict[str, float]:
        out: dict[str, float] = {}
        keys = list(surrogate_metrics[0].keys())
        for k in keys:
            values = [m[k] for m in surrogate_metrics]
            arr = np.asarray(values, dtype=np.float64)
            out[f"{k}_mean"] = float(np.mean(arr))
            out[f"{k}_std"] = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
            out[f"{k}_min"] = float(np.min(arr))
            out[f"{k}_max"] = float(np.max(arr))
        return out

    def _primary_metrics(self, scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
        """Project-native metrics (always available, independent of TSB-AD)."""
        return {
            "vus_roc": compute_vus_roc(scores, labels, max_buffer=self.config.max_buffer),
            "vus_pr": compute_vus_pr(scores, labels, max_buffer=self.config.max_buffer),
        }

    def _full_metrics(self, scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
        """Full metric set, including TSB-AD metrics when installed."""
        metrics = self._primary_metrics(scores, labels)

        if self.config.compute_tsb_ad_metrics:
            try:
                from TSB_AD.evaluation.metrics import get_metrics
                tsb = get_metrics(
                    scores,
                    labels.astype(int),
                    slidingWindow=self.config.max_buffer,
                    pred=None,
                    version="opt",
                    thre=250,
                )
                for k, v in tsb.items():
                    metrics[f"tsb_ad_{k.lower().replace(' ', '_')}"] = float(v)
            except Exception:
                # TSB-AD may not be installed or may fail on edge cases.
                pass

        return metrics

    def run(self) -> BenchmarkResult:
        """Evaluate the model on every evaluation series."""
        if not self.dataset.is_downloaded:
            raise FileNotFoundError(
                f"Dataset {self.dataset.name} not found at {self.dataset.dataset_dir}. "
                f"Run download script first."
            )

        metric_fn = self._full_metrics
        series_iter = self.dataset.iter_series("eval")
        if self.config.max_series:
            series_iter = (s for i, s in enumerate(series_iter) if i < self.config.max_series)

        series_results = [self._run_series(s, metric_fn) for s in series_iter]

        # Aggregate over successful runs.
        success_results = [r for r in series_results if r.error is None]
        if not success_results:
            raise RuntimeError("All series failed; no aggregate can be computed")

        # Collect the union of all metric keys across successful results,
        # since some series may be missing certain metrics (e.g. AUC-PR
        # when only one class is present in the labels).
        metric_keys: set[str] = set()
        for r in success_results:
            metric_keys.update(r.metrics.keys())
        aggregate: dict[str, Any] = {}
        for k in sorted(metric_keys):
            values = [r.metrics[k] for r in success_results if k in r.metrics and r.metrics[k] is not None]
            if not values:
                continue
            arr = np.asarray(values, dtype=np.float64)
            aggregate[f"{k}_mean"] = float(np.mean(arr))
            aggregate[f"{k}_median"] = float(np.median(arr))
            aggregate[f"{k}_std"] = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
            aggregate[f"{k}_n"] = len(values)

        # Baseline aggregates.
        for baseline_name in ["persistence", "surrogate"]:
            b_keys: set[str] = set()
            for r in success_results:
                if baseline_name in r.baselines:
                    b_keys.update(r.baselines[baseline_name].keys())
            for k in sorted(b_keys):
                values = [
                    r.baselines[baseline_name][k]
                    for r in success_results
                    if baseline_name in r.baselines
                    and k in r.baselines[baseline_name]
                    and r.baselines[baseline_name][k] is not None
                ]
                if not values:
                    continue
                arr = np.asarray(values, dtype=np.float64)
                aggregate[f"{baseline_name}_{k}_mean"] = float(np.mean(arr))

        provenance = {
            "dataset": self.dataset.provenance(),
            "model": self.model.name,
            "config": {
                "warmup_fraction": self.config.warmup_fraction,
                "max_buffer": self.config.max_buffer,
                "n_surrogates": self.config.n_surrogates,
                "seed": self.config.seed,
                "compute_tsb_ad_metrics": self.config.compute_tsb_ad_metrics,
            },
            "n_series_total": len(series_results),
            "n_series_success": len(success_results),
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        return BenchmarkResult(
            dataset_name=self.dataset.name,
            model_name=self.model.name,
            series_results=series_results,
            aggregate=aggregate,
            provenance=provenance,
        )
