"""Model wrappers for benchmark evaluation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from tsad.benchmarks.base import ModelWrapper
from tsad.pipeline import TSADPipeline


class TSADPipelineWrapper(ModelWrapper):
    """Wrap ``TSADPipeline`` as a streaming anomaly detector for benchmarks.

    The wrapper streams the signal through ``pipeline.step(x)`` and returns
    the fused anomaly score ``A_t`` for every point.  It can optionally call
    ``reset`` for every series (default) or keep state across series.
    """

    name = "open-phase-ensemble"

    def __init__(
        self,
        pipeline_factory: Callable[[], TSADPipeline] | None = None,
        reset_per_series: bool = True,
        **kwargs: Any,
    ) -> None:
        self.pipeline_factory = pipeline_factory or (lambda: TSADPipeline(**kwargs))
        self.reset_per_series = reset_per_series
        self.pipeline: TSADPipeline | None = None

    def fit(self, train_signal: np.ndarray, train_labels: np.ndarray | None = None) -> None:
        """No-op fit: the pipeline is online and state resets during predict."""

    def predict(self, signal: np.ndarray) -> np.ndarray:
        if self.pipeline is None or self.reset_per_series:
            self.pipeline = self.pipeline_factory()

        signal = np.asarray(signal, dtype=np.float64)
        scores = np.zeros(len(signal), dtype=np.float64)
        for i, x in enumerate(signal):
            a_t, _ = self.pipeline.step(float(x))
            scores[i] = float(a_t)
        return scores

    def __repr__(self) -> str:
        return "TSADPipelineWrapper(name='open-phase-ensemble')"


class FunctionWrapper(ModelWrapper):
    """Wrap an arbitrary scoring function for benchmarking."""

    def __init__(self, fn: Callable[[np.ndarray], np.ndarray], name: str = "function") -> None:
        self.fn = fn
        self.name = name
        self._fitted = False

    def fit(self, train_signal: np.ndarray, train_labels: np.ndarray | None = None) -> None:
        self._fitted = True

    def predict(self, signal: np.ndarray) -> np.ndarray:
        if not self._fitted:
            # Allow stateless functions that don't need a fit step.
            pass
        return self.fn(signal)
