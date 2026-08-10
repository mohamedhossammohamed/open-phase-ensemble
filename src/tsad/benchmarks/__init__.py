"""Industry-standard benchmark integration for open-phase-ensemble.

This package provides a generic, extensible harness for evaluating
`TSADPipeline` and other detection methods against standard time-series
anomaly-detection benchmarks.  It is intentionally decoupled from the
`TSB-AD` Python package: loaders read the public data archives directly
and primary metrics are computed with the project's own VUS
implementation.  Optional TSB-AD metrics (Affiliation-F1, Event-F1, ...)
are used only when `TSB-AD` is installed.

Supported benchmarks
--------------------
- TSB-AD-U  (NeurIPS 2024) — univariate, 870 series, 350 eval + 48 tuning
- TSB-AD-M  (NeurIPS 2024) — multivariate, 200 series, 180 eval + 20 tuning
- TSB-UAD   (PVLDB 2022)   — univariate, 12,686 series
- NAB       (Numenta 2015) — historical, 58 series
- UCR-Anomaly Archive (2021) — historical, 250 series
- Yahoo S5  (2015)         — historical, 367 series

Scientific honesty guarantees
-----------------------------
- Train and evaluation splits are enforced by the benchmark loaders.
- Hyperparameter tuning is restricted to the training split.
- Evaluation always uses a chronological warm-up on the *test* split.
- Primary leaderboard metric is **VUS-PR** (PVLDB 2022 / NeurIPS 2024).
- Point-adjustment (PA-F1) is reported only as a secondary diagnostic.
- Each run emits a provenance manifest with checksums, split hashes and
  hyperparameters.
"""

from __future__ import annotations

from tsad.benchmarks.base import (
    BenchmarkConfig,
    BenchmarkDataset,
    BenchmarkResult,
    BenchmarkRunner,
    BenchmarkSeries,
    ModelWrapper,
    SeriesResult,
)
from tsad.benchmarks.historical import (
    NAB,
    TSB_UAD,
    UCRAnomalyArchive,
    YahooS5,
    BENCHMARK_REGISTRY,
    get_benchmark,
    list_benchmarks,
)
from tsad.benchmarks.tsb_ad import TSB_AD_M, TSB_AD_U
from tsad.benchmarks.wrappers import FunctionWrapper, TSADPipelineWrapper

__all__ = [
    "BenchmarkConfig",
    "BenchmarkDataset",
    "BenchmarkResult",
    "BenchmarkRunner",
    "BenchmarkSeries",
    "ModelWrapper",
    "SeriesResult",
    "TSADPipelineWrapper",
    "FunctionWrapper",
    "TSB_AD_U",
    "TSB_AD_M",
    "TSB_UAD",
    "NAB",
    "UCRAnomalyArchive",
    "YahooS5",
    "BENCHMARK_REGISTRY",
    "get_benchmark",
    "list_benchmarks",
]
