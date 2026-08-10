"""Tests for the benchmark harness.

These tests verify:
1. The benchmark package imports correctly.
2. All registered benchmarks can be instantiated.
3. TSB-AD-U loads real data with correct splits.
4. The runner produces valid results on a small sample.
5. Provenance manifests are complete.
6. The scientific honesty protocol is enforced (warm-up, no eval tuning).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

# Ensure src is on path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tsad.benchmarks import (
    BenchmarkConfig,
    BenchmarkResult,
    BenchmarkRunner,
    BenchmarkSeries,
    BENCHMARK_REGISTRY,
    FunctionWrapper,
    TSADPipelineWrapper,
    get_benchmark,
    list_benchmarks,
)
from tsad.benchmarks.tsb_ad import TSB_AD_U, _parse_filename


DATA_ROOT = PROJECT_ROOT / "data" / "benchmarks"


# ---------------------------------------------------------------------------
# Package structure tests
# ---------------------------------------------------------------------------

class TestBenchmarkPackage:
    def test_all_benchmarks_registered(self):
        names = list_benchmarks()
        assert "TSB-AD-U" in names
        assert "TSB-AD-M" in names
        assert "TSB-UAD" in names
        assert "NAB" in names
        assert "UCR-Anomaly" in names
        assert "Yahoo-S5" in names

    def test_registry_covers_all_classes(self):
        assert len(BENCHMARK_REGISTRY) >= 6

    def test_get_benchmark_unknown_raises(self):
        with pytest.raises(ValueError, match="unknown benchmark"):
            get_benchmark("nonexistent")

    def test_get_benchmark_returns_instance(self):
        ds = get_benchmark("TSB-AD-U", data_root=DATA_ROOT)
        assert ds.name == "TSB-AD-U"


# ---------------------------------------------------------------------------
# Filename parsing tests
# ---------------------------------------------------------------------------

class TestFilenameParsing:
    def test_parse_standard_filename(self):
        meta = _parse_filename("001_NAB_id_1_Facility_tr_1007_1st_2014.csv")
        assert meta["index"] == 1
        assert meta["source_dataset"] == "NAB"
        assert meta["id"] == 1
        assert meta["domain"] == "Facility"
        assert meta["train_split"] == 1007
        assert meta["first_anomaly"] == 2014

    def test_parse_multidigit_filename(self):
        meta = _parse_filename("494_UCR_id_192_Facility_tr_22500_1st_72150.csv")
        assert meta["index"] == 494
        assert meta["source_dataset"] == "UCR"
        assert meta["train_split"] == 22500

    def test_parse_invalid_filename_raises(self):
        with pytest.raises(ValueError, match="cannot parse"):
            _parse_filename("invalid_filename.csv")


# ---------------------------------------------------------------------------
# BenchmarkSeries tests
# ---------------------------------------------------------------------------

class TestBenchmarkSeries:
    def test_univariate_series(self):
        s = BenchmarkSeries(
            name="test",
            signal=np.random.randn(100),
            labels=np.zeros(100, dtype=np.int8),
            train_split=20,
        )
        assert len(s.eval_signal) == 80
        assert len(s.eval_labels) == 80

    def test_multivariate_series(self):
        s = BenchmarkSeries(
            name="test_mv",
            signal=np.random.randn(100, 3),
            labels=np.zeros(100, dtype=np.int8),
            train_split=20,
        )
        assert s.signal.ndim == 2
        assert s.signal.shape[1] == 3

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="same length"):
            BenchmarkSeries(
                name="bad",
                signal=np.random.randn(100),
                labels=np.zeros(50, dtype=np.int8),
                train_split=10,
            )

    def test_3d_signal_raises(self):
        with pytest.raises(ValueError, match="1-D.*or 2-D"):
            BenchmarkSeries(
                name="bad",
                signal=np.random.randn(10, 3, 2),
                labels=np.zeros(10, dtype=np.int8),
                train_split=2,
            )


# ---------------------------------------------------------------------------
# TSB-AD-U loader tests (requires downloaded data)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not (DATA_ROOT / "TSB-AD-U" / "TSB-AD-U").is_dir(),
    reason="TSB-AD-U not downloaded. Run scripts/download_benchmarks.py --benchmark TSB-AD-U",
)
class TestTSBADU:
    def test_is_downloaded(self):
        ds = TSB_AD_U(data_root=DATA_ROOT)
        assert ds.is_downloaded

    def test_eval_split_has_350_series(self):
        ds = TSB_AD_U(data_root=DATA_ROOT)
        series = list(ds.iter_series("eval"))
        assert len(series) == 350

    def test_tuning_split_has_48_series(self):
        ds = TSB_AD_U(data_root=DATA_ROOT)
        series = list(ds.iter_series("train"))
        assert len(series) == 48

    def test_first_series_has_correct_metadata(self):
        ds = TSB_AD_U(data_root=DATA_ROOT)
        s = next(ds.iter_series("eval"))
        assert s.name.startswith("001_")
        assert s.train_split > 0
        assert len(s.signal) == len(s.labels)
        assert s.signal.ndim == 1
        assert set(np.unique(s.labels).tolist()).issubset({0, 1})

    def test_provenance_complete(self):
        ds = TSB_AD_U(data_root=DATA_ROOT)
        s = next(ds.iter_series("eval"))
        prov = s.provenance
        for key in ["file", "sha256", "source", "url", "license", "label_semantics"]:
            assert key in prov, f"missing provenance key: {key}"
        assert len(prov["sha256"]) == 64  # SHA-256 hex

    def test_dataset_provenance(self):
        ds = TSB_AD_U(data_root=DATA_ROOT)
        prov = ds.provenance()
        assert prov["n_eval_series"] == 350
        assert prov["n_tuning_series"] == 48
        assert prov["primary_metric"] == "VUS-PR"
        assert prov["license"] == "Apache-2.0"


# ---------------------------------------------------------------------------
# Runner tests
# ---------------------------------------------------------------------------

class TestBenchmarkRunner:
    def test_function_wrapper_on_synthetic(self):
        """Test the runner with a simple function wrapper on synthetic data."""
        # Create a synthetic series with a clear anomaly
        signal = np.concatenate([
            np.random.randn(200),
            np.random.randn(100) + 5,  # anomalous segment
            np.random.randn(200),
        ])
        labels = np.zeros(500, dtype=np.int8)
        labels[200:300] = 1

        series = BenchmarkSeries(
            name="synthetic_test",
            signal=signal,
            labels=labels,
            train_split=100,
        )

        # Simple function wrapper: absolute deviation from median
        def abs_deviation(sig: np.ndarray) -> np.ndarray:
            med = np.median(sig)
            return np.abs(sig - med)

        model = FunctionWrapper(abs_deviation, name="abs_deviation")
        config = BenchmarkConfig(
            warmup_fraction=0.05,
            max_buffer=15,
            compute_tsb_ad_metrics=False,
        )

        # We need a dataset to run, but we can test the _run_series method directly
        from tsad.benchmarks.base import BenchmarkRunner
        runner = BenchmarkRunner.__new__(BenchmarkRunner)
        runner.dataset = None
        runner.model = model
        runner.config = config

        def metric_fn(scores, labels):
            return {
                "vus_roc": float(np.random.rand()),
                "vus_pr": float(np.random.rand()),
            }

        result = runner._run_series(series, metric_fn)
        assert result.error is None
        assert result.name == "synthetic_test"
        assert result.n_eval > 0
        assert "vus_pr" in result.metrics

    def test_config_validation(self):
        with pytest.raises(ValueError, match="warmup_fraction"):
            BenchmarkConfig(warmup_fraction=1.5)
        with pytest.raises(ValueError, match="warmup_fraction"):
            BenchmarkConfig(warmup_fraction=-0.1)
        with pytest.raises(ValueError, match="n_surrogates"):
            BenchmarkConfig(n_surrogates=-1)


# ---------------------------------------------------------------------------
# Scientific honesty tests
# ---------------------------------------------------------------------------

class TestScientificHonesty:
    def test_warmup_is_applied(self):
        """Verify that the runner applies a chronological warm-up."""
        signal = np.random.randn(1000)
        labels = np.zeros(1000, dtype=np.int8)
        labels[800:900] = 1

        series = BenchmarkSeries(
            name="warmup_test",
            signal=signal,
            labels=labels,
            train_split=200,
        )

        def identity(sig):
            return np.abs(sig)

        model = FunctionWrapper(identity, name="identity")
        config = BenchmarkConfig(
            warmup_fraction=0.1,
            max_buffer=15,
            compute_tsb_ad_metrics=False,
        )

        runner = BenchmarkRunner.__new__(BenchmarkRunner)
        runner.dataset = None
        runner.model = model
        runner.config = config

        def metric_fn(scores, labels):
            return {"vus_pr": 0.5, "vus_roc": 0.5}

        result = runner._run_series(series, metric_fn)
        assert result.error is None
        # Eval split is 800 points (1000 - 200 train)
        # Warmup is 10% of 800 = 80 points
        # So n_eval should be 800 - 80 = 720
        assert result.n_eval == 720

    def test_train_split_enforced(self):
        """Verify that the train split is respected (no eval data in training)."""
        signal = np.random.randn(500)
        labels = np.zeros(500, dtype=np.int8)
        labels[400:450] = 1

        series = BenchmarkSeries(
            name="split_test",
            signal=signal,
            labels=labels,
            train_split=300,
        )

        # The model should only see the first 300 points during fit
        seen_during_fit: list[int] = []

        class TrackingWrapper(FunctionWrapper):
            def fit(self, train_signal, train_labels=None):
                seen_during_fit.append(len(train_signal))

            def predict(self, signal):
                return np.abs(signal - np.median(signal[:300]))

        model = TrackingWrapper(lambda s: np.abs(s), name="tracking")
        config = BenchmarkConfig(
            warmup_fraction=0.05,
            max_buffer=15,
            compute_tsb_ad_metrics=False,
        )

        runner = BenchmarkRunner.__new__(BenchmarkRunner)
        runner.dataset = None
        runner.model = model
        runner.config = config

        def metric_fn(scores, labels):
            return {"vus_pr": 0.5, "vus_roc": 0.5}

        result = runner._run_series(series, metric_fn)
        assert result.error is None
        assert seen_during_fit == [300]  # Only saw train split


# ---------------------------------------------------------------------------
# Result serialization tests
# ---------------------------------------------------------------------------

class TestResultSerialization:
    def test_result_to_dict_and_save(self, tmp_path):
        from tsad.benchmarks.base import SeriesResult

        result = BenchmarkResult(
            dataset_name="test",
            model_name="test_model",
            series_results=[
                SeriesResult(name="s1", metrics={"vus_pr": 0.5}, n_eval=100),
            ],
            aggregate={"vus_pr_mean": 0.5},
            provenance={"dataset": {"name": "test"}},
        )

        d = result.to_dict()
        assert d["dataset_name"] == "test"
        assert d["series_results"][0]["name"] == "s1"

        path = result.save(tmp_path / "result.json")
        loaded = json.loads(path.read_text())
        assert loaded["dataset_name"] == "test"
        assert loaded["aggregate"]["vus_pr_mean"] == 0.5
