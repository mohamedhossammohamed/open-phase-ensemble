import numpy as np
import pytest

import tsad.pipeline as pipeline_module
from tsad.detectors.iforest import IsolationForestDetector
from tsad.detectors.mahalanobis import RobustMahalanobisDetector
from tsad.detectors.transformer import MSETransformerAutoencoder
from tsad.gating import CUSUMGating
from tsad.pipeline import TSADPipeline


def test_stream_buffer_keeps_standardized_history():
    pipeline = TSADPipeline(tau=1, d=2, d_target=2)

    last_v = None
    for value in [10.0, 11.0, 12.0, 13.0]:
        last_v = pipeline.step(value)[0]

    standardized = pipeline.ingestion.get_standardized_buffer()

    assert standardized.shape == pipeline.ingestion.get_buffer().shape
    assert np.isfinite(standardized).all()
    assert np.isfinite(last_v)


def test_stream_buffer_standardizes_against_past_observations_only():
    buffer = pipeline_module.StreamBuffer(window_size=10)
    for value in [0.0, 0.0, 10.0]:
        buffer.step(value)

    current = 100.0
    actual = buffer.step(current)
    previous = np.array([0.0, 0.0, 10.0])
    median = np.median(previous)
    mad = np.median(np.abs(previous - median)) + 1e-6
    expected = (current - median) / (1.4826 * mad)

    assert np.isclose(actual, expected)


def test_pipeline_embeds_standardized_history(monkeypatch):
    captured = []
    original_delay_embed = pipeline_module.delay_embed

    def capture_delay_embed(values, tau, d):
        captured.append(values.copy())
        return original_delay_embed(values, tau=tau, d=d)

    monkeypatch.setattr(pipeline_module, "delay_embed", capture_delay_embed)
    pipeline = TSADPipeline(tau=1, d=2, d_target=2)

    for value in [10.0, 11.0, 12.0, 13.0]:
        pipeline.step(value)

    assert captured
    expected = pipeline.ingestion.get_standardized_buffer()
    max_lag = (pipeline.d - 1) * pipeline.tau
    assert np.array_equal(captured[-1], expected[-(max_lag + 1):])


def test_mahalanobis_does_not_fit_on_sample_before_scoring():
    detector = RobustMahalanobisDetector(dim=2, block_size=100)
    normal = np.zeros(2, dtype=np.float64)

    for _ in range(20):
        detector.score(normal, v_t=0.0)
        detector.update(0.0)
        detector.add_vector(normal)

    mean_before = detector.mean.copy()
    detector.score(np.array([10.0, 10.0]), v_t=10.0)

    assert np.array_equal(detector.mean, mean_before)


def test_isolation_forest_does_not_add_sample_before_scoring():
    detector = IsolationForestDetector(dim=2, n_estimators=10, subsample=50)
    normal = np.zeros(2, dtype=np.float64)

    for _ in range(60):
        detector.score(normal, v_t=0.0)
        detector.update(0.0)
        detector.add_vector(normal)

    count_before = len(detector.buffer)
    detector.score(np.array([10.0, 10.0]), v_t=10.0)

    assert len(detector.buffer) == count_before


def test_transformer_updates_weights_after_observation():
    torch = pytest.importorskip("torch")
    detector = MSETransformerAutoencoder(
        dim=2,
        d_model=8,
        n_heads=2,
        n_layers=1,
        seq_len=3,
        train_interval=1,
    )
    before = [parameter.detach().clone() for parameter in detector.model.parameters()]

    for value in range(6):
        vector = np.array([float(value), float(value) / 2.0])
        detector.score(vector, v_t=float(value))
        detector.update(float(value))

    assert any(
        not torch.equal(previous, current.detach())
        for previous, current in zip(before, detector.model.parameters())
    )


def test_cusum_compares_error_with_pre_update_reference():
    gating = CUSUMGating(k_c=0.0, h_c_mult=5.0, t_drift=20)

    for _ in range(10):
        gating.step(0.0)

    gating.step(10.0)

    assert gating.last_reference_mean == 0.0
