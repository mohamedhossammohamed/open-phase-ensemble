import numpy as np

from tsad.evaluation.protocol import (
    _causal_block_downsample,
    evaluate_stream,
    persistence_scores,
)


class FakePipeline:
    def __init__(self):
        self.last_scores = np.zeros(2, dtype=np.float64)
        self._previous = 0.0

    def step(self, value):
        score = abs(float(value) - self._previous)
        self._previous = float(value)
        self.last_scores = np.array([score, score / 2.0])
        return score, float(value)


def test_persistence_scores_are_causal_first_differences():
    scores = persistence_scores(np.array([1.0, 3.0, 2.0]))

    assert np.array_equal(scores, np.array([0.0, 2.0, 1.0]))


def test_causal_block_downsampling_preserves_sparse_labels():
    signal = np.arange(10, dtype=np.float64)
    labels = np.zeros(10, dtype=np.int8)
    labels[3] = 1

    sampled_signal, sampled_labels, stride = _causal_block_downsample(signal, labels, 4)

    assert stride == 3
    assert np.array_equal(sampled_signal, np.array([2.0, 5.0, 8.0, 9.0]))
    assert np.array_equal(sampled_labels, np.array([0, 1, 0, 0], dtype=np.int8))


def test_evaluation_protocol_reports_warmup_baseline_and_surrogate_count():
    signal = np.sin(np.linspace(0.0, 20.0, 80))
    labels = np.zeros(len(signal), dtype=np.int8)
    labels[50:55] = 1

    result = evaluate_stream(
        signal,
        labels,
        warmup_fraction=0.25,
        n_surrogates=2,
        max_points=None,
        seed=7,
        pipeline_factory=FakePipeline,
    )

    assert result["warmup_points"] == 20
    assert result["n_evaluated"] == 60
    assert result["n_surrogates"] == 2
    assert "persistence_vus_roc" in result
    assert len(result["surrogate_vus_roc"]) == 2
    assert set(result["detector_vus_roc"]) == {"detector_0", "detector_1"}
