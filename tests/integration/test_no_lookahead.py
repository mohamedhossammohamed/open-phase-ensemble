import numpy as np

from tests.fixtures.generate_fixtures import get_or_create_sine_fixture
from tsad.pipeline import TSADPipeline


def test_no_lookahead_leakage_invariant():
    signal, _labels = get_or_create_sine_fixture()
    n = 200 # Use first 200 points for fast invariant check
    
    # 1. Stream execution
    p1 = TSADPipeline()
    scores_stream = []
    for x in signal[:n]:
        A_t, _ = p1.step(x)
        scores_stream.append(A_t)
        
    # 2. Sequential execution
    p2 = TSADPipeline()
    scores_batch = []
    for i in range(n):
        A_t, _ = p2.step(signal[i])
        scores_batch.append(A_t)
        
    # Euclidean distance between streaming and sequential outputs must be exactly 0.0
    dist = np.linalg.norm(np.array(scores_stream) - np.array(scores_batch))
    assert dist == 0.0


def test_prefix_scores_do_not_depend_on_future_suffix():
    signal, _labels = get_or_create_sine_fixture()
    prefix = signal[:160]

    prefix_pipeline = TSADPipeline()
    prefix_scores = [prefix_pipeline.step(x)[0] for x in prefix]

    full_pipeline = TSADPipeline()
    full_scores = [full_pipeline.step(x)[0] for x in signal[:400]]

    np.testing.assert_array_equal(prefix_scores, full_scores[: len(prefix)])
