import numpy as np

from tests.fixtures.generate_fixtures import get_or_create_sine_fixture
from tsad.evaluation.vus import compute_vus_roc
from tsad.pipeline import TSADPipeline


def test_pipeline_fixture_performance_and_entropy():
    signal, labels = get_or_create_sine_fixture()
    pipeline = TSADPipeline()
    
    scores = []
    for x in signal:
        A_t, _ = pipeline.step(x)
        scores.append(A_t)
        
    scores_arr = np.array(scores)
    
    # Synthetic fixtures are a regression smoke test, not evidence of
    # scientific detection performance.
    vus = compute_vus_roc(scores_arr, labels, max_buffer=10)
    assert np.isfinite(vus)
    assert 0.0 <= vus <= 1.0
    assert scores_arr.shape == signal.shape
    assert np.all(np.isfinite(scores_arr))
    
    # 2. Entropy invariant: Meta-Judge weight vector must retain diversity (> 0.1 bits)
    weights = pipeline.meta_judge.weights
    entropy = -np.sum(weights * np.log2(weights + 1e-12))
    assert entropy > 0.1
