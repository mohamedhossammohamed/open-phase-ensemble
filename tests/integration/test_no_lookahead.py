import numpy as np
import pytest
from tsad.pipeline import TSADPipeline
from tests.fixtures.generate_fixtures import get_or_create_sine_fixture

def test_no_lookahead_leakage_invariant():
    signal, labels = get_or_create_sine_fixture()
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
