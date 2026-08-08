import hashlib
import numpy as np
import pytest
from tsad.pipeline import TSADPipeline
from tests.fixtures.generate_fixtures import get_or_create_sine_fixture

def run_pipeline_and_hash():
    signal, labels = get_or_create_sine_fixture()
    pipeline = TSADPipeline()
    scores = []
    for x in signal[:1000]:
        A_t, _ = pipeline.step(x)
        scores.append(A_t)
        
    scores_bytes = np.array(scores, dtype=np.float64).tobytes()
    return hashlib.sha256(scores_bytes).hexdigest()

def test_100_percent_execution_determinism():
    hash1 = run_pipeline_and_hash()
    hash2 = run_pipeline_and_hash()
    hash3 = run_pipeline_and_hash()
    
    assert hash1 == hash2 == hash3
