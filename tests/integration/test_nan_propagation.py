import numpy as np

from tsad.pipeline import TSADPipeline


def test_nan_propagation_handling():
    pipeline = TSADPipeline()
    np.random.seed(42)
    data = np.random.normal(0, 1, size=100)
    
    scores = []
    for i, val in enumerate(data):
        if i == 50:
            # Inject NaN
            A_t, v_hat = pipeline.step(np.nan)
        else:
            A_t, v_hat = pipeline.step(val)
            
        scores.append(A_t)
        assert not np.isnan(A_t)
        assert not np.isnan(v_hat)
        assert not np.isinf(A_t)
        assert not np.isinf(v_hat)
        
    assert len(scores) == 100
