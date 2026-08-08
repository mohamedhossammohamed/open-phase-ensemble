import numpy as np
from scipy import stats
from tsad.evaluation.vus import compute_vus_roc, compute_vus_pr
from tsad.evaluation.iaaft import generate_iaaft_surrogate

def test_vus_roc_perfect_and_random():
    labels = np.zeros(1000, dtype=int)
    labels[200:250] = 1
    labels[700:720] = 1
    
    # Perfect predictor
    perfect_scores = labels.astype(float)
    vus_perfect = compute_vus_roc(perfect_scores, labels, max_buffer=10)
    assert vus_perfect == 1.0
    
    # Random predictor
    np.random.seed(42)
    random_scores = np.random.rand(1000)
    vus_random = compute_vus_roc(random_scores, labels, max_buffer=10)
    assert 0.4 < vus_random < 0.6

def test_iaaft_properties():
    np.random.seed(42)
    t = np.linspace(0, 10, 1000)
    x = np.sin(2 * np.pi * t) + np.random.normal(0, 0.1, size=1000)
    
    surrogate = generate_iaaft_surrogate(x)
    
    # 1. Amplitude distribution match: sorted values should be identical
    assert np.allclose(np.sort(x), np.sort(surrogate))
    
    # 2. Autocorrelation match: lag-1 autocorrelation should be close
    autocorr_x = np.corrcoef(x[:-1], x[1:])[0, 1]
    autocorr_surr = np.corrcoef(surrogate[:-1], surrogate[1:])[0, 1]
    assert abs(autocorr_x - autocorr_surr) < 0.05
    
    # 3. Kolmogorov-Smirnov test: distributions should be indistinguishable (p-value > 0.05)
    ks_stat, p_val = stats.ks_2samp(x, surrogate)
    assert p_val > 0.05
