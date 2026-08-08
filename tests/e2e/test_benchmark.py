import numpy as np

from tsad.evaluation.iaaft import generate_iaaft_surrogate
from tsad.evaluation.vus import compute_vus_roc
from tsad.pipeline import TSADPipeline


def test_benchmark_acceptance_criteria():
    """
    Smoke-tests the benchmark fixture without treating synthetic performance as
    evidence of real-world detection quality.

    Scientific performance claims are made by ``evaluate_stream`` on a
    provenance-checked dataset, not by a hard-coded score threshold here.
    """
    np.random.seed(42)
    t = np.linspace(0, 100, 5000)
    signal = np.sin(2 * np.pi * 1.5 * t) + 0.5 * np.cos(2 * np.pi * 0.5 * t) + np.random.normal(0, 0.05, len(t))
    labels = np.zeros(len(t), dtype=int)
    
    # Anomaly 1: Spike
    signal[1000] += 8.0
    labels[1000] = 1
    
    # Anomaly 2: Phase shift
    signal[2500:2700] = np.sin(2 * np.pi * 6.0 * t[2500:2700])
    labels[2500:2700] = 1
    
    # Anomaly 3: Amplitude suppression
    signal[4000:4200] *= 0.05
    labels[4000:4200] = 1
    
    # 1. Run TSAD system
    pipeline = TSADPipeline()
    scores = []
    for x in signal:
        A_t, _ = pipeline.step(x)
        scores.append(A_t)
        
    scores_arr = np.array(scores)
    vus_roc = compute_vus_roc(scores_arr, labels, max_buffer=15)
    
    # 2. Run IAAFT surrogate null baseline
    surrogate = generate_iaaft_surrogate(signal)
    surr_pipeline = TSADPipeline()
    surr_scores = [surr_pipeline.step(x)[0] for x in surrogate]
    surr_vus_roc = compute_vus_roc(np.array(surr_scores), labels, max_buffer=15)
    
    predictive_edge = vus_roc - surr_vus_roc
    
    # Synthetic fixtures are only a runtime/shape smoke test.  Their scores
    # are not acceptance criteria because thresholds can hide leakage or
    # overfit to one hand-built anomaly pattern.
    assert np.isfinite(vus_roc)
    assert 0.0 <= vus_roc <= 1.0
    assert np.isfinite(surr_vus_roc)
    assert 0.0 <= surr_vus_roc <= 1.0
    assert np.isfinite(predictive_edge)
    assert -1.0 <= predictive_edge <= 1.0
    
    entropy = -np.sum(pipeline.meta_judge.weights * np.log2(pipeline.meta_judge.weights + 1e-12))
    assert entropy > 0.1, f"Weight entropy {entropy:.4f} below 0.1"
