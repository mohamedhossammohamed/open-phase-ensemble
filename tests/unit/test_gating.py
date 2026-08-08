import numpy as np

from tsad.config import CUSUM_KC, T_DRIFT
from tsad.gating import CUSUMGating, GatingState
from tsad.hyperopt import BayesianHyperparameterTuner
from tsad.score_calibration import QuantileScoreCalibrator


def test_cusum_stable_signal_no_alarm():
    gating = CUSUMGating(k_c=CUSUM_KC, t_drift=T_DRIFT)
    np.random.seed(42)
    errors = np.random.normal(0.1, 0.01, size=100)
    
    for err in errors:
        state = gating.step(error=err)
        assert state == GatingState.NORMAL
        assert gating.is_adaptation_allowed()

def test_cusum_anomaly_spike_triggers_freeze():
    gating = CUSUMGating(k_c=CUSUM_KC, t_drift=T_DRIFT)
    np.random.seed(42)
    # Warm up baseline
    for _ in range(50):
        gating.step(error=float(np.random.normal(0.1, 0.01)))
        
    # Inject massive error spike -> triggers ALARM
    for _ in range(10):
        state = gating.step(error=10.0)
        
    assert state == GatingState.ANOMALY_ALARM
    assert not gating.is_adaptation_allowed()

test_drift_state = None

def test_cusum_persistent_drift_triggers_flush():
    gating = CUSUMGating(k_c=CUSUM_KC, t_drift=50) # Use smaller T_drift=50 for fast test
    np.random.seed(42)
    # Warm up
    for _ in range(50):
        gating.step(error=float(np.random.normal(0.1, 0.01)))
        
    states = []
    for _ in range(60):
        st = gating.step(error=2.0) # Permanent shift
        states.append(st)
        
    # Should transition to CONCEPT_DRIFT after 50 consecutive alarm steps
    assert GatingState.CONCEPT_DRIFT in states
    # After reset, adaptation should be allowed again
    assert gating.is_adaptation_allowed()

def test_bayesian_hyperparameter_bounds_and_hysteresis():
    tuner = BayesianHyperparameterTuner(hysteresis_margin=0.02)
    
    # Assert proposed tau and d stay within clamped bounds
    np.random.seed(42)
    for _ in range(10):
        tau, d = tuner.propose_next()
        assert 1 <= tau <= 100
        assert 1 <= d <= 20
        
    # Hysteresis test: improvement < margin should be rejected
    rejected = tuner.evaluate_candidate(skill_gain=0.01) # < 0.02
    assert not rejected
    
    accepted = tuner.evaluate_candidate(skill_gain=0.05) # >= 0.02
    assert accepted

def test_quantile_score_calibration():
    calibrator = QuantileScoreCalibrator()
    old_scores = np.random.uniform(0.0, 1.0, size=1000)
    new_scores = np.random.uniform(0.2, 0.8, size=1000) # Shifted scale
    
    calibrator.fit(old_scores, new_scores)
    
    # 0.85 percentile mapping check
    mapped = calibrator.calibrate(0.85)
    assert 0.0 <= mapped <= 1.0
