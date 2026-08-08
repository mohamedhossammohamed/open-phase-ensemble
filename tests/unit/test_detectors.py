import numpy as np
import pytest
from tsad.detectors.mahalanobis import RobustMahalanobisDetector
from tsad.detectors.iforest import IsolationForestDetector
from tsad.detectors.simplex import SimplexProjectionDetector
from tsad.detectors.matrix_profile import MatrixProfileDetector
from tsad.detectors.sarima import SARIMADetector
from tsad.detectors.transformer import AnomalyTransformerDetector
from tests.fixtures.generate_fixtures import get_or_create_sine_fixture

def test_ledoit_wolf_math_and_shrinkage():
    np.random.seed(42)
    base = np.random.randn(200, 2)
    proj = np.random.randn(2, 8)
    X_singular = np.dot(base, proj)
    
    det = RobustMahalanobisDetector(dim=8, block_size=200)
    for row in X_singular:
        score, forecast = det.score(row, v_t=row[0])
        det.update(row[0])
        
    assert det.delta > 0.0
    det_val = np.linalg.det(det.Sigma_LW)
    assert det_val > 0.0
    assert not np.isnan(score)

def test_isolation_forest_detector():
    np.random.seed(42)
    det = IsolationForestDetector(dim=8, n_estimators=50, subsample=100)
    
    normal_data = np.random.normal(0, 1, size=(200, 8))
    for row in normal_data:
        det.score(row, v_t=row[0])
        det.update(row[0])
        
    normal_score, _ = det.score(normal_data[0], v_t=0.0)
    outlier_score, _ = det.score(np.array([10.0] * 8), v_t=10.0)
    
    assert outlier_score > normal_score

def test_simplex_projection_detector():
    signal, labels = get_or_create_sine_fixture()
    det = SimplexProjectionDetector(dim=8, tau=2, forecast_horizon=1)
    
    forecasts = []
    for i in range(100):
        Z = np.ones(8) * signal[i]
        s_t, v_hat = det.score(Z, v_t=signal[i])
        det.update(signal[i])
        forecasts.append(v_hat)
        
    assert abs(np.sum(det.last_weights) - 1.0) < 1e-6

def test_matrix_profile_detector():
    signal, labels = get_or_create_sine_fixture()
    det = MatrixProfileDetector(w_mp=10)
    
    scores = []
    for i in range(3500):
        Z = np.zeros(8)
        s_t, _ = det.score(Z, v_t=signal[i])
        det.update(signal[i])
        scores.append(s_t)
        
    # Anomaly 1 at index 3000 will elevate score in the window [3000:3010]
    spike_score = np.max(scores[3000:3010])
    normal_score = np.median(scores[100:2000])
    assert spike_score > normal_score

def test_sarima_detector():
    np.random.seed(42)
    det = SARIMADetector(p=1, d=0, q=1)
    
    x = 0.0
    for _ in range(50):
        x = 0.7 * x + np.random.normal(0, 0.1)
        score, forecast = det.score(np.zeros(8), v_t=x)
        det.update(x)
        
    assert 0.0 <= score <= 1.0

def test_anomaly_transformer_detector():
    det = AnomalyTransformerDetector(dim=8, d_model=32, n_heads=4, n_layers=2)
    Z = np.random.randn(8)
    s_t, forecast = det.score(Z, v_t=0.5)
    det.update(0.5)
    
    assert 0.0 <= s_t <= 1.0
    assert not np.isnan(s_t)
    assert not np.isnan(forecast)
