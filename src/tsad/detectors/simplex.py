
import numpy as np

from tsad.config import D_TARGET, EPSILON
from tsad.detectors.base import DetectorABC
from tsad.representation import HNSWIndex


class SimplexProjectionDetector(DetectorABC):
    """
    Detector 1: Simplex Projection (Empirical Dynamic Modeling).
    Tracks forward trajectory evolution of E+1 nearest phase-space neighbors.
    Calibrated score s_t outputs 0.0 for normal trajectory predictions.
    """
    def __init__(self, dim: int = D_TARGET, tau: int = 1, forecast_horizon: int = 1):
        self.dim = dim
        self.tau = tau
        self.forecast_horizon = forecast_horizon
        self.hnsw = HNSWIndex(dim=dim, max_elements=10000)
        self.v_history: list[float] = []
        self.Z_history: list[np.ndarray] = []
        self.err_history: list[float] = []
        self.last_weights = np.ones(dim + 1) / (dim + 1)

    def score(self, Z_t: np.ndarray, v_t: float) -> tuple[float, float]:
        E = self.dim
        k = E + 1
        
        if len(self.Z_history) <= k + self.forecast_horizon:
            return 0.0, v_t
            
        indices, distances = self.hnsw.knn_query(Z_t, k=k)
        
        if len(indices) == 0 or distances[0] == 0:
            d_min = distances[0] if len(distances) > 0 else EPSILON
        else:
            d_min = distances[0]
            
        weights = np.exp(-distances / (d_min + EPSILON))
        weights_sum = np.sum(weights)
        if weights_sum > 0:
            weights /= weights_sum
        else:
            weights = np.ones(k) / k
            
        self.last_weights = weights
        
        v_hat = 0.0
        valid_count = 0
        for idx, w in zip(indices, weights):
            target_idx = idx + self.forecast_horizon
            if target_idx < len(self.v_history):
                v_hat += w * self.v_history[target_idx]
                valid_count += 1
                
        if valid_count == 0:
            v_hat = v_t
            
        err = abs(v_t - v_hat)
        self.err_history.append(err)
        if len(self.err_history) > 500:
            self.err_history.pop(0)
            
        med_err = np.median(self.err_history)
        std_err = np.std(self.err_history) + EPSILON
        
        s_t = float(np.clip((err - (med_err + 2.0 * std_err)) / (3.0 * std_err), 0.0, 1.0))
        return s_t, float(v_hat)

    def update(self, v_true: float):
        self.v_history.append(v_true)

    def add_vector(self, Z_t: np.ndarray):
        self.Z_history.append(Z_t)
        self.hnsw.add_items(Z_t)
