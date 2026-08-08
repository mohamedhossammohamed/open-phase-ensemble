
import numpy as np

from tsad.config import D_TARGET, EPSILON
from tsad.detectors.base import DetectorABC


class RobustMahalanobisDetector(DetectorABC):
    """
    Detector 2: Robust Mahalanobis Distance with Ledoit-Wolf Shrinkage.
    Calibrated anomaly score: s_t in [0, 1] where 0 represents normal background noise.
    """
    def __init__(self, dim: int = D_TARGET, block_size: int = 1000):
        self.dim = dim
        self.block_size = block_size
        self.buffer = []
        self.mu = 0.0
        self.delta = 0.0
        self.Sigma_LW = np.eye(dim, dtype=np.float64)
        self.inv_Sigma = np.eye(dim, dtype=np.float64)
        self.mean = np.zeros(dim, dtype=np.float64)
        self.dist_history = []

    def _update_covariance(self):
        if len(self.buffer) < 2:
            return
            
        X = np.array(self.buffer, dtype=np.float64)
        n, d = X.shape
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        Sigma_sample = np.dot(X_centered.T, X_centered) / max(1, n - 1)
        self.mu = float(np.trace(Sigma_sample) / d)
        
        target = self.mu * np.eye(d)
        outer_products = X_centered[:, :, None] * X_centered[:, None, :]
        delta_num = float(np.sum((outer_products - Sigma_sample) ** 2)) / float(n ** 2)
        
        delta_den = np.sum((Sigma_sample - target) ** 2)
        if delta_den < EPSILON:
            self.delta = 1.0
        else:
            self.delta = float(np.clip(delta_num / delta_den, 0.001, 1.0))
            
        self.Sigma_LW = (1.0 - self.delta) * Sigma_sample + self.delta * target
        
        try:
            self.inv_Sigma = np.linalg.pinv(self.Sigma_LW)
        except np.linalg.LinAlgError:
            self.inv_Sigma = np.eye(d)

    def score(self, Z_t: np.ndarray, v_t: float) -> tuple[float, float]:
        diff = Z_t - self.mean
        dist_sq = float(np.dot(np.dot(diff, self.inv_Sigma), diff.T))
        dist = np.sqrt(max(0.0, dist_sq))
        
        self.dist_history.append(dist)
        if len(self.dist_history) > 500:
            self.dist_history.pop(0)
            
        # Calibrate score using 3-sigma baseline: normal baseline outputs ~0.0
        med_dist = np.median(self.dist_history)
        std_dist = np.std(self.dist_history) + EPSILON
        
        s_t = float(np.clip((dist - (med_dist + 2.0 * std_dist)) / (3.0 * std_dist), 0.0, 1.0))
        v_hat = float(self.mean[0]) if len(self.buffer) > 0 else v_t
        return s_t, v_hat

    def update(self, v_true: float):
        pass

    def add_vector(self, Z_t: np.ndarray):
        self.buffer.append(Z_t.copy())
        if len(self.buffer) > self.block_size:
            self.buffer.pop(0)
        self._update_covariance()
