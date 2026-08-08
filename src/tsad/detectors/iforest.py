
import numpy as np
from sklearn.ensemble import IsolationForest

from tsad.config import D_TARGET, SEED
from tsad.detectors.base import DetectorABC


class IsolationForestDetector(DetectorABC):
    """
    Detector 4: Isolation Forest for subspace partitioning & global point anomalies.
    T=50 isolation trees, subsample S=256. Calibrated score outputs 0.0 for normal baseline.
    """
    def __init__(self, dim: int = D_TARGET, n_estimators: int = 50, subsample: int = 256):
        self.dim = dim
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.buffer = []
        self.model = IsolationForest(
            n_estimators=n_estimators,
            max_samples=min(subsample, 256),
            random_state=SEED,
            warm_start=False,
            n_jobs=-1
        )
        self.is_fitted = False
        self.step_counter = 0
        self.raw_scores_history = []

    def fit_from_buffer(self, buffer_data: np.ndarray):
        """Fit Isolation Forest model from buffer data."""
        if len(buffer_data) >= 50:
            self.model.fit(buffer_data)
            self.is_fitted = True

    def score(self, Z_t: np.ndarray, v_t: float) -> tuple[float, float]:
        if not self.is_fitted:
            return 0.0, v_t
            
        Z_2d = Z_t.reshape(1, -1)
        raw_score = -self.model.score_samples(Z_2d)[0]
        self.raw_scores_history.append(raw_score)
        if len(self.raw_scores_history) > 500:
            self.raw_scores_history.pop(0)
            
        med_score = np.median(self.raw_scores_history)
        std_score = np.std(self.raw_scores_history) + 1e-6
        
        s_t = float(np.clip((raw_score - (med_score + 2.0 * std_score)) / (3.0 * std_score), 0.0, 1.0))
        v_hat = float(v_t)
        return s_t, v_hat

    def update(self, v_true: float):
        pass

    def add_vector(self, Z_t: np.ndarray):
        self.buffer.append(Z_t.copy())
        self.step_counter += 1
        if len(self.buffer) > 1000:
            self.buffer.pop(0)
        if len(self.buffer) >= 50 and (not self.is_fitted or self.step_counter % 500 == 0):
            self.fit_from_buffer(np.array(self.buffer))
