import numpy as np
from typing import Tuple
from tsad.detectors.base import DetectorABC
from tsad.config import EPSILON

class SARIMADetector(DetectorABC):
    """
    Detector 5: SARIMA Residuals (Linear Stochastic Control).
    Calibrated score outputs 0.0 for normal linear residuals.
    """
    def __init__(self, p: int = 2, d: int = 0, q: int = 1):
        self.p = p
        self.d = d
        self.q = q
        self.history = []
        self.coeffs = np.ones(p) / max(1, p)
        self.residuals = []

    def score(self, Z_t: np.ndarray, v_t: float) -> Tuple[float, float]:
        if len(self.history) < self.p:
            return 0.0, v_t
            
        recent = np.array(self.history[-self.p:], dtype=np.float64)
        v_hat = float(np.dot(recent[::-1], self.coeffs))
        
        err = abs(v_t - v_hat)
        self.residuals.append(err)
        if len(self.residuals) > 500:
            self.residuals.pop(0)
            
        med_err = np.median(self.residuals)
        std_err = np.std(self.residuals) + EPSILON
        
        s_t = float(np.clip((err - (med_err + 2.0 * std_err)) / (3.0 * std_err), 0.0, 1.0))
        return s_t, v_hat

    def update(self, v_true: float):
        self.history.append(v_true)
        if len(self.history) > 1000:
            self.history.pop(0)
            
        if len(self.history) >= 20 and len(self.history) % 20 == 0:
            self._fit_ar()

    def _fit_ar(self):
        y = np.array(self.history[self.p:], dtype=np.float64)
        N = len(y)
        X = np.zeros((N, self.p), dtype=np.float64)
        for i in range(N):
            X[i] = self.history[i : i + self.p][::-1]
            
        try:
            lambda_reg = 1e-3
            XtX = np.dot(X.T, X) + lambda_reg * np.eye(self.p)
            Xty = np.dot(X.T, y)
            self.coeffs = np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError:
            pass
