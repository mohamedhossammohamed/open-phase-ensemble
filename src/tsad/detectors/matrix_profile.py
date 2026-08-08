import numpy as np
from typing import Tuple
from tsad.detectors.base import DetectorABC
from tsad.config import EPSILON

class MatrixProfileDetector(DetectorABC):
    """
    Detector 3: Fast Subsequence Distance Search (STOMP Matrix Profile).
    Calibrated score outputs 0.0 for normal historical patterns.
    """
    def __init__(self, w_mp: int = 10, max_history: int = 500):
        self.w_mp = max(3, w_mp)
        self.max_history = max_history
        self.history = []
        self.last_profile_val = 0.0
        self.profile_history = []

    def score(self, Z_t: np.ndarray, v_t: float) -> Tuple[float, float]:
        n = len(self.history)
        if n < 2 * self.w_mp:
            return 0.0, v_t
            
        current_subseq = np.array(self.history[-self.w_mp:], dtype=np.float64)
        hist_arr = np.array(self.history[:-self.w_mp], dtype=np.float64)
        
        if len(hist_arr) >= self.w_mp:
            windows = np.lib.stride_tricks.sliding_window_view(hist_arr, self.w_mp)
            if len(windows) > 200:
                windows = windows[::2]
                
            dists = np.linalg.norm(windows - current_subseq, axis=1)
            min_dist = float(np.min(dists))
        else:
            min_dist = 0.0
            
        self.last_profile_val = min_dist
        self.profile_history.append(min_dist)
        if len(self.profile_history) > 500:
            self.profile_history.pop(0)
            
        med_p = np.median(self.profile_history)
        std_p = np.std(self.profile_history) + EPSILON
        
        s_t = float(np.clip((min_dist - (med_p + 2.0 * std_p)) / (3.0 * std_p), 0.0, 1.0))
        v_hat = float(v_t)
        return s_t, v_hat

    def update(self, v_true: float):
        self.history.append(v_true)
        if len(self.history) > self.max_history:
            self.history.pop(0)
