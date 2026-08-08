
import numpy as np

from tsad.config import (
    EPSILON,
    FIXED_SHARE_SIGMA,
    HEDGE_ETA,
    K_DETECTORS,
    REPLAY_BUFFER_SIZE,
)


class MetaJudge:
    """
    Module 4: Meta-Judge Online Ensemble Combiner.
    Applies Hedge Multiplicative Weights Update Algorithm (Freund & Schapire, 1997)
    with Fixed-Share Mixing Floor (Herbster & Warmuth, 1998).
    Fused output A_t is the mathematically correct convex weighted sum of detector scores.
    """
    def __init__(self, k_detectors: int = K_DETECTORS, eta: float = HEDGE_ETA, sigma: float = FIXED_SHARE_SIGMA):
        self.k = k_detectors
        self.eta = eta
        self.sigma = sigma
        self.weights = np.ones(self.k, dtype=np.float64) / self.k

    def update_weights(self, loss_vector: np.ndarray):
        """
        Updates expert weights based on loss vector loss_vector in [0, 1]^K.
        Applies multiplicative update followed by fixed-share mixing floor.
        """
        loss_vector = np.clip(loss_vector, 0.0, 1.0)
        
        # 1. Multiplicative update step
        unnorm_weights = self.weights * np.exp(-self.eta * loss_vector)
        sum_w = np.sum(unnorm_weights) + EPSILON
        w_bar = unnorm_weights / sum_w
        
        # 2. Fixed-Share mixing step (guarantees weight floor w_k >= sigma / K)
        pool = np.sum(w_bar * self.sigma)
        self.weights = (1.0 - self.sigma) * w_bar + (pool / self.k)
        
        # Renormalize to ensure exact convex sum = 1.0
        self.weights = self.weights / np.sum(self.weights)

    def fuse(self, scores: np.ndarray, forecasts: np.ndarray) -> tuple[float, float]:
        """
        Calculates fused anomaly score A_t and fused forecast v_hat* via convex weighted sum.
        A_t = dot(weights, scores)
        v_hat* = dot(weights, forecasts)
        """
        A_t = float(np.dot(self.weights, scores))
        v_hat_star = float(np.dot(self.weights, forecasts))
        return A_t, v_hat_star

class StratifiedReplayBuffer:
    """
    Stratified Replay Buffer storing past state vectors Z_t and fused scores A_t.
    """
    def __init__(self, capacity: int = REPLAY_BUFFER_SIZE, n_quantiles: int = 5):
        self.capacity = capacity
        self.n_quantiles = n_quantiles
        self.buffer = []

    def add(self, Z_t: np.ndarray, A_t: float):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((Z_t.copy(), float(A_t)))

    def sample(self, batch_size: int = 32, n: int | None = None):
        if n is not None:
            batch_size = n
        if len(self.buffer) == 0:
            return np.empty((0, 8)) if n is not None else ([], [])
        indices = np.random.choice(len(self.buffer), size=min(batch_size, len(self.buffer)), replace=False)
        Z_batch = [self.buffer[i][0] for i in indices]
        A_batch = [self.buffer[i][1] for i in indices]
        if n is not None:
            return np.array(Z_batch)
        return Z_batch, A_batch

    def __len__(self) -> int:
        return len(self.buffer)
