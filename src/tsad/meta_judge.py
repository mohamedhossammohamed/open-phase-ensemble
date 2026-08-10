
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
    Divides storage into quantile bins based on anomaly score A_t to preserve
    statistically representative samples of both rare anomalies and nominal states.

    NOTE: Currently write-only in the live pipeline — add() is called every step
    but sample() is never invoked. Retained for future replay-based training.
    Does not affect any pipeline output or benchmark score.
    """
    def __init__(self, capacity: int = REPLAY_BUFFER_SIZE, n_quantiles: int = 5):
        self.capacity = capacity
        self.n_quantiles = max(1, n_quantiles)
        self.quantile_capacity = max(1, capacity // self.n_quantiles)
        self.bins: list[list[tuple[np.ndarray, float]]] = [[] for _ in range(self.n_quantiles)]

    def _get_bin_index(self, A_t: float) -> int:
        clamped = float(np.clip(A_t, 0.0, 0.999999))
        return min(self.n_quantiles - 1, int(clamped * self.n_quantiles))

    def add(self, Z_t: np.ndarray, A_t: float):
        bin_idx = self._get_bin_index(A_t)
        target_bin = self.bins[bin_idx]
        if len(target_bin) >= self.quantile_capacity:
            target_bin.pop(0)
        target_bin.append((Z_t.copy(), float(A_t)))

    def sample(self, batch_size: int = 32, n: int | None = None) -> tuple[list[np.ndarray], list[float]] | np.ndarray:
        if n is not None:
            batch_size = n

        non_empty_bins = [b for b in self.bins if len(b) > 0]
        if not non_empty_bins:
            return np.empty((0, 8)) if n is not None else ([], [])

        per_bin_count = max(1, batch_size // len(non_empty_bins))
        sampled_pairs: list[tuple[np.ndarray, float]] = []

        for b in non_empty_bins:
            sample_size = min(len(b), per_bin_count)
            indices = np.random.choice(len(b), size=sample_size, replace=False)
            for idx in indices:
                sampled_pairs.append(b[idx])

        if len(sampled_pairs) > batch_size:
            sampled_pairs = sampled_pairs[:batch_size]

        Z_batch = [pair[0] for pair in sampled_pairs]
        A_batch = [pair[1] for pair in sampled_pairs]

        if n is not None:
            return np.array(Z_batch) if Z_batch else np.empty((0, 8))
        return Z_batch, A_batch

    def __len__(self) -> int:
        return sum(len(b) for b in self.bins)

