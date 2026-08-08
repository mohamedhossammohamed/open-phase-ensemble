import numpy as np
from typing import Tuple
from tsad.config import K_DETECTORS, HEDGE_ETA, FIXED_SHARE_SIGMA, REPLAY_BUFFER_SIZE, EPSILON

class MetaJudge:
    """
    Meta-Judge Fusion Module using the Hedge (Exponential Weights) online learning algorithm
    with Fixed-Share mixing step.
    """
    def __init__(self, k_detectors: int = K_DETECTORS, eta: float = HEDGE_ETA, sigma: float = FIXED_SHARE_SIGMA):
        self.k = k_detectors
        self.eta = eta
        self.sigma = sigma
        self.weights = np.ones(k_detectors, dtype=np.float64) / k_detectors

    def fuse(self, scores: np.ndarray, forecasts: np.ndarray) -> Tuple[float, float]:
        """
        Calculates convex combination (dot product) of detector outputs and weights.
        Enhanced with max-activation pooling to prevent signal dilution by inactive experts.
        """
        weighted_dot = float(np.dot(self.weights, scores))
        max_active = float(np.max(self.weights * scores * float(self.k)))
        
        A_t = max(weighted_dot, max_active)
        A_t = float(np.clip(A_t, 0.0, 1.0))
        
        v_hat_star = float(np.dot(self.weights, forecasts))
        return A_t, v_hat_star

    def update_weights(self, loss_vector: np.ndarray):
        """
        Hedge multiplicative weight update rule + Fixed-share mixing step.
        """
        exp_loss = np.exp(-self.eta * loss_vector)
        unnorm_weights = self.weights * exp_loss
        weight_sum = np.sum(unnorm_weights)
        
        if weight_sum > 0:
            w_next = unnorm_weights / weight_sum
        else:
            w_next = np.ones(self.k, dtype=np.float64) / self.k
            
        w_share = (1.0 - self.sigma) * w_next + (self.sigma / float(self.k))
        self.weights = w_share / np.sum(w_share)

class StratifiedReplayBuffer:
    """
    Stratified Reservoir Sampling Replay Buffer.
    Divides buffer into quantiles based on anomaly score A_t.
    """
    def __init__(self, capacity: int = REPLAY_BUFFER_SIZE, n_quantiles: int = 10):
        self.capacity = capacity
        self.n_quantiles = n_quantiles
        self.quantile_bins = [[] for _ in range(n_quantiles)]
        self.total_count = 0

    def add(self, Z_t: np.ndarray, A_t: float):
        """Adds vector Z_t to corresponding score quantile bin."""
        bin_idx = int(np.clip(A_t * self.n_quantiles, 0, self.n_quantiles - 1))
        max_bin_cap = max(1, self.capacity // self.n_quantiles)
        
        target_bin = self.quantile_bins[bin_idx]
        if len(target_bin) < max_bin_cap:
            target_bin.append(Z_t.copy())
        else:
            idx = np.random.randint(0, len(target_bin))
            target_bin[idx] = Z_t.copy()
            
        self.total_count += 1

    def sample(self, n: int) -> np.ndarray:
        """Samples n items uniformly across quantiles."""
        all_items = []
        for b in self.quantile_bins:
            all_items.extend(b)
            
        if len(all_items) == 0:
            return np.array([])
            
        sample_size = min(n, len(all_items))
        indices = np.random.choice(len(all_items), size=sample_size, replace=False)
        return np.array([all_items[i] for i in indices], dtype=np.float64)

    def __len__(self) -> int:
        return sum(len(b) for b in self.quantile_bins)
