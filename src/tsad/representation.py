import numpy as np
from typing import Tuple
from tsad.config import D_TARGET, MAX_TAU, MAX_D, R_TOL, A_TOL, EPSILON, SEED

try:
    import hnswlib
    HAS_HNSW = True
except ImportError:
    HAS_HNSW = False

def compute_ami(v: np.ndarray, max_lag: int = MAX_TAU, n_bins: int = 30) -> int:
    """
    Computes Average Mutual Information (AMI) for lags up to max_lag.
    Returns lag tau corresponding to the first local minimum.
    """
    if len(v) < max_lag + 2:
        return 1
        
    n = len(v)
    v_min, v_max = np.min(v), np.max(v)
    if v_max - v_min < EPSILON:
        return 1
        
    hist_v, bin_edges = np.histogram(v, bins=n_bins, range=(v_min, v_max))
    p_v = hist_v / float(n)
    p_v = p_v[p_v > 0]
    H_v = -np.sum(p_v * np.log2(p_v))
    
    ami_list = []
    for lag in range(1, max_lag + 1):
        x = v[:-lag]
        y = v[lag:]
        hist_2d, _, _ = np.histogram2d(x, y, bins=n_bins, range=[[v_min, v_max], [v_min, v_max]])
        p_xy = hist_2d / float(len(x))
        p_x = np.sum(p_xy, axis=1)
        p_y = np.sum(p_xy, axis=0)
        
        # I(X;Y) = H(X) + H(Y) - H(X,Y)
        p_xy_flat = p_xy.flatten()
        p_xy_pos = p_xy_flat[p_xy_flat > 0]
        H_xy = -np.sum(p_xy_pos * np.log2(p_xy_pos))
        
        p_x_pos = p_x[p_x > 0]
        p_y_pos = p_y[p_y > 0]
        H_x = -np.sum(p_x_pos * np.log2(p_x_pos))
        H_y = -np.sum(p_y_pos * np.log2(p_y_pos))
        
        I_lag = max(0.0, H_x + H_y - H_xy)
        ami_list.append(I_lag)
        
    # Find first local minimum
    for i in range(1, len(ami_list) - 1):
        if ami_list[i] < ami_list[i - 1] and ami_list[i] <= ami_list[i + 1]:
            return i + 1
            
    return int(np.argmin(ami_list) + 1)

def compute_fnn(v: np.ndarray, tau: int, max_d: int = MAX_D, r_tol: float = R_TOL, a_tol: float = A_TOL) -> int:
    """
    Computes False Nearest Neighbors (FNN) fraction (Kennel et al.).
    Returns optimal embedding dimension d.
    """
    std_v = np.std(v) + EPSILON
    for d in range(1, max_d):
        X_d = delay_embed(v, tau=tau, d=d)
        X_d1 = delay_embed(v, tau=tau, d=d+1)
        
        n_points = min(len(X_d), len(X_d1))
        if n_points < 10:
            return d
            
        X_d = X_d[:n_points]
        X_d1 = X_d1[:n_points]
        
        # Query 1-NN in d dimensions
        fnn_count = 0
        for i in range(min(n_points, 500)):
            dists = np.linalg.norm(X_d - X_d[i], axis=1)
            dists[i] = float("inf")
            nn_idx = np.argmin(dists)
            R_d = dists[nn_idx]
            
            if R_d < EPSILON:
                continue
                
            R_d1 = abs(X_d1[i, -1] - X_d1[nn_idx, -1])
            R_total = np.sqrt(R_d ** 2 + R_d1 ** 2)
            
            # Kennel's criteria
            if (R_d1 / R_d > r_tol) or (R_total / std_v > a_tol):
                fnn_count += 1
                
        fnn_frac = fnn_count / float(min(n_points, 500))
        if fnn_frac < 0.01:
            return d
            
    return max_d

def delay_embed(v: np.ndarray, tau: int, d: int) -> np.ndarray:
    """
    Reconstructs phase space trajectory matrix X using Takens' delay embedding.
    X_t = [v_t, v_{t-tau}, ..., v_{t-(d-1)tau}]
    """
    n = len(v)
    max_lag = (d - 1) * tau
    if n <= max_lag:
        return np.tile(v, (1, d))
        
    num_vectors = n - max_lag
    X = np.zeros((num_vectors, d), dtype=np.float64)
    for i in range(d):
        X[:, i] = v[max_lag - i * tau : n - i * tau]
    return X

def compress_projection(X: np.ndarray, target_d: int = D_TARGET, seed: int = SEED) -> np.ndarray:
    """
    Johnson-Lindenstrauss Random Projection to reduce or zero-pad dimension to D_TARGET.
    """
    n, d = X.shape
    if d == target_d:
        return X.copy()
    elif d < target_d:
        padded = np.zeros((n, target_d), dtype=np.float64)
        padded[:, :d] = X
        return padded
    else:
        rng = np.random.RandomState(seed)
        R = rng.randn(d, target_d) / np.sqrt(target_d)
        return np.dot(X, R)

def online_mad_normalize(Z_t: np.ndarray, window: np.ndarray) -> np.ndarray:
    """
    Multidimensional MAD normalization.
    """
    med = np.median(window)
    mad = np.median(np.abs(window - med)) + EPSILON
    return (Z_t - med) / (1.4826 * mad)

class HNSWIndex:
    """
    Approximate Nearest Neighbor graph search index wrapper (hnswlib with scipy/numpy fallback).
    Resizes dynamically when max_elements is reached.
    """
    def __init__(self, dim: int = D_TARGET, max_elements: int = 100000):
        self.dim = dim
        self.max_elements = max_elements
        self.count = 0
        
        if HAS_HNSW:
            self.index = hnswlib.Index(space="l2", dim=dim)
            self.index.init_index(max_elements=max_elements, ef_construction=100, M=16)
            self.index.set_ef(30)
        else:
            self.index = None
            self.data = []

    def add_items(self, data: np.ndarray):
        data_2d = np.atleast_2d(data).astype(np.float32)
        n = len(data_2d)
        
        if HAS_HNSW:
            if self.count + n > self.max_elements:
                self.max_elements *= 2
                self.index.resize_index(self.max_elements)
                
            ids = np.arange(self.count, self.count + n)
            self.index.add_items(data_2d, ids)
            self.count += n
        else:
            for row in data_2d:
                self.data.append(row)
                self.count += 1

    def knn_query(self, query: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        if self.count == 0:
            return np.array([]), np.array([])
            
        q_2d = query.reshape(1, -1).astype(np.float32)
        k_actual = min(k, self.count)
        
        if HAS_HNSW:
            labels, distances = self.index.knn_query(q_2d, k=k_actual)
            return labels[0], np.sqrt(np.maximum(0.0, distances[0]))
        else:
            arr = np.array(self.data)
            dists = np.linalg.norm(arr - q_2d[0], axis=1)
            indices = np.argsort(dists)[:k_actual]
            return indices, dists[indices]
