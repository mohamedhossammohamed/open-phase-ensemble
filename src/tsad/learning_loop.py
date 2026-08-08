import numpy as np
from tsad.config import K_DETECTORS, W_CORR, EPSILON

class OnlineLearningLoop:
    """
    Online Learning Loop Module.
    Calculates label-free loss signal based on negative Pearson correlation
    between anomaly scores S_t and error signals E_t.
    
    For predictive detectors: E_t = |v_t - v_hat_t|
    For spatial/manifold detectors (where v_hat == v_t): E_t = s_{t, k}
    """
    def __init__(self, k_detectors: int = K_DETECTORS, window_size: int = W_CORR):
        self.k = k_detectors
        self.window_size = window_size
        self.score_history = [[] for _ in range(k_detectors)]
        self.error_history = [[] for _ in range(k_detectors)]

    def step(self, true_v: float, forecasts: np.ndarray, scores: np.ndarray) -> np.ndarray:
        """
        Processes newly observed true value v_t, historical forecasts v_hat, and scores s_t.
        Returns loss vector ell_t in R^K.
        """
        loss_vec = np.zeros(self.k, dtype=np.float64)
        
        for k in range(self.k):
            raw_err = abs(true_v - forecasts[k])
            # If forecast is identical to true_v, use anomaly score as deviation metric
            if raw_err < EPSILON:
                e_k = scores[k]
            else:
                e_k = raw_err
                
            s_k = scores[k]
            
            self.score_history[k].append(s_k)
            self.error_history[k].append(e_k)
            
            if len(self.score_history[k]) > self.window_size:
                self.score_history[k].pop(0)
                self.error_history[k].pop(0)
                
            S_arr = np.array(self.score_history[k], dtype=np.float64)
            E_arr = np.array(self.error_history[k], dtype=np.float64)
            
            if len(S_arr) < 5:
                loss_vec[k] = 0.5
            else:
                s_std = np.std(S_arr)
                e_std = np.std(E_arr)
                
                if s_std < EPSILON or e_std < EPSILON:
                    loss_vec[k] = 0.5
                else:
                    corr = float(np.corrcoef(S_arr, E_arr)[0, 1])
                    if np.isnan(corr):
                        corr = 0.0
                    # Loss = 1.0 - corr (higher correlation -> lower loss)
                    loss_vec[k] = float(np.clip(1.0 - corr, 0.0, 1.0))
                    
        return loss_vec
