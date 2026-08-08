import numpy as np

from tsad.config import EPSILON


class StreamBuffer:
    """
    Ingestion & Preprocessing Module.
    Rolling median/MAD standardization.
    v_t = (x_t - Median(W)) / (1.4826 * MAD(W) + EPSILON)
    """
    def __init__(self, window_size: int = 200):
        self.window_size = window_size
        self.raw_buffer: list[float] = []
        self.standardized_buffer: list[float] = []

    def step(self, x_t: float) -> float:
        """
        Ingests scalar x_t, handles NaNs via forward-fill, returns robust Z-score v_t.
        """
        if np.isnan(x_t) or np.isinf(x_t):
            x_t = float(self.raw_buffer[-1]) if len(self.raw_buffer) > 0 else 0.0
            
        if self.raw_buffer:
            buf_arr = np.array(self.raw_buffer, dtype=np.float64)
            med = float(np.median(buf_arr))
            mad = float(np.median(np.abs(buf_arr - med))) + EPSILON
        else:
            med = float(x_t)
            mad = EPSILON
        
        v_t = (x_t - med) / (1.4826 * mad)
        self.raw_buffer.append(x_t)
        if len(self.raw_buffer) > self.window_size:
            self.raw_buffer.pop(0)
        self.standardized_buffer.append(v_t)
        if len(self.standardized_buffer) > self.window_size:
            self.standardized_buffer.pop(0)
        return float(v_t)

    def get_buffer(self) -> np.ndarray:
        return np.array(self.raw_buffer, dtype=np.float64)

    def get_standardized_buffer(self) -> np.ndarray:
        """Returns the causal rolling history of standardized observations."""
        return np.array(self.standardized_buffer, dtype=np.float64)

    def get_standardized_tail(self, length: int) -> np.ndarray:
        """Returns only the recent standardized values needed by an embedding."""
        if length <= 0:
            return np.empty(0, dtype=np.float64)
        return np.array(self.standardized_buffer[-length:], dtype=np.float64)

    def __len__(self) -> int:
        """Returns current number of elements stored in buffer."""
        return len(self.raw_buffer)
