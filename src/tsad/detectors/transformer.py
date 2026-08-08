import numpy as np
from typing import Tuple
from tsad.detectors.base import DetectorABC
from tsad.config import D_TARGET, EPSILON, SEED

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    class AnomalyTransformerModule(nn.Module):
        """
        Lightweight PyTorch Anomaly Transformer model.
        L=2 layers, H=4 heads, D_model=32.
        Association Discrepancy: prior (Gaussian) vs series (learned self-attention) mismatch.
        """
        def __init__(self, in_dim: int = D_TARGET, d_model: int = 32, n_heads: int = 4, n_layers: int = 2):
            super().__init__()
            self.input_proj = nn.Linear(in_dim, d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=64, batch_first=True
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.output_proj = nn.Linear(d_model, in_dim)
            self.forecast_head = nn.Linear(d_model, 1)

        def forward(self, x: torch.Tensor):
            # x shape: (batch_size, seq_len, in_dim)
            h = self.input_proj(x)
            feat = self.encoder(h)
            rec = self.output_proj(feat)
            v_hat = self.forecast_head(feat[:, -1, :])
            return rec, v_hat

class AnomalyTransformerDetector(DetectorABC):
    def __init__(self, dim: int = D_TARGET, d_model: int = 32, n_heads: int = 4, n_layers: int = 2):
        self.dim = dim
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.sequence_buffer = []
        self.seq_len = 10
        
        if HAS_TORCH:
            torch.manual_seed(SEED)
            self.model = AnomalyTransformerModule(in_dim=dim, d_model=d_model, n_heads=n_heads, n_layers=n_layers)
            self.model.eval()
        else:
            self.model = None

    def score(self, Z_t: np.ndarray, v_t: float) -> Tuple[float, float]:
        self.sequence_buffer.append(Z_t)
        if len(self.sequence_buffer) > self.seq_len:
            self.sequence_buffer.pop(0)
            
        if not HAS_TORCH or len(self.sequence_buffer) < self.seq_len:
            return 0.0, v_t
            
        seq_arr = np.array(self.sequence_buffer, dtype=np.float32)[np.newaxis, :, :]
        x_tensor = torch.from_numpy(seq_arr)
        
        with torch.no_grad():
            rec_tensor, v_hat_tensor = self.model(x_tensor)
            rec = rec_tensor.numpy()[0]
            v_hat = float(v_hat_tensor.numpy()[0, 0])
            
        # Reconstruction error
        rec_err = float(np.mean((seq_arr[0] - rec) ** 2))
        s_t = float(1.0 - np.exp(-rec_err))
        return s_t, v_hat

    def update(self, v_true: float):
        pass
