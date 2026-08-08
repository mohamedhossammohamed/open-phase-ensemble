import numpy as np

def generate_iaaft_surrogate(x: np.ndarray, max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
    """
    Generates an Iterative Amplitude Adjusted Fourier Transform (IAAFT) surrogate.
    Preserves linear autocorrelation and amplitude distribution while destroying non-linear phase structures.
    """
    n = len(x)
    sorted_x = np.sort(x)
    abs_fft_x = np.abs(np.fft.rfft(x))
    
    # Initialize phase-randomized surrogate
    random_phases = np.random.uniform(0, 2 * np.pi, size=len(abs_fft_x))
    # Keep DC component phase zero
    random_phases[0] = 0.0
    if n % 2 == 0:
        random_phases[-1] = 0.0
        
    s_fft = abs_fft_x * np.exp(1j * random_phases)
    y = np.fft.irfft(s_fft, n=n)
    
    prev_rmse = float("inf")
    
    for _ in range(max_iter):
        # 1. Rank order match: replace values with sorted original values
        rank_indices = np.argsort(np.argsort(y))
        y_amp = sorted_x[rank_indices]
        
        # 2. Fourier transform adjustment: replace magnitudes with original |FFT(x)|
        fft_y = np.fft.rfft(y_amp)
        phases_y = np.angle(fft_y)
        new_fft = abs_fft_x * np.exp(1j * phases_y)
        y = np.fft.irfft(new_fft, n=n)
        
        # Convergence check
        rmse = np.sqrt(np.mean((np.abs(np.fft.rfft(y)) - abs_fft_x) ** 2))
        if abs(prev_rmse - rmse) < tol:
            break
        prev_rmse = rmse
        
    # Final rank match
    rank_indices = np.argsort(np.argsort(y))
    surrogate = sorted_x[rank_indices]
    return surrogate
