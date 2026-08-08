import os

import numpy as np

from tsad.config import SEED

FIXTURES_DIR = os.path.dirname(os.path.abspath(__file__))

def generate_sine_fixture(n_points=10000, seed=SEED):
    """
    Synthesizes a 10,000-point sinusoidal wave with superimposed Gaussian noise
    and 3 explicitly injected anomalies:
    1. Point spike (+10 sigma) at index 3000
    2. Variance drop (amplitude x0.1) at indices 5000:5200
    3. Frequency shift (period 50 -> 25) at indices 7000:7500
    """
    np.random.seed(seed)
    t = np.arange(n_points)
    
    # Base sine wave with period 50 and noise std 0.05
    signal = np.sin(2 * np.pi * t / 50.0) + np.random.normal(0, 0.05, size=n_points)
    labels = np.zeros(n_points, dtype=int)
    
    # Anomaly 1: Point spike at index 3000
    signal[3000] += 10.0
    labels[3000] = 1
    
    # Anomaly 2: Amplitude suppression (variance drop) at indices 5000..5200
    signal[5000:5200] = np.sin(2 * np.pi * t[5000:5200] / 50.0) * 0.1 + np.random.normal(0, 0.005, size=200)
    labels[5000:5200] = 1
    
    # Anomaly 3: Frequency shift (period 25) at indices 7000..7500
    signal[7000:7500] = np.sin(2 * np.pi * t[7000:7500] / 25.0) + np.random.normal(0, 0.05, size=500)
    labels[7000:7500] = 1
    
    return signal, labels

def generate_lorenz_fixture(n_points=10000, dt=0.01, sigma=10.0, rho=28.0, beta=8.0/3.0):
    """Generates a 3D Lorenz attractor system trajectory."""
    xs = np.empty(n_points)
    ys = np.empty(n_points)
    zs = np.empty(n_points)
    
    xs[0], ys[0], zs[0] = (0.0, 1.0, 1.05)
    
    for i in range(n_points - 1):
        dx = sigma * (ys[i] - xs[i]) * dt
        dy = (xs[i] * (rho - zs[i]) - ys[i]) * dt
        dz = (xs[i] * ys[i] - beta * zs[i]) * dt
        xs[i + 1] = xs[i] + dx
        ys[i + 1] = ys[i] + dy
        zs[i + 1] = zs[i] + dz
        
    return xs, ys, zs

def get_or_create_sine_fixture():
    path = os.path.join(FIXTURES_DIR, "sine_fixture.npz")
    if not os.path.exists(path):
        signal, labels = generate_sine_fixture()
        np.savez_compressed(path, signal=signal, labels=labels)
    else:
        data = np.load(path)
        signal, labels = data["signal"], data["labels"]
    return signal, labels

def get_or_create_lorenz_fixture():
    path = os.path.join(FIXTURES_DIR, "lorenz_fixture.npz")
    if not os.path.exists(path):
        xs, ys, zs = generate_lorenz_fixture()
        np.savez_compressed(path, xs=xs, ys=ys, zs=zs)
    else:
        data = np.load(path)
        xs, ys, zs = data["xs"], data["ys"], data["zs"]
    return xs, ys, zs

if __name__ == "__main__":
    get_or_create_sine_fixture()
    get_or_create_lorenz_fixture()
    print("Fixtures generated successfully.")
