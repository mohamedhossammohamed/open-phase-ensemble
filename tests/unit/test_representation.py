import numpy as np

from tests.fixtures.generate_fixtures import generate_lorenz_fixture
from tsad.config import A_TOL, D_TARGET, R_TOL
from tsad.representation import (
    HNSWIndex,
    compress_projection,
    compute_ami,
    compute_fnn,
    delay_embed,
    online_mad_normalize,
)


def test_ami_sine_wave():
    # Sinusoid with period 50 -> AMI first local minimum should be around period / 4 = 12.5 (5 to 15)
    t = np.arange(1000)
    v = np.sin(2 * np.pi * t / 50.0)
    tau = compute_ami(v, max_lag=50)
    assert 5 <= tau <= 15

def test_fnn_lorenz_attractor():
    # Lorenz attractor has fractal dimension ~2.06, requiring d >= 3 embedding
    xs, _ys, _zs = generate_lorenz_fixture(n_points=3000)
    tau = compute_ami(xs, max_lag=30)
    d = compute_fnn(xs, tau=tau, max_d=10, r_tol=R_TOL, a_tol=A_TOL)
    assert 3 <= d <= 6

def test_delay_embed_shape():
    v = np.arange(100, dtype=float)
    tau = 2
    d = 4
    X = delay_embed(v, tau=tau, d=d)
    assert X.shape == (94, 4)


def test_latest_delay_vector_only_needs_latest_history():
    v = np.arange(200, dtype=float)
    tau = 2
    d = 8
    max_lag = (d - 1) * tau

    full_latest = delay_embed(v, tau=tau, d=d)[-1]
    tail_latest = delay_embed(v[-(max_lag + 1):], tau=tau, d=d)[-1]

    np.testing.assert_array_equal(full_latest, tail_latest)

def test_johnson_lindenstrauss_projection():
    np.random.seed(42)
    X = np.random.randn(100, 12)
    Z = compress_projection(X, target_d=D_TARGET, seed=42)
    assert Z.shape == (100, D_TARGET)
    
    p1, p2 = X[0], X[1]
    dist_X = np.linalg.norm(p1 - p2)
    dist_Z = np.linalg.norm(Z[0] - Z[1])
    assert 0.5 <= (dist_Z / dist_X) <= 2.0

def test_padding_projection():
    X = np.ones((50, 4))
    Z = compress_projection(X, target_d=D_TARGET)
    assert Z.shape == (50, D_TARGET)
    assert np.all(Z[:, :4] == 1.0)
    assert np.all(Z[:, 4:] == 0.0)

def test_hnsw_knn_query():
    np.random.seed(42)
    dim = 8
    hnsw = HNSWIndex(dim=dim, max_elements=1000)
    data = np.random.randn(100, dim)
    hnsw.add_items(data)
    
    indices, distances = hnsw.knn_query(data[0], k=3)
    assert indices[0] == 0
    assert distances[0] < 1e-5

def test_online_mad_normalize():
    window = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])
    Z_t = np.array([3.0, 4.0])
    Z_norm = online_mad_normalize(Z_t, window)
    assert Z_norm.shape == Z_t.shape
    assert not np.isnan(Z_norm).any()
