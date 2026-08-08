import numpy as np

from tsad.config import FIXED_SHARE_SIGMA, HEDGE_ETA
from tsad.learning_loop import OnlineLearningLoop
from tsad.meta_judge import MetaJudge, StratifiedReplayBuffer


def test_hedge_weights_sum_to_one():
    mj = MetaJudge(k_detectors=6, eta=HEDGE_ETA, sigma=FIXED_SHARE_SIGMA)
    assert np.isclose(np.sum(mj.weights), 1.0)
    
    # Simulate 100 updates with arbitrary loss
    np.random.seed(42)
    for _ in range(100):
        loss_vec = np.random.rand(6)
        mj.update_weights(loss_vec)
        assert np.isclose(np.sum(mj.weights), 1.0, atol=1e-7)

def test_fixed_share_weight_floor():
    mj = MetaJudge(k_detectors=6, eta=HEDGE_ETA, sigma=FIXED_SHARE_SIGMA)
    
    # Inflict severe penalty on detector 0
    for _ in range(500):
        loss_vec = np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        mj.update_weights(loss_vec)
        
    min_floor = FIXED_SHARE_SIGMA / 6.0
    assert mj.weights[0] >= min_floor - 1e-8

def test_meta_judge_fusion():
    mj = MetaJudge(k_detectors=3)
    mj.weights = np.array([0.5, 0.3, 0.2])
    
    scores = np.array([0.8, 0.4, 0.1])
    forecasts = np.array([10.0, 20.0, 30.0])
    
    A_t, v_hat_star = mj.fuse(scores, forecasts)
    # A_t = 0.5*0.8 + 0.3*0.4 + 0.2*0.1 = 0.4 + 0.12 + 0.02 = 0.54
    # v_hat = 0.5*10 + 0.3*20 + 0.2*30 = 5 + 6 + 6 = 17.0
    assert np.isclose(A_t, 0.54)
    assert np.isclose(v_hat_star, 17.0)

def test_learning_loop_pearson_loss():
    loop = OnlineLearningLoop(k_detectors=2, window_size=100)
    
    # Detector 0: perfect positive correlation between anomaly score & prediction error
    # Detector 1: no correlation (random)
    np.random.seed(42)
    for i in range(100):
        err_0 = float(i)
        err_1 = float(np.random.rand() * 100)
        
        # Anomaly score follows prediction error for detector 0
        s_0 = float(i) / 100.0
        s_1 = float(np.random.rand())
        
        loss_vec = loop.step(
            true_v=1.0,
            forecasts=np.array([1.0 - err_0, 1.0 - err_1]),
            scores=np.array([s_0, s_1])
        )
        
    # Detector 0 loss should be close to 0.0 (high correlation)
    # Detector 1 loss should be higher
    assert loss_vec[0] < loss_vec[1]

def test_stratified_replay_buffer():
    buf = StratifiedReplayBuffer(capacity=100, n_quantiles=5)
    np.random.seed(42)
    
    for i in range(500):
        A_t = np.random.rand()
        Z_t = np.ones(8) * float(i)
        buf.add(Z_t, A_t)
        
    assert len(buf) == 100
    sample = buf.sample(n=20)
    assert sample.shape == (20, 8)
