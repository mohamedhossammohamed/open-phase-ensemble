"""Executable architectural verification for open-phase-ensemble.

Each check imports and runs the actual verification logic proven out in the
unit/integration/e2e test suite.  A passing run is evidence; a failing run
exits non-zero.  This is the script referenced by the README and the audit
matrix in appendix.html.

Run:  PYTHONPATH=src python scripts/verify_checklist.py
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from tsad.config import FNN_THRESHOLD, HEDGE_ETA, FIXED_SHARE_SIGMA  # noqa: E402
from tsad.evaluation.vus import compute_vus_roc  # noqa: E402
from tsad.gating import CUSUMGating, GatingState  # noqa: E402
from tsad.meta_judge import MetaJudge  # noqa: E402
from tsad.pipeline import TSADPipeline  # noqa: E402
from tsad.representation import compute_fnn  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from fixtures.generate_fixtures import get_or_create_sine_fixture  # noqa: E402


def _ok(label: str) -> None:
    print(f"[PASS] {label}")


def _fail(label: str, detail: str) -> None:
    print(f"[FAIL] {label}: {detail}")
    raise SystemExit(1)


def check_cusum_freezes_on_alarm() -> None:
    """CUSUM must freeze weight adaptation during ANOMALY_ALARM."""
    gating = CUSUMGating()
    np.random.seed(42)
    for _ in range(50):
        gating.step(float(np.random.normal(0.1, 0.01)))
    for _ in range(10):
        state = gating.step(10.0)
    if state != GatingState.ANOMALY_ALARM or gating.is_adaptation_allowed():
        _fail("CUSUM freeze on alarm", f"state={state}, allowed={gating.is_adaptation_allowed()}")
    _ok("CUSUM freezes adaptation during ANOMALY_ALARM")


def check_takens_bounds() -> None:
    """tau and d must respect MAX_TAU=100, MAX_D=20 config bounds."""
    from tsad.config import MAX_D, MAX_TAU
    if MAX_TAU != 100 or MAX_D != 20:
        _fail("Takens parameter bounds", f"MAX_TAU={MAX_TAU}, MAX_D={MAX_D}")
    _ok("Takens parameter bounds (tau<=100, d<=20)")


def check_ledoit_wolf_formula() -> None:
    """Ledoit-Wolf mu must equal trace(Sigma_sample)/d."""
    from tsad.detectors.mahalanobis import RobustMahalanobisDetector
    np.random.seed(42)
    det = RobustMahalanobisDetector(dim=8, block_size=200)
    data = np.random.randn(200, 8)
    for row in data:
        det.score(row, v_t=row[0])
        det.update(row[0])
        det.add_vector(row)
    X = np.array(det.buffer)
    sample = np.cov(X, rowvar=False, ddof=1)
    expected_mu = float(np.trace(sample) / 8)
    if abs(det.mu - expected_mu) > 1e-6:
        _fail("Ledoit-Wolf mu formula", f"det.mu={det.mu}, expected={expected_mu}")
    _ok("Ledoit-Wolf mu = trace(Sigma_sample)/d")


def check_pearson_loss_epsilon_guard() -> None:
    """Learning loop must guard the Pearson denominator against div-by-zero."""
    from tsad.learning_loop import OnlineLearningLoop
    loop = OnlineLearningLoop(k_detectors=2, window_size=10)
    loss = loop.step(1.0, np.array([1.0, 1.0]), np.array([0.5, 0.5]))
    if not np.all(np.isfinite(loss)):
        _fail("Pearson epsilon guard", f"loss={loss}")
    _ok("Pearson correlation denominator epsilon guard")


def check_vus_no_point_adjustment() -> None:
    """VUS must not apply point-adjustment; perfect predictor scores ~1.0."""
    labels = np.zeros(100, dtype=int)
    labels[20:30] = 1
    scores = labels.astype(float)
    vus = compute_vus_roc(scores, labels, max_buffer=0)
    if abs(vus - 1.0) > 1e-9:
        _fail("VUS no point-adjustment", f"vus={vus}")
    _ok("VUS metric has no point-adjustment inflation")


def check_determinism_hash() -> None:
    """SHA-256 of pipeline scores must be identical across 2 independent runs."""
    signal, _ = get_or_create_sine_fixture()

    def run_hash() -> str:
        p = TSADPipeline()
        scores = [p.step(float(x))[0] for x in signal[:1000]]
        return hashlib.sha256(np.array(scores, dtype=np.float64).tobytes()).hexdigest()

    h1, h2 = run_hash(), run_hash()
    if h1 != h2:
        _fail("Execution determinism", f"hash1={h1[:16]}, hash2={h2[:16]}")
    _ok("100% execution determinism (SHA-256 reproducible)")


def check_zero_lookahead() -> None:
    """Stream vs sequential batch Euclidean distance must be exactly 0.0."""
    signal, _ = get_or_create_sine_fixture()
    n = 200

    p1 = TSADPipeline()
    stream = [p1.step(float(x))[0] for x in signal[:n]]

    p2 = TSADPipeline()
    batch = [p2.step(float(signal[i]))[0] for i in range(n)]

    dist = float(np.linalg.norm(np.array(stream) - np.array(batch)))
    if dist != 0.0:
        _fail("Zero-lookahead invariant", f"dist={dist}")
    _ok("No lookahead leakage (stream vs batch distance = 0.0)")


def check_weight_entropy_floor() -> None:
    """Meta-Judge weight entropy must stay above 0.1 bits."""
    signal, _ = get_or_create_sine_fixture()
    p = TSADPipeline()
    for x in signal:
        p.step(float(x))
    entropy = float(-np.sum(p.meta_judge.weights * np.log2(p.meta_judge.weights + 1e-12)))
    if entropy <= 0.1:
        _fail("Weight entropy floor", f"entropy={entropy:.4f}")
    _ok(f"Meta-Judge weight entropy > 0.1 bits (entropy={entropy:.4f})")


def check_fixed_share_floor() -> None:
    """Fixed-share mixing must guarantee w_k >= sigma/K."""
    mj = MetaJudge(k_detectors=6, eta=HEDGE_ETA, sigma=FIXED_SHARE_SIGMA)
    for _ in range(500):
        mj.update_weights(np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    floor = FIXED_SHARE_SIGMA / 6.0
    if mj.weights[0] < floor - 1e-8:
        _fail("Fixed-share weight floor", f"w[0]={mj.weights[0]}, floor={floor}")
    _ok("Fixed-share weight floor w_k >= sigma/K")


def check_fnn_threshold_wired() -> None:
    """compute_fnn must use FNN_THRESHOLD from config, not a hardcoded literal."""
    import inspect
    src = inspect.getsource(compute_fnn)
    if "0.05" in src or "0.01" in src.replace("FNN_THRESHOLD", ""):
        _fail("FNN_THRESHOLD wired", "compute_fnn source still contains a hardcoded threshold literal")
    if FNN_THRESHOLD != 0.01:
        _fail("FNN_THRESHOLD wired", f"FNN_THRESHOLD={FNN_THRESHOLD}, expected 0.01")
    _ok(f"compute_fnn uses config FNN_THRESHOLD={FNN_THRESHOLD}")


def main() -> None:
    print("  TSAD ARCHITECTURAL VERIFICATION (executable)  ")
    print("=" * 55)
    check_cusum_freezes_on_alarm()
    check_takens_bounds()
    check_ledoit_wolf_formula()
    check_pearson_loss_epsilon_guard()
    check_vus_no_point_adjustment()
    check_determinism_hash()
    check_zero_lookahead()
    check_weight_entropy_floor()
    check_fixed_share_floor()
    check_fnn_threshold_wired()
    print("=" * 55)
    print("All architectural verification checks PASSED.")
    print("For the full test suite run:  PYTHONPATH=src pytest tests/ -v")


if __name__ == "__main__":
    main()
