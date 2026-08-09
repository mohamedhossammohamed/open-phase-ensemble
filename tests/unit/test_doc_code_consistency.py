"""Regression tests that fail if documentation and code drift apart again.

Each test targets one item from the doc/code reconciliation.  These tests are
intentionally brittle on doc strings and source-inspection so that a future
edit reintroducing a mismatch is caught immediately.
"""

from __future__ import annotations

import inspect
import os
from pathlib import Path

import numpy as np
import pytest

from tsad.config import FNN_THRESHOLD, HEDGE_ETA
from tsad.learning_loop import OnlineLearningLoop
from tsad.meta_judge import MetaJudge
from tsad.representation import compute_fnn

ROOT = Path(__file__).resolve().parents[2]


# ── Item 1: loss is Pearson, not Spearman ──────────────────────────────

def test_item1_learning_loop_uses_pearson_not_spearman():
    """learning_loop.py must use np.corrcoef (Pearson), not scipy.stats.spearmanr."""
    src = inspect.getsource(OnlineLearningLoop)
    assert "np.corrcoef" in src, "learning_loop must use np.corrcoef (Pearson)"
    assert "spearmanr" not in src, "learning_loop must not use spearmanr"
    # README must say Pearson, not Spearman as the loss
    readme = (ROOT / "README.md").read_text()
    assert "Pearson correlation loss" in readme
    assert "Spearman Rank Loss" not in readme
    method = (ROOT / "method.html").read_text()
    assert "PearsonCorr" in method
    assert "SpearmanCorr" not in method


# ── Item 2: fixed Hedge, not AdaHedge ──────────────────────────────────

def test_item2_fixed_hedge_not_adahedge():
    """meta_judge.py must use a fixed eta, not an adaptive AdaHedge schedule."""
    src = inspect.getsource(MetaJudge)
    assert "self.eta" in src
    assert "eta_t" not in src, "meta_judge must not implement adaptive eta_t"
    assert "mixability" not in src.lower()
    # Docs must not claim AdaHedge (allow "not AdaHedge" disclaimers)
    readme = (ROOT / "README.md").read_text()
    for line in readme.splitlines():
        if "AdaHedge" in line and "not" not in line.lower() and "no" not in line.lower():
            pytest.fail(f"README still claims AdaHedge without disclaimer: {line.strip()}")
    method = (ROOT / "method.html").read_text()
    for line in method.splitlines():
        if "AdaHedge" in line and "not" not in line.lower() and "no" not in line.lower():
            pytest.fail(f"method.html still claims AdaHedge without disclaimer: {line.strip()}")
    assert "Fixed-rate Hedge" in method or "fixed" in method.lower()


# ── Item 3: Simplex Projection, not S-Map ──────────────────────────────

def test_item3_simplex_not_smap():
    """detectors/simplex.py must implement Simplex, not S-Map."""
    from tsad.detectors.simplex import SimplexProjectionDetector
    src = inspect.getsource(SimplexProjectionDetector)
    assert "theta" not in src.lower(), "simplex detector must not have a theta/S-Map parameter"
    assert "S-Map" not in src
    # Docs must not claim S-Map
    readme = (ROOT / "README.md").read_text()
    assert "S-Map" not in readme or "not S-Map" in readme
    method = (ROOT / "method.html").read_text()
    # Allow "not S-Map" clarifications
    for line in method.splitlines():
        if "S-Map" in line and "not S-Map" not in line and "no" not in line.lower():
            pytest.fail(f"method.html still claims S-Map without disclaimer: {line.strip()}")


# ── Item 4: batch ridge refit, not RLS ─────────────────────────────────

def test_item4_batch_ridge_not_rls():
    """detectors/sarima.py must use batch OLS+ridge, not online RLS."""
    from tsad.detectors.sarima import ARFilterDetector
    src = inspect.getsource(ARFilterDetector)
    assert "_fit_ar" in src, "AR detector must have a batch refit method"
    assert "0.99" not in src, "AR detector must not claim RLS forgetting factor 0.99"
    # Docs must not claim RLS or "Online AR"
    readme = (ROOT / "README.md").read_text()
    assert "RLS" not in readme or "not online RLS" in readme
    # "Online AR" must not appear as a description (only "Batch AR" is correct)
    for line in readme.splitlines():
        if "Online AR" in line and "not" not in line.lower():
            pytest.fail(f"README still says 'Online AR' without disclaimer: {line.strip()}")
    method = (ROOT / "method.html").read_text()
    for line in method.splitlines():
        if "RLS" in line and "not" not in line.lower() and "no" not in line.lower():
            pytest.fail(f"method.html still claims RLS without disclaimer: {line.strip()}")
    # results.html audit section must not say "online AR ridge filter"
    results = (ROOT / "results.html").read_text()
    for line in results.splitlines():
        if "online AR" in line.lower() and "not" not in line.lower():
            pytest.fail(f"results.html still says 'online AR' without disclaimer: {line.strip()}")


# ── Item 5: block buffer, not EWMA ─────────────────────────────────────

def test_item5_block_buffer_not_ewma():
    """detectors/mahalanobis.py must use a block buffer, not EWMA."""
    from tsad.detectors.mahalanobis import RobustMahalanobisDetector
    src = inspect.getsource(RobustMahalanobisDetector)
    assert "block_size" in src, "Mahalanobis must use a block buffer"
    assert "0.005" not in src, "Mahalanobis must not claim EWMA alpha=0.005"
    assert "ewma" not in src.lower()
    # Docs must not claim EWMA
    readme = (ROOT / "README.md").read_text()
    assert "EWMA" not in readme or "not EWMA" in readme
    method = (ROOT / "method.html").read_text()
    for line in method.splitlines():
        if "EWMA" in line and "not" not in line.lower() and "no" not in line.lower():
            pytest.fail(f"method.html still claims EWMA without disclaimer: {line.strip()}")


# ── Item 6: no stumpy dependency, single-window raw Euclidean ───────────

def test_item6_no_stumpy_dependency():
    """pyproject.toml must not list stumpy; matrix_profile.py must not import it."""
    pyproject = (ROOT / "pyproject.toml").read_text()
    assert "stumpy" not in pyproject, "stumpy must be removed from pyproject.toml dependencies"
    mp_src = inspect.getsource(__import__("tsad.detectors.matrix_profile", fromlist=["MatrixProfileDetector"]).MatrixProfileDetector)
    assert "import stumpy" not in mp_src, "matrix_profile.py must not import stumpy"
    assert "sliding_window_view" in mp_src, "matrix_profile must use sliding_window_view"
    # Docs must not claim STOMP/STUMPY
    readme = (ROOT / "README.md").read_text()
    assert "STOMP" not in readme or "not STUMPY/STOMP" in readme
    method = (ROOT / "method.html").read_text()
    for line in method.splitlines():
        if "STOMP" in line and "not" not in line.lower() and "no" not in line.lower():
            pytest.fail(f"method.html still claims STOMP without disclaimer: {line.strip()}")
    results = (ROOT / "results.html").read_text()
    # Allow "not STUMPY" disclaimers but not STUMPY listed as a used library
    for line in results.splitlines():
        if "STUMPY" in line and "not" not in line.lower() and "no" not in line.lower():
            pytest.fail(f"results.html still lists STUMPY without disclaimer: {line.strip()}")


# ── Item 7: FNN_THRESHOLD wired from config ────────────────────────────

def test_item7_fnn_threshold_wired_from_config():
    """compute_fnn must import and use FNN_THRESHOLD from config, not hardcode 0.05."""
    src = inspect.getsource(compute_fnn)
    assert "FNN_THRESHOLD" in src, "compute_fnn must reference FNN_THRESHOLD"
    # The old hardcoded 0.05 must not appear (excluding the config import line)
    assert "0.05" not in src, "compute_fnn must not hardcode 0.05"
    assert FNN_THRESHOLD == 0.01, f"FNN_THRESHOLD must be 0.01, got {FNN_THRESHOLD}"
    # Verify the import exists in representation.py
    rep_src = (ROOT / "src/tsad/representation.py").read_text()
    assert "FNN_THRESHOLD" in rep_src, "representation.py must import FNN_THRESHOLD"


# ── Item 8: AMI/FNN are utilities, not wired into pipeline ─────────────

def test_item8_ami_fnn_not_wired_into_pipeline():
    """TSADPipeline must not call compute_ami/compute_fnn; docs must say they're utilities."""
    from tsad.pipeline import TSADPipeline
    src = inspect.getsource(TSADPipeline)
    assert "compute_ami" not in src, "pipeline must not call compute_ami"
    assert "compute_fnn" not in src, "pipeline must not call compute_fnn"
    # Docs must acknowledge this
    readme = (ROOT / "README.md").read_text()
    assert "available utilities" in readme or "not currently wired" in readme.lower()
    method = (ROOT / "method.html").read_text()
    assert "not currently wired" in method.lower() or "available utilities" in method.lower()


# ── Item 9: IAAFT only, no AR surrogate ────────────────────────────────

def test_item9_iaaft_only_no_ar_surrogate():
    """method.html must not claim AR surrogates; only IAAFT is implemented."""
    method = (ROOT / "method.html").read_text()
    assert "AR Surrogate" not in method, "method.html must not claim AR surrogates"
    assert "Autoregressive (AR) Surrogates" not in method
    # Verify no AR surrogate generator exists in the evaluation module
    iaaft_src = (ROOT / "src/tsad/evaluation/iaaft.py").read_text()
    assert "generate_ar_surrogate" not in iaaft_src
    protocol_src = (ROOT / "src/tsad/evaluation/protocol.py").read_text()
    assert "ar_surrogate" not in protocol_src.lower()


# ── Item 10: CITATION.cff uses current detector names ──────────────────

def test_item10_citation_uses_current_names():
    """CITATION.cff must use ARFilterDetector/MSETransformerAutoencoder, not SARIMA/AnomalyTransformer."""
    citation = (ROOT / "CITATION.cff").read_text()
    assert "SARIMA" not in citation, "CITATION.cff must not use retired name 'SARIMA'"
    assert "Anomaly Transformer" not in citation, "CITATION.cff must not use retired name 'Anomaly Transformer'"
    assert "AR Linear Ridge Filter" in citation
    assert "MSE Transformer Autoencoder" in citation


# ── Item 11: index.html says six modules ───────────────────────────────

def test_item11_index_html_says_six_modules():
    """index.html must say six modules, not five."""
    index = (ROOT / "index.html").read_text()
    assert "six modular" in index.lower() or "six modular" in index
    assert "five modular" not in index.lower()


# ── Item 12: nan_to_num guards present in pipeline ─────────────────────

def test_item12_pipeline_has_nan_guards():
    """pipeline.py must have nan_to_num guards on scores, forecasts, and fused output."""
    src = (ROOT / "src/tsad/pipeline.py").read_text()
    assert "np.nan_to_num" in src, "pipeline.py must have nan_to_num guards"
    # Must guard scores, forecasts, A_t, and v_hat_star
    assert "nan_to_num(s_k" in src or "nan_to_num(s_k" in src
    assert "nan_to_num(v_hat_k" in src or "nan_to_num(v_hat_k" in src
    assert "nan_to_num(A_t" in src
    assert "nan_to_num(v_hat_star" in src
    # Docs must mention NaN safety
    method = (ROOT / "method.html").read_text()
    assert "NaN" in method or "nan" in method.lower()


# ── Item 13: verify_checklist.py is executable, not print-only ─────────

def test_item13_verify_checklist_is_executable():
    """verify_checklist.py must contain real assertions, not just print('PASSED')."""
    script = (ROOT / "scripts/verify_checklist.py").read_text()
    assert "def check_" in script, "verify_checklist must define check functions"
    assert "_fail(" in script, "verify_checklist must have a failure path"
    assert "SystemExit(1)" in script, "verify_checklist must exit non-zero on failure"
    # Must import and use actual modules
    assert "from tsad" in script, "verify_checklist must import tsad modules"
    assert "TSADPipeline" in script
    assert "compute_vus_roc" in script
    assert "CUSUMGating" in script
    # Must not be the old print-only stub
    assert script.count('print("PASSED")') == 0 or "_ok(" in script
