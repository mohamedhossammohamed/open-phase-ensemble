import sys
from pathlib import Path
import numpy as np
import pytest

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(DATA_DIR))

from download import _stft_crossfade, prepare_cwru_data
from tsad.datasets import load_npz_dataset


def test_stft_crossfade_preserves_length_and_finiteness():
    t = np.linspace(0, 1, 1024)
    h_win = np.sin(2 * np.pi * 10 * t)
    f_win = np.sin(2 * np.pi * 50 * t)

    res = _stft_crossfade(h_win, f_win, nperseg=256, noverlap=192)

    assert len(res) == len(h_win)
    assert np.isfinite(res).all()


def test_prepare_cwru_data_synthesizes_valid_benchmark(tmp_path):
    from scipy.io import savemat

    h_mat = tmp_path / "97_Normal_0.mat"
    f_mat = tmp_path / "282_B007_0.mat"

    t = np.linspace(0, 10, 8192)
    h_signal = np.sin(2 * np.pi * 5 * t)
    f_signal = np.sin(2 * np.pi * 30 * t) + np.random.normal(0, 0.5, len(t))

    savemat(str(h_mat), {"X097_DE_time": h_signal})
    savemat(str(f_mat), {"X282_DE_time": f_signal})

    out_npz = tmp_path / "cwru_ball_0.007_0hpto0hp.npz"

    prepare_cwru_data(
        h_mat,
        f_mat,
        output_path=out_npz,
        transition_samples=1024,
        location="ball",
        severity="0.007",
        load_baseline="0hp",
        load_fault="0hp",
    )

    assert out_npz.exists()
    assert out_npz.with_suffix(".json").exists()

    signal, labels, manifest = load_npz_dataset(out_npz)

    assert len(signal) == len(labels)
    assert signal.dtype == np.float64
    assert labels.dtype == np.float32

    assert labels[0] == 0.0
    assert labels[-1] == 1.0
    transition_labels = labels[labels > 0.0]
    assert (transition_labels < 1.0).any()
    assert "healthy baseline labeled 0.0" in manifest["label_semantics"]
