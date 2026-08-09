import hashlib
import json

import numpy as np
import pytest

from tsad.datasets import load_npz_dataset


def _write_dataset(tmp_path, *, synthetic=False, checksum=None):
    path = tmp_path / "sample.npz"
    np.savez_compressed(
        path,
        signal=np.linspace(0.0, 1.0, 10),
        labels=np.array([0, 0, 0, 1, 1, 0, 0, 0, 0, 0], dtype=np.int8),
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "name": "sample",
        "source": "synthetic-fixture" if synthetic else "test-source",
        "source_url": "https://example.invalid/dataset",
        "license": "test",
        "label_semantics": "binary anomaly labels",
        "synthetic": synthetic,
        "sha256": checksum or digest,
    }
    path.with_suffix(".json").write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_loader_requires_provenance_manifest(tmp_path):
    path = tmp_path / "missing_manifest.npz"
    np.savez(path, signal=np.zeros(3), labels=np.zeros(3))

    with pytest.raises(ValueError, match="provenance manifest"):
        load_npz_dataset(path)


def test_loader_rejects_synthetic_data_by_default(tmp_path):
    path = _write_dataset(tmp_path, synthetic=True)

    with pytest.raises(ValueError, match="synthetic"):
        load_npz_dataset(path)

    signal, labels, manifest = load_npz_dataset(path, allow_synthetic=True)
    assert signal.shape == labels.shape
    assert manifest["synthetic"] is True


def test_loader_accepts_verified_real_manifest(tmp_path):
    path = _write_dataset(tmp_path)

    signal, labels, manifest = load_npz_dataset(path)

    assert len(signal) == len(labels) == 10
    assert set(np.unique(labels)) <= {0, 1}
    assert manifest["source"] == "test-source"


def test_loader_rejects_checksum_mismatch(tmp_path):
    path = _write_dataset(tmp_path, checksum="0" * 64)

    with pytest.raises(ValueError, match="checksum"):
        load_npz_dataset(path)


def test_loader_accepts_continuous_float32_labels(tmp_path):
    path = tmp_path / "gradual.npz"
    signal = np.linspace(0.0, 1.0, 10, dtype=np.float64)
    labels = np.linspace(0.0, 1.0, 10, dtype=np.float32)
    np.savez_compressed(path, signal=signal, labels=labels)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "name": "gradual",
        "source": "test-source",
        "source_url": "https://example.invalid/dataset",
        "license": "test",
        "label_semantics": "gradual anomaly labels in [0, 1]",
        "synthetic": False,
        "sha256": digest,
    }
    path.with_suffix(".json").write_text(json.dumps(manifest), encoding="utf-8")

    sig, lbl, _ = load_npz_dataset(path)
    assert sig.dtype == np.float64
    assert lbl.dtype == np.float32
    assert np.all((lbl >= 0.0) & (lbl <= 1.0))

