"""Provenance-aware dataset loading for benchmark inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(
    path: str | Path,
    *,
    name: str,
    source: str,
    source_url: str,
    license_name: str,
    label_semantics: str,
    synthetic: bool = False,
) -> Path:
    """Write a checksum-bearing provenance sidecar next to an NPZ dataset."""
    dataset_path = Path(path)
    manifest_path = dataset_path.with_suffix(".json")
    manifest = {
        "name": name,
        "source": source,
        "source_url": source_url,
        "license": license_name,
        "label_semantics": label_semantics,
        "synthetic": synthetic,
        "sha256": _sha256(dataset_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def load_npz_dataset(
    path: str | Path,
    *,
    allow_synthetic: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load a benchmark stream only when its provenance is explicit and valid."""
    dataset_path = Path(path)
    manifest_path = dataset_path.with_suffix(".json")
    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset file does not exist: {dataset_path}")
    if not manifest_path.exists():
        raise ValueError(f"provenance manifest is required: {manifest_path}")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid provenance manifest: {manifest_path}") from exc

    required = {"name", "source", "source_url", "license", "label_semantics", "synthetic", "sha256"}
    missing = sorted(required - manifest.keys())
    if missing:
        raise ValueError(f"provenance manifest missing fields: {', '.join(missing)}")
    if manifest["synthetic"] and not allow_synthetic:
        raise ValueError(
            f"synthetic dataset '{manifest['name']}' is disabled for benchmark evaluation"
        )
    if manifest["sha256"] != _sha256(dataset_path):
        raise ValueError(f"dataset checksum does not match provenance manifest: {dataset_path}")

    with np.load(dataset_path) as data:
        if "signal" not in data or "labels" not in data:
            raise ValueError("dataset must contain 'signal' and 'labels' arrays")
        signal = np.asarray(data["signal"], dtype=np.float64)
        labels = np.asarray(data["labels"], dtype=np.int8)

    if signal.ndim != 1 or labels.ndim != 1 or len(signal) != len(labels):
        raise ValueError("signal and labels must be one-dimensional arrays of equal length")
    if len(signal) == 0 or not np.isfinite(signal).all():
        raise ValueError("signal must be non-empty and finite")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("labels must be binary values in {0, 1}")

    return signal, labels, manifest
