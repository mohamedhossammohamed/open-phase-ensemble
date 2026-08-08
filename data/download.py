"""Download and normalize real benchmark data with explicit provenance."""

# isort: skip_file

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from tsad.datasets import write_manifest


DATA_DIR = Path(__file__).resolve().parent
RAW_DIR = DATA_DIR / "raw"
PHYSIONET_RECORDS = ["100", "101", "102", "103", "105", "106", "109", "112"]
PHYSIONET_URL = "https://physionet.org/content/mitdb/1.0.0/"
ABNORMAL_BEAT_SYMBOLS = {"A", "a", "J", "S", "V", "E", "F", "/", "f", "Q", "?"}


def _save_npz_with_manifest(
    path: Path,
    signal: np.ndarray,
    labels: np.ndarray,
    *,
    name: str,
    source: str,
    source_url: str,
    label_semantics: str,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, signal=signal.astype(np.float64), labels=labels.astype(np.int8))
    write_manifest(
        path,
        name=name,
        source=source,
        source_url=source_url,
        license_name="See upstream dataset terms",
        label_semantics=label_semantics,
        synthetic=False,
    )


def download_physionet_data(records: list[str] | None = None):
    """Download MIT-BIH records and convert beat annotations into binary labels."""
    try:
        import wfdb
    except ImportError as exc:
        raise RuntimeError("install wfdb to download PhysioNet data") from exc

    records = records or PHYSIONET_RECORDS
    physio_dir = RAW_DIR / "physionet"
    physio_dir.mkdir(parents=True, exist_ok=True)

    for record_name in records:
        record_base = physio_dir / record_name
        if not (record_base.with_suffix(".dat")).exists():
            wfdb.dl_database("mitdb", str(physio_dir), records=[record_name], keep_subdirs=False)

        record = wfdb.rdrecord(str(record_base))
        annotations = wfdb.rdann(str(record_base), "atr")
        signal = np.asarray(record.p_signal[:, 0], dtype=np.float64)
        labels = np.zeros(len(signal), dtype=np.int8)
        for sample, symbol in zip(annotations.sample, annotations.symbol):
            if symbol in ABNORMAL_BEAT_SYMBOLS and 0 <= sample < len(labels):
                labels[sample] = 1

        _save_npz_with_manifest(
            physio_dir / f"{record_name}.npz",
            signal,
            labels,
            name=f"MIT-BIH Arrhythmia record {record_name}",
            source="PhysioNet MIT-BIH Arrhythmia Database v1.0.0",
            source_url=PHYSIONET_URL,
            label_semantics="abnormal MIT-BIH beat annotations at annotated sample indices",
        )


def _extract_cwru_signal(mat_path: Path) -> np.ndarray:
    try:
        from scipy.io import loadmat
    except ImportError as exc:
        raise RuntimeError("install scipy to prepare CWRU MAT files") from exc

    data = loadmat(mat_path)
    candidates = []
    for key, value in data.items():
        if key.startswith("__"):
            continue
        array = np.asarray(value).squeeze()
        if array.ndim == 1 and array.size > 100:
            priority = 0 if key.endswith("_DE_time") else 1
            candidates.append((priority, -array.size, key, array))
    if not candidates:
        raise ValueError(f"no one-dimensional vibration signal found in {mat_path}")
    candidates.sort(key=lambda item: item[:3])
    return np.asarray(candidates[0][3], dtype=np.float64)


def prepare_cwru_data(healthy_mat: str | Path, faulty_mat: str | Path):
    """Create a transparent real-signal transition benchmark from two CWRU MAT files.

    The healthy record is labeled 0 and the fault record is labeled 1. This is a
    documented proxy protocol, not a claim that CWRU supplies point-level anomaly
    timestamps inside a single recording.
    """
    healthy_path = Path(healthy_mat)
    faulty_path = Path(faulty_mat)
    healthy = _extract_cwru_signal(healthy_path)
    faulty = _extract_cwru_signal(faulty_path)
    signal = np.concatenate([healthy, faulty])
    labels = np.concatenate([
        np.zeros(len(healthy), dtype=np.int8),
        np.ones(len(faulty), dtype=np.int8),
    ])
    output_path = RAW_DIR / "cwru" / "cwru_bearing.npz"
    _save_npz_with_manifest(
        output_path,
        signal,
        labels,
        name="CWRU Bearing real healthy-to-fault transition proxy",
        source="Case Western Reserve University Bearing Dataset via documented mirror",
        source_url=(
            "https://engineering.case.edu/bearingdatacenter/download-data-file; "
            "https://github.com/s-whynot/CWRU-dataset"
        ),
        label_semantics="healthy source record labeled 0; faulty source record labeled 1",
    )
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physionet-record", action="append", dest="physionet_records")
    parser.add_argument("--cwru-healthy", type=Path)
    parser.add_argument("--cwru-faulty", type=Path)
    args = parser.parse_args()

    if args.physionet_records:
        download_physionet_data(args.physionet_records)
    if args.cwru_healthy or args.cwru_faulty:
        if not args.cwru_healthy or not args.cwru_faulty:
            parser.error("--cwru-healthy and --cwru-faulty must be provided together")
        prepare_cwru_data(args.cwru_healthy, args.cwru_faulty)
    if not args.physionet_records and not args.cwru_healthy:
        parser.error(
            "no dataset requested; use --physionet-record 100 or provide real CWRU MAT files"
        )


if __name__ == "__main__":
    main()
