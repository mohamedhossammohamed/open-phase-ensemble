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
    label_dtype: str = "int8",
):
    path.parent.mkdir(parents=True, exist_ok=True)
    if label_dtype == "float32":
        npz_labels = labels.astype(np.float32)
    else:
        npz_labels = labels.astype(np.int8)
    np.savez_compressed(path, signal=signal.astype(np.float64), labels=npz_labels)
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
            label_dtype="int8",
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


def _stft_crossfade(
    healthy_win: np.ndarray,
    faulty_win: np.ndarray,
    nperseg: int = 256,
    noverlap: int = 192,
) -> np.ndarray:
    """Linearly interpolate STFT magnitude from healthy to faulty preserving healthy phase."""
    from scipy.signal import istft, stft

    target_len = len(healthy_win)
    if target_len < nperseg:
        weights = np.linspace(0.0, 1.0, target_len, dtype=np.float64)
        return (1.0 - weights) * healthy_win + weights * faulty_win

    _, _, z_healthy = stft(healthy_win, nperseg=nperseg, noverlap=noverlap)
    _, _, z_faulty = stft(faulty_win, nperseg=nperseg, noverlap=noverlap)

    n_frames = z_healthy.shape[1]
    w = np.linspace(0.0, 1.0, n_frames, dtype=np.float64)

    mag_healthy = np.abs(z_healthy)
    mag_faulty = np.abs(z_faulty)

    mag_interp = (1.0 - w) * mag_healthy + w * mag_faulty
    phase_healthy = np.angle(z_healthy)
    z_interp = mag_interp * np.exp(1j * phase_healthy)

    _, x_rec = istft(z_interp, nperseg=nperseg, noverlap=noverlap)

    if len(x_rec) > target_len:
        x_rec = x_rec[:target_len]
    elif len(x_rec) < target_len:
        x_rec = np.pad(x_rec, (0, target_len - len(x_rec)), mode="edge")

    return x_rec


def prepare_cwru_data(
    healthy_mat: str | Path,
    faulty_mat: str | Path,
    output_path: str | Path | None = None,
    transition_samples: int = 4096,
    location: str = "bearing",
    severity: str = "0.007",
    load_baseline: str = "0hp",
    load_fault: str = "0hp",
) -> Path:
    """Synthesize a physically defensible STFT cross-fade CWRU streaming fault-onset benchmark."""
    healthy_path = Path(healthy_mat)
    faulty_path = Path(faulty_mat)
    healthy = _extract_cwru_signal(healthy_path)
    faulty = _extract_cwru_signal(faulty_path)

    if len(healthy) <= transition_samples:
        raise ValueError(
            f"Healthy signal length ({len(healthy)}) must be > transition_samples ({transition_samples})"
        )
    if len(faulty) <= transition_samples:
        raise ValueError(
            f"Faulty signal length ({len(faulty)}) must be > transition_samples ({transition_samples})"
        )

    healthy_pre = healthy[:-transition_samples]
    healthy_win = healthy[-transition_samples:]
    faulty_win = faulty[:transition_samples]
    faulty_post = faulty[transition_samples:]

    transition_signal = _stft_crossfade(healthy_win, faulty_win)

    signal = np.concatenate([healthy_pre, transition_signal, faulty_post])

    labels_pre = np.zeros(len(healthy_pre), dtype=np.float32)
    labels_trans = np.linspace(0.0, 1.0, len(transition_signal), dtype=np.float32)
    labels_post = np.ones(len(faulty_post), dtype=np.float32)
    labels = np.concatenate([labels_pre, labels_trans, labels_post])

    if output_path is None:
        filename = f"cwru_{location}_{severity}_{load_baseline}to{load_fault}.npz"
        output_path = RAW_DIR / "cwru" / filename
    else:
        output_path = Path(output_path)

    _save_npz_with_manifest(
        output_path,
        signal,
        labels,
        name=f"CWRU {location} {severity} transition ({load_baseline} -> {load_fault})",
        source="Case Western Reserve University Bearing Dataset via documented mirror",
        source_url=(
            "https://engineering.case.edu/bearingdatacenter/download-data-file; "
            "https://github.com/s-whynot/CWRU-dataset"
        ),
        label_semantics=(
            "healthy baseline labeled 0.0; STFT cross-fade transition ramping linearly "
            "from 0.0 to 1.0; faulty segment labeled 1.0. Warmup/fit data should be sourced "
            "from a distinct baseline recording rather than this test stream."
        ),
        label_dtype="float32",
    )
    return output_path


def _parse_cwru_mat_metadata(filename: str) -> dict[str, str] | None:
    """Parse fault location, severity, and load condition from raw CWRU filename."""
    stem = Path(filename).stem
    if stem.startswith("._"):
        return None

    if "Normal" in stem or "normal" in stem:
        parts = stem.split("_")
        load_code = parts[-1] if parts[-1].isdigit() else "0"
        return {
            "type": "healthy",
            "load": f"{load_code}hp",
        }

    location = None
    if "IR" in stem or "Inner" in stem:
        location = "inner"
    elif "OR" in stem or "Outer" in stem:
        location = "outer"
    elif "B" in stem or "Ball" in stem:
        location = "ball"

    if location is None:
        return None

    severity = None
    if "007" in stem:
        severity = "0.007"
    elif "014" in stem:
        severity = "0.014"
    elif "021" in stem:
        severity = "0.021"
    elif "028" in stem:
        severity = "0.028"

    if severity is None:
        severity = "unknown"

    parts = stem.split("_")
    load_code = parts[-1] if parts[-1].isdigit() else "0"

    return {
        "type": "faulty",
        "location": location,
        "severity": severity,
        "load": f"{load_code}hp",
    }


def prepare_all_cwru_benchmarks(
    cwru_dir: str | Path | None = None,
    transition_samples: int = 4096,
    load_baseline: str | None = None,
    load_fault: str | None = None,
) -> list[Path]:
    """Scan raw CWRU directory and generate benchmarks for all available conditions."""
    cwru_dir = Path(cwru_dir) if cwru_dir else RAW_DIR / "cwru"
    if not cwru_dir.exists():
        print(f"CWRU directory not found: {cwru_dir}")
        return []

    mat_files = [f for f in cwru_dir.glob("*.mat") if not f.name.startswith("._")]
    if not mat_files:
        print(f"No .mat files found in {cwru_dir}")
        return []

    healthy_files = {}
    faulty_files = []

    for mat_path in mat_files:
        meta = _parse_cwru_mat_metadata(mat_path.name)
        if not meta:
            continue
        if meta["type"] == "healthy":
            healthy_files[meta["load"]] = mat_path
        elif meta["type"] == "faulty":
            faulty_files.append((mat_path, meta))

    if not healthy_files:
        print("No healthy baseline .mat files found.")
        return []

    generated = []
    expected_locations = ["inner", "outer", "ball"]
    expected_severities = ["0.007", "0.014", "0.021"]
    found_combos = set()

    for faulty_path, meta in faulty_files:
        loc = meta["location"]
        sev = meta["severity"]
        f_load = meta["load"]
        found_combos.add((loc, sev))

        b_load = load_baseline if load_baseline else (f_load if load_fault is None else load_baseline)
        if b_load not in healthy_files:
            b_load = next(iter(healthy_files.keys()))

        h_path = healthy_files[b_load]
        out_load_f = load_fault if load_fault else f_load

        out_path = cwru_dir / f"cwru_{loc}_{sev}_{b_load}to{out_load_f}.npz"
        res = prepare_cwru_data(
            h_path,
            faulty_path,
            output_path=out_path,
            transition_samples=transition_samples,
            location=loc,
            severity=sev,
            load_baseline=b_load,
            load_fault=out_load_f,
        )
        generated.append(res)

    missing = []
    for loc in expected_locations:
        for sev in expected_severities:
            if (loc, sev) not in found_combos:
                missing.append(f"{loc} ({sev}\")")

    if missing:
        print(f"Notice: Raw CWRU data missing locally for expected combinations: {', '.join(missing)}")
        print(f"Generated benchmark files for {len(generated)} available condition(s).")

    return generated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physionet-record", action="append", dest="physionet_records")
    parser.add_argument("--cwru-healthy", type=Path)
    parser.add_argument("--cwru-faulty", type=Path)
    parser.add_argument(
        "--transition-samples",
        type=int,
        default=4096,
        help="STFT cross-fade transition window length in samples (default: 4096)",
    )
    parser.add_argument(
        "--load-baseline",
        type=str,
        default=None,
        help="HP load condition for healthy baseline (e.g. '0hp')",
    )
    parser.add_argument(
        "--load-fault",
        type=str,
        default=None,
        help="HP load condition for fault segment (e.g. '2hp')",
    )
    parser.add_argument(
        "--cwru-all",
        action="store_true",
        help="Generate benchmark streams for all raw CWRU MAT files present",
    )
    args = parser.parse_args()

    if args.physionet_records:
        download_physionet_data(args.physionet_records)
    if args.cwru_healthy or args.cwru_faulty:
        if not args.cwru_healthy or not args.cwru_faulty:
            parser.error("--cwru-healthy and --cwru-faulty must be provided together")
        prepare_cwru_data(
            args.cwru_healthy,
            args.cwru_faulty,
            transition_samples=args.transition_samples,
            load_baseline=args.load_baseline or "0hp",
            load_fault=args.load_fault or "0hp",
        )
    elif args.cwru_all:
        prepare_all_cwru_benchmarks(
            transition_samples=args.transition_samples,
            load_baseline=args.load_baseline,
            load_fault=args.load_fault,
        )
    elif not args.physionet_records:
        download_physionet_data()
        prepare_all_cwru_benchmarks(
            transition_samples=args.transition_samples,
            load_baseline=args.load_baseline,
            load_fault=args.load_fault,
        )


if __name__ == "__main__":
    main()
