"""TSB-AD benchmark loaders (NeurIPS 2024)."""

from __future__ import annotations

import csv
import hashlib
import re
import shutil
import zipfile
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from tsad.benchmarks.base import BenchmarkDataset, BenchmarkSeries


DATASET_URLS = {
    "TSB-AD-U": "https://www.thedatum.org/datasets/TSB-AD-U.zip",
    "TSB-AD-M": "https://www.thedatum.org/datasets/TSB-AD-M.zip",
}

SPLIT_LIST_URL = (
    "https://raw.githubusercontent.com/TheDatumOrg/TSB-AD/main/Datasets/File_List"
)


def _download(url: str, dest: Path) -> None:
    """Download ``url`` to ``dest`` using the standard library."""
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, dest.open("wb") as out:
        shutil.copyfileobj(response, out)


def _parse_filename(name: str) -> dict[str, Any]:
    """Parse a TSB-AD CSV filename into metadata fields.

    Example: 001_NAB_id_1_Facility_tr_1007_1st_2014.csv
    """
    pattern = re.compile(
        r"^(?P<index>\d+)_(?P<dataset>[A-Za-z0-9_]+?)_id_(?P<id>\d+)_"
        r"(?P<domain>[A-Za-z]+)_tr_(?P<tr>\d+)_1st_(?P<first_anomaly>\d+)\.csv$"
    )
    match = pattern.match(name)
    if not match:
        raise ValueError(f"cannot parse TSB-AD filename: {name}")
    return {
        "index": int(match.group("index")),
        "source_dataset": match.group("dataset"),
        "id": int(match.group("id")),
        "domain": match.group("domain"),
        "train_split": int(match.group("tr")),
        "first_anomaly": int(match.group("first_anomaly")),
    }


class TSB_AD_U(BenchmarkDataset):
    """TSB-AD univariate benchmark (NeurIPS 2024)."""

    name = "TSB-AD-U"

    def __init__(self, data_root: str | Path | None = None) -> None:
        super().__init__(data_root)
        self.csv_dir = self.dataset_dir / "TSB-AD-U"
        self.split_dir = self.data_root / "TSB-AD-splits"

    @property
    def is_downloaded(self) -> bool:
        return self.csv_dir.is_dir() and any(self.csv_dir.glob("*.csv"))

    def download(self, *, force: bool = False) -> None:
        if self.is_downloaded and not force:
            return

        zip_path = self.dataset_dir / "TSB-AD-U.zip"
        if not zip_path.exists() or force:
            _download(DATASET_URLS[self.name], zip_path)

        if not self.csv_dir.is_dir() or force:
            with zipfile.ZipFile(zip_path, "r") as archive:
                archive.extractall(self.dataset_dir)

        # Download official split lists.
        for split in ["Eva", "Tuning"]:
            filename = f"{self.name}-{split}.csv"
            split_path = self.split_dir / filename
            if not split_path.exists() or force:
                self.split_dir.mkdir(parents=True, exist_ok=True)
                _download(f"{SPLIT_LIST_URL}/{filename}", split_path)

    def _split_files(self, split: str) -> set[str]:
        """Return the set of filenames for the requested split."""
        split_csv = self.split_dir / f"{self.name}-{split}.csv"
        if not split_csv.exists():
            raise FileNotFoundError(f"split list not found: {split_csv}")
        files: set[str] = set()
        with split_csv.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                files.add(row["file_name"].strip())
        return files

    def iter_series(self, split: str) -> Iterator[BenchmarkSeries]:
        if split == "train":
            # TSB-AD uses a tuning split for model selection.
            return self._iter_from_list("Tuning")
        if split == "eval":
            return self._iter_from_list("Eva")
        raise ValueError(f"unknown split: {split!r}")

    def _iter_from_list(self, list_name: str) -> Iterator[BenchmarkSeries]:
        files = self._split_files(list_name)
        for name in sorted(files):
            path = self.csv_dir / name
            if not path.exists():
                raise FileNotFoundError(f"expected TSB-AD series missing: {path}")
            yield self._load_series(path)

    def _load_series(self, path: Path) -> BenchmarkSeries:
        import pandas as pd

        metadata = _parse_filename(path.name)
        df = pd.read_csv(path)
        signal = np.asarray(df["Data"].values, dtype=np.float64)
        labels = np.asarray(df["Label"].values, dtype=np.int8)

        # Enforce binary labels and finite signal.
        if not np.isfinite(signal).all():
            finite = np.isfinite(signal)
            signal = np.where(finite, signal, np.nan)
            # Forward-fill NaNs as a last-resort data-cleaning step.
            mask = np.isnan(signal)
            if mask.any():
                signal[mask] = np.interp(
                    np.where(mask)[0], np.where(~mask)[0], signal[~mask]
                )
        labels = (labels != 0).astype(np.int8)

        return BenchmarkSeries(
            name=path.name,
            signal=signal,
            labels=labels,
            train_split=metadata["train_split"],
            metadata=metadata,
            provenance={
                "file": str(path),
                "sha256": _sha256_file(path),
                "source": "TSB-AD (NeurIPS 2024)",
                "url": DATASET_URLS[self.name],
                "license": "Apache-2.0",
                "label_semantics": "binary: 1 = anomalous, 0 = normal",
            },
        )

    def provenance(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source": "TSB-AD (NeurIPS 2024 D&B Track)",
            "url": "https://thedatumorg.github.io/TSB-AD/",
            "data_url": DATASET_URLS[self.name],
            "split_list_url": SPLIT_LIST_URL,
            "n_series_total": 870,
            "n_eval_series": 350,
            "n_tuning_series": 48,
            "primary_metric": "VUS-PR",
            "license": "Apache-2.0",
        }


class TSB_AD_M(BenchmarkDataset):
    """TSB-AD multivariate benchmark (NeurIPS 2024).

    The project is currently univariate, but the loader is included for
    completeness and for future multivariate extensions.
    """

    name = "TSB-AD-M"

    def __init__(self, data_root: str | Path | None = None) -> None:
        super().__init__(data_root)
        self.csv_dir = self.dataset_dir / "TSB-AD-M"
        self.split_dir = self.data_root / "TSB-AD-splits"

    @property
    def is_downloaded(self) -> bool:
        return self.csv_dir.is_dir() and any(self.csv_dir.glob("*.csv"))

    def download(self, *, force: bool = False) -> None:
        if self.is_downloaded and not force:
            return

        zip_path = self.dataset_dir / "TSB-AD-M.zip"
        if not zip_path.exists() or force:
            _download(DATASET_URLS[self.name], zip_path)

        if not self.csv_dir.is_dir() or force:
            with zipfile.ZipFile(zip_path, "r") as archive:
                archive.extractall(self.dataset_dir)

        for split in ["Eva", "Tuning"]:
            filename = f"{self.name}-{split}.csv"
            split_path = self.split_dir / filename
            if not split_path.exists() or force:
                self.split_dir.mkdir(parents=True, exist_ok=True)
                _download(f"{SPLIT_LIST_URL}/{filename}", split_path)

    def _split_files(self, split: str) -> set[str]:
        split_csv = self.split_dir / f"{self.name}-{split}.csv"
        if not split_csv.exists():
            raise FileNotFoundError(f"split list not found: {split_csv}")
        files: set[str] = set()
        with split_csv.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                files.add(row["file_name"].strip())
        return files

    def iter_series(self, split: str) -> Iterator[BenchmarkSeries]:
        if split == "train":
            return self._iter_from_list("Tuning")
        if split == "eval":
            return self._iter_from_list("Eva")
        raise ValueError(f"unknown split: {split!r}")

    def _iter_from_list(self, list_name: str) -> Iterator[BenchmarkSeries]:
        files = self._split_files(list_name)
        for name in sorted(files):
            path = self.csv_dir / name
            if not path.exists():
                raise FileNotFoundError(f"expected TSB-AD series missing: {path}")
            yield self._load_series(path)

    def _load_series(self, path: Path) -> BenchmarkSeries:
        import pandas as pd

        metadata = _parse_filename(path.name)
        df = pd.read_csv(path)
        # Multivariate files have columns Data_0, Data_1, ..., Data_n, Label.
        data_cols = [c for c in df.columns if c.startswith("Data")]
        if not data_cols:
            # Fallback: assume the last column is the label and the rest are data.
            data_cols = list(df.columns[:-1])

        signal = np.asarray(df[data_cols].values, dtype=np.float64)
        labels = np.asarray(df["Label"].values, dtype=np.int8)
        labels = (labels != 0).astype(np.int8)

        return BenchmarkSeries(
            name=path.name,
            signal=signal,  # type: ignore[arg-type]
            labels=labels,
            train_split=metadata["train_split"],
            metadata={**metadata, "n_variates": signal.shape[1]},
            provenance={
                "file": str(path),
                "sha256": _sha256_file(path),
                "source": "TSB-AD (NeurIPS 2024)",
                "url": DATASET_URLS[self.name],
                "license": "Apache-2.0",
                "label_semantics": "binary: 1 = anomalous, 0 = normal",
            },
        )

    def provenance(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source": "TSB-AD (NeurIPS 2024 D&B Track)",
            "url": "https://thedatumorg.github.io/TSB-AD/",
            "data_url": DATASET_URLS[self.name],
            "split_list_url": SPLIT_LIST_URL,
            "n_series_total": 200,
            "n_eval_series": 180,
            "n_tuning_series": 20,
            "primary_metric": "VUS-PR",
            "license": "Apache-2.0",
            "notes": "Project pipeline is univariate; multivariate evaluation requires a wrapper.",
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
