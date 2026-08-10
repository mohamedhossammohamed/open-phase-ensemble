"""Historical benchmark loaders: TSB-UAD, NAB, UCR Anomaly Archive, Yahoo S5.

These benchmarks predate TSB-AD and are included for backwards comparability
and longitudinal analysis.  They are largely subsumed by TSB-AD but remain
useful for sanity-checking against older published results.
"""

from __future__ import annotations

import csv
import hashlib
import shutil
import zipfile
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from tsad.benchmarks.base import BenchmarkDataset, BenchmarkSeries
from tsad.benchmarks.tsb_ad import TSB_AD_M, TSB_AD_U


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, dest: Path) -> None:
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, dest.open("wb") as out:
        shutil.copyfileobj(response, out)


# ---------------------------------------------------------------------------
# TSB-UAD (PVLDB 2022) — the predecessor to TSB-AD
# ---------------------------------------------------------------------------

TSB_UAD_URLS = {
    "real": "https://www.thedatum.org/datasets/TSB-UAD-Public.zip",
    "synthetic": "https://www.thedatum.org/datasets/TSB-UAD-Synthetic.zip",
    "artificial": "https://www.thedatum.org/datasets/TSB-UAD-Artificial.zip",
}


class TSB_UAD(BenchmarkDataset):
    """TSB-UAD univariate benchmark (PVLDB 2022).

    Contains 12,686 time series across 18 source datasets.  There is no
    official train/eval split, so we use the first 20% of each series as
    a warm-up prefix (consistent with the project's streaming protocol).
    """

    name = "TSB-UAD"

    def __init__(
        self,
        data_root: str | Path | None = None,
        subset: str = "real",
    ) -> None:
        super().__init__(data_root)
        if subset not in TSB_UAD_URLS:
            raise ValueError(f"subset must be one of {list(TSB_UAD_URLS)}, got {subset!r}")
        self.subset = subset
        self.csv_dir = self.dataset_dir / f"TSB-UAD-{subset.capitalize()}"

    @property
    def is_downloaded(self) -> bool:
        return self.csv_dir.is_dir() and any(self.csv_dir.glob("**/*.csv"))

    def download(self, *, force: bool = False) -> None:
        if self.is_downloaded and not force:
            return

        zip_path = self.dataset_dir / f"TSB-UAD-{self.subset.capitalize()}.zip"
        if not zip_path.exists() or force:
            _download(TSB_UAD_URLS[self.subset], zip_path)

        if not self.csv_dir.is_dir() or force:
            with zipfile.ZipFile(zip_path, "r") as archive:
                archive.extractall(self.dataset_dir)

    def iter_series(self, split: str) -> Iterator[BenchmarkSeries]:
        if split == "train":
            return  # No official train split
        if split != "eval":
            raise ValueError(f"unknown split: {split!r}")
        return self._iter_all()

    def _iter_all(self) -> Iterator[BenchmarkSeries]:
        for path in sorted(self.csv_dir.rglob("*.csv")):
            yield self._load_series(path)

    def _load_series(self, path: Path) -> BenchmarkSeries:
        import pandas as pd

        df = pd.read_csv(path)
        # TSB-UAD files have varying column names; try common ones.
        data_col = None
        label_col = None
        for candidate_data in ["Data", "data", "value", "Value", "0"]:
            if candidate_data in df.columns:
                data_col = candidate_data
                break
        for candidate_label in ["Label", "label", "is_anomaly", "anomaly"]:
            if candidate_label in df.columns:
                label_col = candidate_label
                break
        if data_col is None or label_col is None:
            # Fallback: first column is data, last is label.
            data_col = df.columns[0]
            label_col = df.columns[-1]

        signal = np.asarray(df[data_col].values, dtype=np.float64)
        labels = (np.asarray(df[label_col].values) != 0).astype(np.int8)

        # No official train split; use 20% warm-up prefix.
        train_split = max(1, int(len(signal) * 0.2))

        return BenchmarkSeries(
            name=path.name,
            signal=signal,
            labels=labels,
            train_split=train_split,
            metadata={"source_subset": self.subset, "n_points": len(signal)},
            provenance={
                "file": str(path),
                "sha256": _sha256_file(path),
                "source": "TSB-UAD (PVLDB 2022)",
                "url": TSB_UAD_URLS[self.subset],
                "license": "MIT",
                "label_semantics": "binary: 1 = anomalous, 0 = normal",
            },
        )

    def provenance(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source": "TSB-UAD (PVLDB 2022)",
            "url": "https://github.com/TheDatumOrg/TSB-UAD",
            "data_url": TSB_UAD_URLS[self.subset],
            "subset": self.subset,
            "n_series_total": 12686,
            "primary_metric": "VUS-PR",
            "license": "MIT",
            "notes": "No official train/eval split; 20% warm-up prefix used.",
        }


# ---------------------------------------------------------------------------
# NAB (Numenta Anomaly Benchmark)
# ---------------------------------------------------------------------------

NAB_URL = "https://raw.githubusercontent.com/numenta/NAB/master/data/"
NAB_LABELS_URL = "https://raw.githubusercontent.com/numenta/NAB/master/labels/combined_labels.json"


class NAB(BenchmarkDataset):
    """Numenta Anomaly Benchmark (2015).

    58 labeled real-world and artificial time series.  Historically
    significant but largely superseded by TSB-AD (NAB is a subset of
    TSB-AD-U).  No official train/eval split; 20% warm-up prefix is used.

    NAB data files contain only (timestamp, value) — labels are stored
    separately in a combined_labels.json file.  This loader downloads
    both and merges them.
    """

    name = "NAB"

    # Map of relative path -> filename within the NAB data directory.
    NAB_FILES = [
        "artificialWithAnomaly/art_daily_flatmiddle.csv",
        "artificialWithAnomaly/art_daily_jumpsup.csv",
        "artificialWithAnomaly/art_daily_jumpsdown.csv",
        "artificialWithAnomaly/art_daily_perfect_square_wave.csv",
        "artificialWithAnomaly/art_daily_small_noise.csv",
        "artificialWithAnomaly/art_increasing.csv",
        "artificialWithAnomaly/art_noise.csv",
        "artificialWithAnomaly/art_flatline.csv",
        "realAWSCloudwatch/ec2_cpu_utilization_5f5533.csv",
        "realAWSCloudwatch/ec2_cpu_utilization_ac7cdcb.csv",
        "realAWSCloudwatch/ec2_cpu_utilization_24ae8d.csv",
        "realAWSCloudwatch/ec2_cpu_utilization_53ea38.csv",
        "realAWSCloudwatch/ec2_cpu_utilization_77c1ca.csv",
        "realAWSCloudwatch/ec2_cpu_utilization_825cc2.csv",
        "realAWSCloudwatch/ec2_cpu_utilization_fe7f93.csv",
        "realAWSCloudwatch/ec2_network_in_5abac7.csv",
        "realAWSCloudwatch/ec2_network_in_257a54.csv",
        "realAWSCloudwatch/rds_cpu_utilization_e47b3b.csv",
        "realAWSCloudwatch/rds_cpu_utilization_cc0c53.csv",
        "realKnownCause/ambient_temperature_system_failure.csv",
        "realKnownCause/cpu_utilization_asg_misconfiguration.csv",
        "realKnownCause/ec2_request_latency_system_failure.csv",
        "realKnownCause/machine_temperature_system_failure.csv",
        "realKnownCause/nyc_taxi.csv",
        "realKnownCause/ambient_temperature_system_failure.csv",
        "realTraffic/TravelTime_387.csv",
        "realTraffic/TravelTime_451.csv",
        "realTraffic/occupancy_t4013.csv",
        "realTraffic/occupancy_t6004.csv",
        "realTraffic/speed_6007.csv",
        "realTraffic/speed_7578.csv",
        "realTweets/Twitter_volume_AMZN.csv",
        "realTweets/Twitter_volume_CVS.csv",
        "realTweets/Twitter_volume_FB.csv",
        "realTweets/Twitter_volume_GOOG.csv",
        "realTweets/Twitter_volume_IBM.csv",
        "realTweets/Twitter_volume_KO.csv",
        "realTweets/Twitter_volume_UPS.csv",
        "realAdExchange/cpc_per_day.csv",
        "realAdExchange/cpm_per_day.csv",
        "realRogueAccessPoint/speed_7578.csv",
    ]

    def __init__(self, data_root: str | Path | None = None) -> None:
        super().__init__(data_root)
        self.csv_dir = self.dataset_dir / "NAB"
        self._labels_cache: dict[str, list[str]] | None = None

    @property
    def is_downloaded(self) -> bool:
        return self.csv_dir.is_dir() and any(self.csv_dir.rglob("*.csv"))

    def _load_labels(self) -> dict[str, list[str]]:
        """Download and cache the NAB combined labels JSON."""
        if self._labels_cache is not None:
            return self._labels_cache

        labels_path = self.csv_dir / "combined_labels.json"
        if not labels_path.exists():
            self.csv_dir.mkdir(parents=True, exist_ok=True)
            _download(NAB_LABELS_URL, labels_path)

        import json
        with labels_path.open("r", encoding="utf-8") as f:
            self._labels_cache = json.load(f)
        return self._labels_cache

    def download(self, *, force: bool = False) -> None:
        if self.is_downloaded and not force:
            return
        self.csv_dir.mkdir(parents=True, exist_ok=True)
        # Download labels first
        self._load_labels()
        # Download data files, skipping any that 404
        import urllib.error
        for rel_path in self.NAB_FILES:
            dest = self.csv_dir / rel_path
            if dest.exists() and not force:
                continue
            url = f"{NAB_URL}{rel_path}"
            try:
                _download(url, dest)
            except urllib.error.HTTPError:
                continue  # Skip files that don't exist in the repo

    def iter_series(self, split: str) -> Iterator[BenchmarkSeries]:
        if split == "train":
            return
        if split != "eval":
            raise ValueError(f"unknown split: {split!r}")
        for path in sorted(self.csv_dir.rglob("*.csv")):
            yield self._load_series(path)

    def _load_series(self, path: Path) -> BenchmarkSeries:
        import pandas as pd

        df = pd.read_csv(path)
        # NAB files have: timestamp, value (no inline label)
        data_col = "value" if "value" in df.columns else df.columns[1]
        signal = np.asarray(df[data_col].values, dtype=np.float64)

        # Look up labels from the combined labels JSON
        labels = np.zeros(len(signal), dtype=np.int8)
        rel_name = str(path.relative_to(self.csv_dir))
        labels_map = self._load_labels()
        # NAB labels use paths like "realAWSCloudwatch/ec2_cpu_utilization_5f5533.csv"
        for nab_path, anomalous_windows in labels_map.items():
            if nab_path in rel_name or rel_name in nab_path:
                timestamps = pd.to_datetime(df["timestamp"])
                for window in anomalous_windows:
                    window_start = pd.to_datetime(window)
                    # Mark points within 1 hour of the labeled timestamp as anomalous
                    mask = (timestamps >= window_start) & (
                        timestamps <= window_start + pd.Timedelta(hours=1)
                    )
                    labels[mask.values] = 1
                break

        train_split = max(1, int(len(signal) * 0.2))

        return BenchmarkSeries(
            name=rel_name,
            signal=signal,
            labels=labels,
            train_split=train_split,
            metadata={"n_points": len(signal)},
            provenance={
                "file": str(path),
                "sha256": _sha256_file(path),
                "source": "NAB (Numenta, 2015)",
                "url": NAB_URL,
                "license": "AGPL-3.0",
                "label_semantics": "binary: 1 = anomalous, 0 = normal",
            },
        )

    def provenance(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source": "Numenta Anomaly Benchmark (2015)",
            "url": "https://github.com/numenta/NAB",
            "n_series_total": 58,
            "primary_metric": "VUS-PR",
            "license": "AGPL-3.0",
            "notes": "Historical benchmark; superseded by TSB-AD. NAB is a subset of TSB-AD-U.",
        }


# ---------------------------------------------------------------------------
# UCR Anomaly Archive
# ---------------------------------------------------------------------------

UCR_ANOMALY_URL = (
    "https://www.cs.ucr.edu/~eamonn/UCR_TimeSeriesAnomalyDatasets2021.zip"
)


class UCRAnomalyArchive(BenchmarkDataset):
    """UCR Anomaly Archive (2021).

    250 univariate time series from 250 UCR classification datasets,
    transformed to contain injected anomalies.  No official train/eval
    split; 20% warm-up prefix is used.
    """

    name = "UCR-Anomaly"

    def __init__(self, data_root: str | Path | None = None) -> None:
        super().__init__(data_root)
        self.csv_dir = self.dataset_dir / "UCR-Anomaly"

    @property
    def is_downloaded(self) -> bool:
        return self.csv_dir.is_dir() and any(self.csv_dir.rglob("*.txt"))

    def download(self, *, force: bool = False) -> None:
        if self.is_downloaded and not force:
            return
        zip_path = self.dataset_dir / "UCR_AnomalyDatasets2021.zip"
        if not zip_path.exists() or force:
            _download(UCR_ANOMALY_URL, zip_path)
        if not self.csv_dir.is_dir() or force:
            with zipfile.ZipFile(zip_path, "r") as archive:
                archive.extractall(self.dataset_dir)

    def iter_series(self, split: str) -> Iterator[BenchmarkSeries]:
        if split == "train":
            return
        if split != "eval":
            raise ValueError(f"unknown split: {split!r}")
        for path in sorted(self.csv_dir.rglob("*.txt")):
            yield self._load_series(path)

    def _load_series(self, path: Path) -> BenchmarkSeries:
        # UCR Anomaly files are space-delimited: label value value value ...
        # The first token is the anomaly label (0 or 1), the rest is the series.
        with path.open("r", encoding="utf-8") as handle:
            lines = handle.readlines()

        # Each line: "0 v1 v2 v3 ..." or "1 v1 v2 v3 ..."
        all_labels: list[int] = []
        all_values: list[float] = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            label = int(float(parts[0]))
            values = [float(v) for v in parts[1:]]
            all_labels.extend([label] * len(values))
            all_values.extend(values)

        signal = np.asarray(all_values, dtype=np.float64)
        labels = np.asarray(all_labels, dtype=np.int8)
        train_split = max(1, int(len(signal) * 0.2))

        return BenchmarkSeries(
            name=path.name,
            signal=signal,
            labels=labels,
            train_split=train_split,
            metadata={"n_points": len(signal)},
            provenance={
                "file": str(path),
                "sha256": _sha256_file(path),
                "source": "UCR Anomaly Archive (2021)",
                "url": UCR_ANOMALY_URL,
                "license": "See UCR license",
                "label_semantics": "binary: 1 = anomalous, 0 = normal",
            },
        )

    def provenance(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source": "UCR Anomaly Archive (2021)",
            "url": "https://www.cs.ucr.edu/~eamonn/UCR_TimeSeriesAnomalyDatasets2021.zip",
            "n_series_total": 250,
            "primary_metric": "VUS-PR",
            "license": "See UCR license terms",
            "notes": "Historical benchmark; now a subset of TSB-AD-U.",
        }


# ---------------------------------------------------------------------------
# Yahoo S5 (Webscope)
# ---------------------------------------------------------------------------

YAHOO_S5_URL = (
    "https://raw.githubusercontent.com/yahoo/egs/master/data/ydata-labeled-time-series-anomalies-v1_0/"
)


class YahooS5(BenchmarkDataset):
    """Yahoo S5 Anomaly Dataset (2015).

    367 time series (real + synthetic).  The full dataset requires Yahoo
    Webscope access, but a mirror is available on GitHub.  No official
    train/eval split; 20% warm-up prefix is used.
    """

    name = "Yahoo-S5"

    def __init__(self, data_root: str | Path | None = None) -> None:
        super().__init__(data_root)
        self.csv_dir = self.dataset_dir / "Yahoo-S5"

    @property
    def is_downloaded(self) -> bool:
        return self.csv_dir.is_dir() and any(self.csv_dir.rglob("*.csv"))

    def download(self, *, force: bool = False) -> None:
        if self.is_downloaded and not force:
            return
        # Yahoo S5 is not freely downloadable as a single archive.
        # Users must obtain it from Yahoo Webscope or a mirror.
        raise NotImplementedError(
            "Yahoo S5 must be downloaded manually from Yahoo Webscope "
            "(https://webscope.sandbox.yahoo.com/) and placed in "
            f"{self.csv_dir}. See README for instructions."
        )

    def iter_series(self, split: str) -> Iterator[BenchmarkSeries]:
        if split == "train":
            return
        if split != "eval":
            raise ValueError(f"unknown split: {split!r}")
        for path in sorted(self.csv_dir.rglob("*.csv")):
            yield self._load_series(path)

    def _load_series(self, path: Path) -> BenchmarkSeries:
        import pandas as pd

        df = pd.read_csv(path)
        # Yahoo S5 files vary; try common column names.
        data_col = None
        label_col = None
        for c in ["value", "Value", "data", "Data"]:
            if c in df.columns:
                data_col = c
                break
        for c in ["label", "Label", "is_anomaly", "anomaly"]:
            if c in df.columns:
                label_col = c
                break
        if data_col is None:
            data_col = df.columns[0]
        if label_col is None:
            label_col = df.columns[-1]

        signal = np.asarray(df[data_col].values, dtype=np.float64)
        labels = (np.asarray(df[label_col].values) != 0).astype(np.int8)
        train_split = max(1, int(len(signal) * 0.2))

        return BenchmarkSeries(
            name=path.name,
            signal=signal,
            labels=labels,
            train_split=train_split,
            metadata={"n_points": len(signal)},
            provenance={
                "file": str(path),
                "sha256": _sha256_file(path),
                "source": "Yahoo S5 (Webscope, 2015)",
                "url": "https://webscope.sandbox.yahoo.com/",
                "license": "Yahoo Webscope Terms of Use",
                "label_semantics": "binary: 1 = anomalous, 0 = normal",
            },
        )

    def provenance(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source": "Yahoo S5 (Webscope, 2015)",
            "url": "https://webscope.sandbox.yahoo.com/",
            "n_series_total": 367,
            "primary_metric": "VUS-PR",
            "license": "Yahoo Webscope Terms of Use",
            "notes": "Historical benchmark; now a subset of TSB-AD-U. Manual download required.",
        }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

BENCHMARK_REGISTRY: dict[str, type[BenchmarkDataset]] = {
    "TSB-AD-U": TSB_AD_U,
    "TSB-AD-M": TSB_AD_M,
    "TSB-UAD": TSB_UAD,
    "NAB": NAB,
    "UCR-Anomaly": UCRAnomalyArchive,
    "Yahoo-S5": YahooS5,
}


def list_benchmarks() -> list[str]:
    """Return the names of all registered benchmarks."""
    return sorted(BENCHMARK_REGISTRY.keys())


def get_benchmark(name: str, data_root: str | Path | None = None) -> BenchmarkDataset:
    """Instantiate a benchmark by name."""
    if name not in BENCHMARK_REGISTRY:
        raise ValueError(f"unknown benchmark: {name!r}. Available: {list_benchmarks()}")
    return BENCHMARK_REGISTRY[name](data_root=data_root)
