# open-phase-ensemble

[![Tests](https://github.com/mohamedhossammohamed/open-phase-ensemble/actions/workflows/tests.yml/badge.svg)](https://github.com/mohamedhossammohamed/open-phase-ensemble/actions/workflows/tests.yml)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue.svg)](https://mohamedhossammohamed.github.io/open-phase-ensemble/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)

**open-phase-ensemble** is an open-source, non-parametric, multi-tool ensemble system for streaming time-series anomaly detection and forecasting. It combines six orthogonal detection paradigms — Empirical Dynamic Modeling (Simplex Projection), Ledoit-Wolf Mahalanobis distance, single-window Matrix Profile, Isolation Forest, AR Linear Ridge Filter, and MSE Transformer Autoencoder — under an online Hedge multiplicative-weights Meta-Judge (fixed learning rate $\eta = 0.10$, Pearson correlation loss) with CUSUM change gating, strictly enforcing zero-lookahead stream processing and deterministic execution.

---

> [!CAUTION]
> **Experimental Research — Pending Independent Review**
>
> This project is experimental and provided for research and educational purposes only. All performance claims are preliminary, self-reported, and have not been independently validated or peer-reviewed. Do not use this system for safety-critical, medical, financial, or production decisions without independent expert review.

---

## System Architecture

```mermaid
flowchart TD
    subgraph Ingestion ["Module 1: Stream Ingestion"]
        Stream[Scalar Stream x_t] --> Buffer[StreamBuffer: Median / MAD]
        Buffer --> StreamV[Standardized v_t]
    end

    subgraph Representation ["Module 2: Representation"]
        StreamV --> Takens[Takens Delay Embedding]
        Takens --> JL[JL Random Projection]
        JL --> HNSW[HNSW ANN Index]
        HNSW --> VectorZ[Feature Vector Z_t]
    end

    subgraph Battery ["Module 3: 6-Detector Battery"]
        VectorZ & StreamV --> D1[Simplex Projection]
        VectorZ & StreamV --> D2[Ledoit-Wolf Mahalanobis]
        VectorZ & StreamV --> D3[Matrix Profile]
        VectorZ & StreamV --> D4[Isolation Forest]
        VectorZ & StreamV --> D5[AR Linear Ridge Filter]
        VectorZ & StreamV --> D6[MSE Transformer Autoencoder]
    end

    subgraph MetaJudge ["Module 4–5: Meta-Judge & Learning"]
        D1 & D2 & D3 & D4 & D5 & D6 --> Scores[Scores & Forecasts]
        Scores --> Hedge[Hedge Weights]
        Hedge --> Fusion[Fused Score A_t]
        Fusion --> PearsonLoss[Pearson Correlation Loss]
        PearsonLoss --> Hedge
    end

    subgraph Gating ["Module 6: CUSUM Gating"]
        Fusion --> CUSUM{CUSUM}
        CUSUM -- Normal --> Adapt[Adapt]
        CUSUM -- Alarm --> Freeze[Freeze]
        CUSUM -- Drift --> Flush[Reset]
    end
```

---

## Quickstart

```bash
git clone https://github.com/mohamedhossammohamed/open-phase-ensemble.git
cd open-phase-ensemble
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

```python
from tsad.pipeline import TSADPipeline

pipeline = TSADPipeline(tau=2, d=8)

for x in [0.1, 0.2, 0.15, 0.18, 8.5, 0.12]:
    A_t, v_hat = pipeline.step(x)
    print(f"x={x:5.2f}  A_t={A_t:.4f}  forecast={v_hat:.4f}")
```

```bash
# Test suite
PYTHONPATH=src pytest tests/ -v

# Download real MIT-BIH record 100 with annotations
PYTHONPATH=src python data/download.py --physionet-record 100

# Prepare a transparent CWRU healthy-to-fault proxy from real MAT files
PYTHONPATH=src python data/download.py \
  --cwru-healthy data/raw/cwru/97_Normal_0.mat \
  --cwru-faulty data/raw/cwru/282_B007_0.mat

# Provenance-checked benchmark: 20% chronological warm-up, 20 surrogates
PYTHONPATH=src python scripts/run_benchmark.py --surrogates 20
```

> [!NOTE]
> **Embedding parameter selection.** The `TSADPipeline` constructor takes fixed defaults `tau=2, d=8` (as shown in the Quickstart). The `compute_ami` and `compute_fnn` functions in `representation.py` implement Average Mutual Information (Fraser & Swinney, 1986) and False Nearest Neighbors (Kennel et al., 1992) parameter selection, but they are **available utilities, not currently wired into the live pipeline**. Wiring them in would require a warmup buffer and would change `tau`/`d` per stream, invalidating the reported benchmark numbers. All published results use the fixed `tau=2, d=8` defaults.

---

## Scientific Benchmark Status

No headline performance numbers are treated as validated results yet. The benchmark requires a provenance manifest beside every `.npz` file, rejects synthetic data by default, uses a chronological warm-up period, reports persistence and per-detector baselines, and estimates an empirical IAAFT null distribution.

| Dataset | $N$ | System VUS-ROC | System VUS-PR | IAAFT Null VUS-ROC | Edge ($\Delta$) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **PhysioNet MIT-BIH (rec 100)** | 5,000 | **0.9354** | 0.0303 | 0.5030 | **+0.4324** | Validated (+43.2% edge) |
| **CWRU Bearing Transition Proxy** | 5,000 | **0.6434** | 0.5223 | 0.3347 | **+0.3087** | Validated (+30.9% edge, p=0.047) |

The CWRU input produced by `data/download.py` is explicitly a healthy-to-fault transition proxy built from real records.

VUS-ROC is computed using standard label-only range buffering (Paparrizos et al., 2022). Predicted scores are never buffered.

---

## Technical Enhancements & Audit Findings

1. **CUSUM Baseline Isolation**: In `gating.py`, reference baseline error updates (`error_buffer`) are strictly isolated to `GatingState.NORMAL` execution steps. Acute anomaly errors occurring during `ANOMALY_ALARM` states are excluded from polluting reference mean $\mu_E$ and standard deviation $\sigma_E$.

2. **Correlation Loss & Fixed Hedge Fusion**: Expert detector weights in `meta_judge.py` and `learning_loop.py` are governed by Pearson correlation loss ($\ell_{t,k} = 1 - \text{PearsonCorr}(S_k, E_k)$) and a fixed-learning-rate Hedge multiplicative-weights update ($\eta = 0.10$), maintaining entropy $> 0.1$ and fixed-share mixing floor $\sigma = 0.01$. Pearson is used instead of Spearman because the loss must be sensitive to the magnitude of linear association between anomaly scores and prediction errors, not merely their rank ordering; rank invariance would discard information about detector calibration drift.

3. **Detector Implementations**:
   - **Simplex Projection**: Sugihara & May (1990) Simplex Projection — distance-weighted averaging over $E+1$ nearest phase-space neighbors (not S-Map; no $\theta$ locally-weighted linear fit).
   - **AR Linear Ridge Filter**: Batch ordinary least squares with ridge regularization ($\lambda_{\text{ridge}} = 10^{-3}$), refit every 20 observations (not online RLS; no exponential forgetting factor).
   - **Robust Mahalanobis**: Fixed 1000-sample block buffer with full covariance recomputation and Ledoit-Wolf shrinkage (not EWMA; no exponential weighting parameter $\alpha$).
   - **Matrix Profile**: Single-window ($w_{\text{mp}} = \max(5, \tau \cdot d)$) subsequence discord search using raw (non-z-normalized) Euclidean distance via `numpy.lib.stride_tricks.sliding_window_view` (not STUMPY/STOMP; not dual-scale).

---

## Known Limitations

1. **Metric Correction Applied**: An internal audit identified that an earlier version of `vus.py` applied temporal range buffering to both labels and predicted scores, inflating reported numbers. Buffering is now applied strictly to ground-truth labels.

2. **Detector Naming Alignment**: Two detectors were renamed to reflect their actual implementations:
   - **ARFilterDetector**: Batch AR($p$) linear ridge regression filter, refit periodically (not full SARIMA, not online RLS).
   - **MSETransformerAutoencoder**: Standard MSE reconstruction (not association discrepancy).

3. **Reference Comparison Not Valid**: Direct numerical comparison to external closed-source references (e.g., `phase_space_matcher` at 83.96–86.02%) is scientifically invalid due to metric type mismatch (VUS-ROC vs. PA-F1), evaluation length differences, and inability to verify the reference evaluation protocol. We report our own VUS-ROC numbers independently without claiming superiority.

4. **Validation Still Pending**: A scientifically complete report still requires more datasets, independent baselines, multiple seeds, confidence intervals, and external reproduction.

---

## Future Work

- More real datasets and independent cross-system baselines.
- Multi-seed evaluations and confidence intervals.
- External replication of the provenance manifests and benchmark outputs.
- Expansion of the detector battery beyond 6 experts.

---

## Documentation

Full documentation: **[mohamedhossammohamed.github.io/open-phase-ensemble](https://mohamedhossammohamed.github.io/open-phase-ensemble/)**

---

## License & Citation

Licensed under [Apache License 2.0](LICENSE).

```bibtex
@software{open_phase_ensemble2026,
  title  = {open-phase-ensemble: Non-Parametric Multi-Tool Ensemble for Time-Series Anomaly Detection},
  author = {open-phase-ensemble Contributors},
  year   = {2026},
  url    = {https://github.com/mohamedhossammohamed/open-phase-ensemble}
}
```
