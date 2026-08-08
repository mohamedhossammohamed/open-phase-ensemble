# open-phase-ensemble

[![Tests](https://github.com/mohamedhossammohamed/open-phase-ensemble/actions/workflows/tests.yml/badge.svg)](https://github.com/mohamedhossammohamed/open-phase-ensemble/actions/workflows/tests.yml)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue.svg)](https://mohamedhossammohamed.github.io/open-phase-ensemble/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.placeholder-blue)](https://zenodo.org/)

**open-phase-ensemble** is an open-source, non-parametric, multi-tool ensemble system for streaming time-series anomaly detection and forecasting. Designed to openly reproduce and extend phase-space trajectory matching principles, it combines six orthogonal detection paradigms (Empirical Dynamic Modeling, Ledoit-Wolf Mahalanobis distance, STOMP Matrix Profile, Isolation Forest, SARIMA residuals, and Anomaly Transformer attention) under an online Hedge multiplicative-weights Meta-Judge and CUSUM change gating, strictly enforcing zero-lookahead stream processing and 100% execution determinism.

---

> [!CAUTION]
> ### ⚠️ EXPERIMENTAL DISCLAIMER & REVIEW STATUS
> **This project is experimental and provided for research and educational purposes only.** All performance claims are preliminary, self-reported, and have not yet been independently validated or peer-reviewed. This system must not be used for safety-critical, medical, financial, or production decisions without professional assessment and external review by qualified domain experts. Use at your own risk.

---

## 🏛️ System Architecture

```mermaid
flowchart TD
    subgraph Ingestion ["Module 1: Stream Ingestion & Preprocessing"]
        Stream[Scalar Stream x_t] --> Buffer[StreamBuffer: Median / MAD Standardization]
        Buffer --> StreamV[Standardized Scalar v_t]
    end

    subgraph Representation ["Module 2: Multidimensional Representation"]
        StreamV --> Takens[Takens Delay Embedding: AMI & FNN]
        Takens --> MatrixX[Phase-Space Matrix X_t]
        MatrixX --> JL[Johnson-Lindenstrauss Projection: D_target = 8]
        JL --> HNSW[HNSW ANN Graph Indexing]
        HNSW --> VectorZ[Compressed Feature Vector Z_t]
    end

    subgraph Battery ["Module 3: 6-Detector Battery"]
        VectorZ & StreamV --> D1[Detector 1: Simplex Projection EDM]
        VectorZ & StreamV --> D2[Detector 2: Ledoit-Wolf Mahalanobis]
        VectorZ & StreamV --> D3[Detector 3: STOMP Matrix Profile]
        VectorZ & StreamV --> D4[Detector 4: Isolation Forest]
        VectorZ & StreamV --> D5[Detector 5: SARIMA Residuals]
        VectorZ & StreamV --> D6[Detector 6: Anomaly Transformer]
    end

    subgraph MetaJudge ["Module 4 & 5: Meta-Judge & Online Learning Loop"]
        D1 & D2 & D3 & D4 & D5 & D6 --> Scores[Scores s_t & Forecasts v_hat]
        Scores --> Hedge[Hedge Multiplicative Weights w_t]
        Hedge --> Fusion[Fused Anomaly Score A_t & Forecast v_hat*]
        Fusion --> PearsonLoss[Label-free Pearson Correlation Loss]
        PearsonLoss --> Hedge
    end

    subgraph Gating ["Module 6: Adaptation Gating & Tuning"]
        Fusion --> CUSUM{CUSUM Change Detector}
        CUSUM -- Normal --> Adapt[Allow Weight Adaptation & Replay]
        CUSUM -- Anomaly Alarm --> Freeze[Freeze Weight Updates]
        CUSUM -- Concept Drift --> Flush[Reset Baseline & Flush Buffer]
    end
```

---

## ⚡ Quickstart

### Installation

```bash
# Clone repository
git clone https://github.com/mohamedhossammohamed/open-phase-ensemble.git
cd open-phase-ensemble

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install package with dependencies
pip install -e .
```

### Basic Python Usage

```python
from tsad.pipeline import TSADPipeline

# Initialize streaming pipeline (tau=2, d=8)
pipeline = TSADPipeline(tau=2, d=8)

# Process scalar observations online
data_stream = [0.1, 0.2, 0.15, 0.18, 8.5, 0.12]

for x in data_stream:
    A_t, v_hat_star = pipeline.step(x)
    print(f"Observation: {x:5.2f} -> Anomaly Score: {A_t:.4f}, Forecast: {v_hat_star:.4f}")
```

### Running Test Suite & Benchmarks

```bash
# Run full test suite (34/34 tests)
PYTHONPATH=src pytest tests/ -v

# Run live dataset benchmarks
PYTHONPATH=src python scripts/run_benchmark.py
```

---

## 📊 Summary Benchmark Table

> **Note**: All performance numbers are preliminary, self-reported, and pending independent external review.

| Domain / Dataset | Evaluated Points ($N$) | System VUS-ROC | IAAFT Null VUS-ROC | Predictive Edge ($\Delta$) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **PhysioNet MIT-BIH (`record_100`)** | 5,143 | **0.9563** | 0.4798 | **+0.4765** | Preliminary |
| **CWRU Bearing Prognostics** | 5,000 | **0.9808** | 0.4801 | **+0.5007** | Preliminary |

---

## 📚 Full Documentation

Visit our full documentation site at **[mohamedhossammohamed.github.io/open-phase-ensemble](https://mohamedhossammohamed.github.io/open-phase-ensemble/)** to explore:

- [The Journey](https://mohamedhossammohamed.github.io/open-phase-ensemble/journey/) — Motivation and background story
- [Theoretical Foundations](https://mohamedhossammohamed.github.io/open-phase-ensemble/theory/) — Deep dive into Takens' theorem, EDM, Ledoit-Wolf, STOMP, Hedge, and VUS metrics
- [Architecture & Specifications](https://mohamedhossammohamed.github.io/open-phase-ensemble/architecture/) — Data contracts, DAG modules, and zero-lookahead invariants
- [Developer Guide](https://mohamedhossammohamed.github.io/open-phase-ensemble/developer/) — How to extend, add new detectors, and contribute

---

## 📄 License & Citation

This project is licensed under the [MIT License](LICENSE). If you cite this repository in academic work:

```bibtex
@software{open_phase_ensemble2026,
  title = {open-phase-ensemble: Non-Parametric Multi-Tool Ensemble for Time-Series Anomaly Detection and Forecasting},
  author = {open-phase-ensemble Contributors},
  year = {2026},
  url = {https://github.com/mohamedhossammohamed/open-phase-ensemble}
}
```
