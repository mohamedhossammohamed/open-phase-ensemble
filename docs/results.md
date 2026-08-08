# Preliminary Results and Benchmarks

!!! warning "Experimental Research Disclaimer — Pending Independent Review"
    **All performance numbers presented on this page are preliminary, self-reported, and have not yet been independently validated or peer-reviewed.** These results are provided solely to facilitate open scientific scrutiny and independent reproduction. Use at your own risk.

---

## 📊 Summary Benchmark Evaluation (Honest Un-Buffered Metrics)

The system was evaluated against standard time-series anomaly benchmarks using Volume Under Surface (VUS-ROC and VUS-PR) metrics without point-adjustment protocols (PA-F1), and compared against Iterative Amplitude Adjusted Fourier Transform (IAAFT) phase-randomized surrogate null models. Temporal range buffering is applied strictly to ground truth labels, leaving predicted scores un-buffered.

| Benchmark Dataset | Evaluated Points ($N$) | System VUS-ROC | System VUS-PR | IAAFT Null VUS-ROC | **Predictive Edge ($\Delta$)** | Evaluation Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **PhysioNet MIT-BIH (`record_100`)** | 5,143 | **0.8592** | 0.6926 | 0.1822 | **+0.6770** | Un-buffered / Honest |
| **CWRU Bearing Prognostics** | 5,000 | **0.9711** | 0.7111 | 0.4347 | **+0.5364** | Un-buffered / Honest |

---

## 🔍 Known Limitations and Audit Findings

Following a rigorous internal code and scientific audit, the following remediations were implemented:

1. **Evaluation Metric Correction**:
   An internal audit identified a bug in `src/tsad/evaluation/vus.py` where `apply_range_buffer` was max-pooling predicted continuous scores as well as binary ground truth labels. This artificially inflated earlier reported numbers. The evaluation logic has been corrected so range buffering is applied strictly to ground truth labels. The numbers reported above reflect this un-inflated evaluation.

2. **Algorithmic Simplification**:
   - `ARFilterDetector`: Linear autoregressive ridge filter, replacing full non-stationary SARIMA.
   - `MSETransformerAutoencoder`: Standard Mean Squared Error sequence reconstruction, replacing complex association discrepancy attention.

3. **Reference Comparison Status**:
   Direct numerical comparison to external closed-source references is currently paused pending standardized metric alignment.

---

## 📈 Metric Definitions

1. **Volume Under Surface ROC (VUS-ROC)**:
   Integrates Area Under the ROC Curve across a continuous spectrum of temporal buffer thresholds $l \in [0, L_{max}]$ ($L_{max} = 15$). Range buffering is applied strictly to ground truth labels.

2. **Predictive Edge ($\Delta$)**:
   $$\Delta = \text{VUS-ROC}_{\text{system}} - \text{VUS-ROC}_{\text{IAAFT surrogate}}$$
   A positive edge ($\Delta \ge +0.30$) confirms that the system's predictive accuracy stems from true non-linear dynamic pattern recognition rather than linear autocorrelation or baseline noise.

---

## 🧪 Independent Replication Instructions

Researchers can independently reproduce these benchmark numbers using the provided script:

```bash
# Clone repository
git clone https://github.com/mohamedhossammohamed/open-phase-ensemble.git
cd open-phase-ensemble

# Install environment
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# Acquire raw datasets
python data/download.py

# Run live benchmark evaluation script
PYTHONPATH=src python scripts/run_benchmark.py
```
