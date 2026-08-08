# Preliminary Results and Benchmarks

!!! warning "Experimental Research Disclaimer — Pending Independent Review"
    **All performance numbers presented on this page are preliminary, self-reported, and have not yet been independently validated or peer-reviewed.** These results are provided solely to facilitate open scientific scrutiny and independent reproduction. Use at your own risk.

---

## 📊 Summary Benchmark Evaluation

The system was evaluated against standard time-series anomaly benchmarks using Volume Under Surface (VUS-ROC and VUS-PR) metrics without point-adjustment protocols (PA-F1), and compared against Iterative Amplitude Adjusted Fourier Transform (IAAFT) phase-randomized surrogate null models.

| Benchmark Dataset | Evaluated Points ($N$) | System VUS-ROC | System VUS-PR | IAAFT Null VUS-ROC | **Predictive Edge ($\Delta$)** | Evaluation Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **PhysioNet MIT-BIH (`record_100`)** | 5,143 | **0.9563** | 0.5597 | 0.4798 | **+0.4765** | Preliminary / Self-reported |
| **CWRU Bearing Prognostics** | 5,000 | **0.9808** | 0.7085 | 0.4801 | **+0.5007** | Preliminary / Self-reported |

---

## 📈 Metric Definitions

1. **Volume Under Surface ROC (VUS-ROC)**:
   Integrates Area Under the ROC Curve across a continuous spectrum of temporal buffer thresholds $l \in [0, L_{max}]$ ($L_{max} = 15$). This metric strictly avoids the point-adjustment flaw.

2. **Predictive Edge ($\Delta$)**:
   $$\Delta = \text{VUS-ROC}_{\text{system}} - \text{VUS-ROC}_{\text{IAAFT surrogate}}$$
   A positive edge ($\Delta \ge +0.30$) confirms that the system's predictive accuracy stems from true non-linear dynamic pattern recognition rather than linear autocorrelation or baseline noise.

---

## 🧪 Independent Replication Instructions

Researchers can independently reproduce these preliminary benchmark numbers using the provided script:

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
