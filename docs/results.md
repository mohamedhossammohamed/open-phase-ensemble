# Preliminary Results and Benchmarks

!!! note "Preliminary — Pending Independent Review"
    All performance numbers on this page are preliminary, self-reported, and have not been independently validated or peer-reviewed. They are provided to facilitate open scientific scrutiny and independent reproduction.

---

## 📊 Summary Benchmark Metrics

<div class="mat-card-grid">
  <div class="mat-card">
    <div class="mat-card-header">
      <span class="mat-card-icon">🫀</span>
      <div class="mat-card-title">PhysioNet MIT-BIH (rec 100)</div>
    </div>
    <div class="mat-card-value">0.8592</div>
    <div class="mat-card-sub">VUS-ROC (Label-Only Buffer) | Edge Δ = +0.37</div>
  </div>

  <div class="mat-card">
    <div class="mat-card-header">
      <span class="mat-card-icon">⚙️</span>
      <div class="mat-card-title">CWRU Bearing Prognostics</div>
    </div>
    <div class="mat-card-value">0.9711</div>
    <div class="mat-card-sub">VUS-ROC (Label-Only Buffer) | Edge Δ = +0.37</div>
  </div>

  <div class="mat-card">
    <div class="mat-card-header">
      <span class="mat-card-icon">🎯</span>
      <div class="mat-card-title">Zero-Lookahead Invariant</div>
    </div>
    <div class="mat-card-value">0.0000</div>
    <div class="mat-card-sub">Exact Stream vs Batch Euclidean Distance</div>
  </div>
</div>

---

## Benchmark Evaluation

The system was evaluated on standard time-series anomaly benchmarks using Volume Under Surface metrics (VUS-ROC and VUS-PR) as defined by Paparrizos et al. (2022). Temporal range buffering is applied **strictly to ground-truth labels only** — predicted anomaly scores are never buffered. This avoids the artificial inflation caused by point-adjustment protocols (PA-F1).

Performance is compared against Iterative Amplitude Adjusted Fourier Transform (IAAFT) phase-randomized surrogate null models scored under the identical protocol.

!!! note "Stochastic Baseline"
    IAAFT surrogate scores vary across runs because surrogate generation is inherently stochastic. System VUS-ROC scores are fully deterministic.

| Dataset | $N$ | System VUS-ROC | System VUS-PR | IAAFT Null VUS-ROC | Edge ($\Delta$) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **PhysioNet MIT-BIH (record 100)** | 5,143 | **0.8592** | 0.6926 | ~0.49 | ~+0.37 | Preliminary |
| **CWRU Bearing** | 5,000 | **0.9711** | 0.7111 | ~0.61 | ~+0.37 | Preliminary |

A positive predictive edge ($\Delta \ge +0.30$) over the IAAFT null confirms that detection accuracy stems from genuine non-linear dynamic pattern recognition, not linear autocorrelation.

---

## Known Limitations and Audit Findings

An internal code and scientific integrity audit identified and remediated the following issues:

**1. Evaluation Metric Correction**

An earlier version of `src/tsad/evaluation/vus.py` applied `apply_range_buffer` to both ground-truth labels and predicted anomaly scores. This artificially inflated VUS-ROC by approximately +0.06 to +0.12. The function now buffers labels only, consistent with the formal VUS definition. All numbers above reflect the corrected evaluation.

**2. Detector Naming Alignment**

Two detectors were renamed to accurately describe their mathematical implementations:

- **ARFilterDetector** — Online AR($p$) linear ridge regression filter. The previous name "SARIMADetector" implied seasonal ARIMA with MA terms and differencing, which were not implemented.
- **MSETransformerAutoencoder** — Standard Mean Squared Error sequence reconstruction. The previous name "AnomalyTransformerDetector" implied association discrepancy (KL-divergence between prior and series attention), which was not implemented.

**3. Reference Comparison Not Valid**

Direct numerical comparison to the closed-source `phase_space_matcher` reference (83.96–86.02%) is scientifically invalid for the following reasons:

- **Metric mismatch**: The reference numbers are standard PA-F1 or ROC-AUC on full-length series; our system uses VUS-ROC with label-only buffering.
- **Evaluation length mismatch**: Our benchmark subsamples to $N=5{,}000$ points; external benchmarks evaluate on full raw series ($N > 100{,}000$).
- **Unverifiable baseline**: The reference source code and evaluation harness are not available for independent verification.

We report our VUS-ROC numbers independently without claiming superiority over external systems. A fair comparison would require both systems to be evaluated on the same data splits using the same metric implementation.

---

## Metric Definitions

**Volume Under Surface ROC (VUS-ROC)** integrates AUC-ROC across a spectrum of temporal buffer thresholds $l \in [0, L_{\max}]$ with $L_{\max} = 15$. Range buffering expands ground-truth label regions only:

$$\text{VUS-ROC} = \frac{1}{L_{\max} + 1} \sum_{l=0}^{L_{\max}} \text{AUC-ROC}(\text{labels}_l, \text{scores})$$

**Predictive Edge** $\Delta = \text{VUS-ROC}_{\text{system}} - \text{VUS-ROC}_{\text{IAAFT}}$. A positive edge $\Delta \ge +0.30$ indicates genuine non-linear predictive power beyond what linear autocorrelation can explain.

---

## Future Work

- Multi-seed evaluations to generate 95% confidence intervals are planned for a future release to further validate the preliminary point estimates.
- Standardized cross-system benchmark protocol for fair comparison with external references.
- Evaluation on additional benchmark datasets (e.g., NASA IMS, SMD, SWaT).

---

## Independent Replication

```bash
git clone https://github.com/mohamedhossammohamed/open-phase-ensemble.git
cd open-phase-ensemble
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
python data/download.py
PYTHONPATH=src python scripts/run_benchmark.py
```
