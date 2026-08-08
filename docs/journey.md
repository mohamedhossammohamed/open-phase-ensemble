# The Journey

!!! warning "Experimental Research Disclaimer"
    This project is experimental and provided for research and educational purposes only. All performance claims are preliminary, self-reported, and have not yet been independently validated or peer-reviewed. Use at your own risk.

---

## 📖 The Origin: Studying a Closed-Source Reference

The genesis of `open-phase-ensemble` began with a study of a closed-source time-series verification engine ([`phase_space_matcher`](https://github.com/aiempaths/phase_space_matcher)). That reference system demonstrated a compelling premise: raw time-series data could be mapped into high-dimensional phase-space manifolds using Takens' delay embeddings to predict dynamics via trajectory geometry, avoiding the massive parameter counts of deep neural networks.

However, as we audited the mathematical bounds of pure phase-space trajectory matching, a clear structural limitation emerged:
- Pure phase-space trajectory matching excels at non-linear dynamic shifts, but struggles when anomalies present as spatial covariance shifts, high-dimensional variance drops, or global point outliers.
- On standard benchmark suites, single-paradigm trajectory matching plateaus at **83.96%–86.02% VUS-ROC accuracy**.

---

## 💡 The Intuition: Combining Orthogonal Tools

We hypothesized that no single detection algorithm—whether phase-space trajectory matching, robust covariance estimation, subsequence motif searching, or neural self-attention—can be universally optimal across all anomaly types.

Instead of searching for a single "silver bullet" algorithm, we designed a **battery of 6 orthogonal expert detectors**:
1. **Simplex Projection (EDM)** — Phase-space trajectory geometry
2. **Robust Mahalanobis (Ledoit-Wolf)** — Covariance geometry & variance shift
3. **Matrix Profile (STOMP)** — Subsequence motif & discord discovery
4. **Isolation Forest** — Subspace partitioning & global point anomalies
5. **SARIMA Residuals** — Linear stochastic control
6. **Anomaly Transformer** — Neural association discrepancy

By fusing these six distinct paradigms under an online multiplicative-weights algorithm (**the Meta-Judge**), the ensemble automatically boosts the weight of whichever expert detector successfully identifies the anomaly in real time.

---

## 🔓 Choosing to Build Openly

We chose to build `open-phase-ensemble` entirely in the open for three reasons:

1. **Scientific Reproducibility**: Closed-source reference benchmarks cannot be audited by the community. Open code ensures every claim can be independently challenged or verified.
2. **Metric Integrity**: Eliminating point-adjustment protocols (PA-F1) in favor of unadjusted Volume Under Surface (VUS-ROC/PR) prevents artificial score inflation.
3. **Community Collaboration**: By making the modular DAG architecture open source, researchers worldwide can easily plug in new detector modules, surrogate generators, or gating strategies.
