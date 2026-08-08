# Home / Overview

Welcome to **open-phase-ensemble**, an open-source, non-parametric multi-tool ensemble system for streaming time-series anomaly detection and forecasting.

---

!!! warning "Experimental Research Disclaimer"
    **This project is experimental and provided for research and educational purposes only.** All performance claims are preliminary, self-reported, and have not yet been independently validated or peer-reviewed. This system must not be used for safety-critical, medical, financial, or production decisions without professional assessment and external review by qualified domain experts. Use at your own risk.

---

## 🎯 What This System Is

Time-series anomaly detection and forecasting systems frequently suffer from methodological fragmentation and evaluation pitfalls. Many contemporary benchmarks rely on flawed metrics (such as point-adjusted F1 scores) that inflate random guessers to $>90\%$ accuracy, while heavily parameterized deep learning models require high computational overhead and remain opaque.

`open-phase-ensemble` takes a different path:
- **Non-Parametric Foundation**: Reconstructs phase-space manifolds via Takens' delay embedding rather than training millions of neural parameters.
- **Orthogonal 6-Detector Battery**: Combines phase-space predictability, covariance geometry, subsequence motifs, subspace isolation, linear stochastics, and association discrepancy.
- **Online Meta-Judge**: Uses the Hedge multiplicative-weights algorithm to dynamically reweight experts in real time without ground-truth labels.
- **Zero-Lookahead Invariant**: Strictly enforces element-by-element streaming execution, ensuring zero temporal data leakage.
- **Methodological Rigor**: Evaluated using Volume Under Surface (VUS-ROC/PR) and phase-randomized surrogate null models (AR and IAAFT).

---

## 🏛️ Core Philosophy

1. **Open Reproduction & Extension**: Built to openly reproduce and extend phase-space trajectory matching concepts beyond single-algorithm closed systems.
2. **Transparency Over Proprietary Opaqueness**: Every matrix operation, weight update, and gating transition is fully inspectable, auditable, and reproducible.
3. **Determinism**: Enforces 100% bitwise execution reproducibility across random seeds (`SEED=42`).
