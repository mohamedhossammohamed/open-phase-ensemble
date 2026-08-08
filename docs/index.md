# open-phase-ensemble

An open-source, non-parametric, multi-tool ensemble for streaming time-series anomaly detection and forecasting.

---

!!! note "Preliminary Research — Pending Independent Review"
    This project is experimental and provided for research and educational purposes only. All performance claims are preliminary and self-reported. Do not use for safety-critical, medical, financial, or production decisions without independent expert review. See the [full disclaimer](disclaimer.md).

---

## What This System Does

Time-series anomaly detection suffers from methodological fragmentation and evaluation pitfalls. Many benchmarks rely on flawed metrics that inflate random guessers to >90% accuracy. Heavily parameterized deep models remain opaque and computationally expensive.

**open-phase-ensemble** takes a different approach:

- **Non-parametric foundation** — Reconstructs phase-space manifolds via Takens' delay embedding instead of training millions of parameters.
- **Orthogonal 6-detector battery** — Combines phase-space trajectory prediction, covariance geometry, subsequence motifs, subspace isolation, linear autoregression, and MSE transformer reconstruction.
- **Online Meta-Judge** — Uses the Hedge multiplicative-weights algorithm to dynamically reweight detectors in real time without ground-truth labels.
- **Zero-lookahead invariant** — Strict element-by-element streaming. No future data is ever exposed during scoring.
- **Methodological rigor** — Evaluated using standard VUS-ROC/PR with label-only buffering and IAAFT phase-randomized surrogate nulls.

---

## Preliminary Results

!!! note "Pending External Review"
    All numbers below are preliminary point estimates from a single deterministic run.

| Dataset | System VUS-ROC | Predictive Edge |
| :--- | :---: | :---: |
| PhysioNet MIT-BIH (record 100) | **0.8592** | ~+0.37 |
| CWRU Bearing | **0.9711** | ~+0.37 |

See [Results & Benchmarks](results.md) for full methodology, known limitations, and replication instructions.

---

## Core Principles

1. **Transparency** — Every matrix operation, weight update, and gating transition is fully inspectable and auditable.
2. **Reproducibility** — 100% deterministic execution with fixed seeds, verified by SHA-256 hash comparison.
3. **Honesty** — Corrected evaluation metrics, disclosed limitations, no inflated claims.
