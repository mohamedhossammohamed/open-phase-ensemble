# open-phase-ensemble

An open-source, non-parametric, multi-tool ensemble for streaming time-series anomaly detection and forecasting.

---

!!! note "Preliminary Research — Pending Independent Review"
    This project is experimental and provided for research and educational purposes only. All performance claims are preliminary and self-reported. Do not use for safety-critical, medical, financial, or production decisions without independent expert review. See the [full disclaimer](disclaimer.md).

---

## 📊 Benchmark Highlights (Un-Buffered Standard VUS Metrics)

<div class="mat-card-grid">
  <div class="mat-card">
    <div class="mat-card-header">
      <span class="mat-card-icon">🫀</span>
      <div class="mat-card-title">PhysioNet MIT-BIH</div>
    </div>
    <div class="mat-card-value">0.8592</div>
    <div class="mat-card-sub">Predictive Edge Δ = +0.37 over IAAFT Null</div>
  </div>

  <div class="mat-card">
    <div class="mat-card-header">
      <span class="mat-card-icon">⚙️</span>
      <div class="mat-card-title">CWRU Bearing</div>
    </div>
    <div class="mat-card-value">0.9711</div>
    <div class="mat-card-sub">Predictive Edge Δ = +0.37 over IAAFT Null</div>
  </div>

  <div class="mat-card">
    <div class="mat-card-header">
      <span class="mat-card-icon">🧪</span>
      <div class="mat-card-title">Unit & E2E Tests</div>
    </div>
    <div class="mat-card-value">34 / 34</div>
    <div class="mat-card-sub">100% Pass Rate & Zero Lookahead Invariant</div>
  </div>
</div>

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

## ⚡ Interactive System Simulation

See how scalar time-series data streams through the five core compartments in real time:

<div class="sim-container sim-canvas-container" style="background-color: #191919 !important; background-image: none !important; border: 1px solid #212327 !important; border-radius: 8px !important; padding: 1.5rem !important; margin: 2rem 0 !important; color: #ffffff !important;">
  <div class="sim-header" style="background: transparent !important; border-bottom: 1px solid #212327 !important; padding-bottom: 0.8rem !important; margin-bottom: 1.2rem !important; display: flex; justify-content: space-between; align-items: center;">
    <div class="sim-title" style="color: #ffffff !important; font-family: 'Geist Mono', 'JetBrains Mono', monospace; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.12em;">
      <span>⚙️</span> INTERACTIVE DATA FLOW // SYSTEM COMPARTMENTS SIMULATION
    </div>
    <div class="sim-controls" style="display: flex; gap: 0.4rem;">
      <button class="sim-btn sim-btn-play" style="background-color: #0a0a0a !important; color: #ffffff !important; border: 1px solid rgba(255,255,255,0.2) !important; border-radius: 9999px !important; font-family: Inter, sans-serif; font-size: 0.8rem; padding: 0.4rem 1rem; cursor: pointer;">Pause</button>
      <button class="sim-btn sim-btn-anomaly danger" style="background-color: #0a0a0a !important; color: #ff7a17 !important; border: 1px solid rgba(255,122,23,0.5) !important; border-radius: 9999px !important; font-family: Inter, sans-serif; font-size: 0.8rem; padding: 0.4rem 1rem; cursor: pointer;">Inject Anomaly Spike</button>
      <button class="sim-btn sim-btn-reset" style="background-color: #0a0a0a !important; color: #ffffff !important; border: 1px solid rgba(255,255,255,0.2) !important; border-radius: 9999px !important; font-family: Inter, sans-serif; font-size: 0.8rem; padding: 0.4rem 1rem; cursor: pointer;">Reset</button>
    </div>
  </div>
  <canvas class="sim-canvas" style="width: 100%; height: 300px; background-color: #0a0a0a !important; border: 1px solid #212327 !important; border-radius: 6px; display: block;"></canvas>
</div>

---

## Core Principles

1. **Transparency** — Every matrix operation, weight update, and gating transition is fully inspectable and auditable.
2. **Reproducibility** — 100% deterministic execution with fixed seeds, verified by SHA-256 hash comparison.
3. **Honesty** — Corrected evaluation metrics, disclosed limitations, no inflated claims.
