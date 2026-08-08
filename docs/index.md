# open-phase-ensemble

An open-source, non-parametric, multi-tool ensemble for streaming time-series anomaly detection and forecasting.

---

!!! note "Preliminary Research — Pending Independent Review"
    This project is experimental and provided for research and educational purposes only. All performance claims are preliminary and self-reported. Do not use for safety-critical, medical, financial, or production decisions without independent expert review. See the [full disclaimer](disclaimer.md).

---

## Benchmark Highlights (Un-Buffered Standard VUS Metrics)

<div class="stat-grid">
  <div class="stat-card">
    <div class="stat-label">PhysioNet MIT-BIH</div>
    <div class="stat-value">0.8592</div>
    <div class="stat-sub">Predictive Edge $\Delta = +0.37$ over IAAFT Null</div>
  </div>

  <div class="stat-card">
    <div class="stat-label">CWRU Bearing</div>
    <div class="stat-value">0.9711</div>
    <div class="stat-sub">Predictive Edge $\Delta = +0.37$ over IAAFT Null</div>
  </div>

  <div class="stat-card">
    <div class="stat-label">Unit & E2E Tests</div>
    <div class="stat-value">34 / 34</div>
    <div class="stat-sub">100% Pass Rate & Zero Lookahead Invariant</div>
  </div>
</div>

---

## What This System Does

Time-series anomaly detection suffers from methodological fragmentation and evaluation pitfalls. Many benchmarks rely on flawed metrics that inflate random guessers to >90% accuracy. Heavily parameterized deep models remain opaque and computationally expensive.

**open-phase-ensemble** takes a different approach:

<dl class="def-list">
  <dt>Non-parametric foundation</dt>
  <dd>Reconstructs phase-space manifolds via Takens' delay embedding instead of training millions of parameters.</dd>

  <dt>Orthogonal 6-detector battery</dt>
  <dd>Combines phase-space trajectory prediction, covariance geometry, subsequence motifs, subspace isolation, linear autoregression, and MSE transformer reconstruction.</dd>

  <dt>Online Meta-Judge</dt>
  <dd>Uses the Hedge multiplicative-weights algorithm to dynamically reweight detectors in real time without ground-truth labels.</dd>

  <dt>Zero-lookahead invariant</dt>
  <dd>Enforces strict element-by-element streaming. No future data is ever exposed during scoring.</dd>

  <dt>Methodological rigor</dt>
  <dd>Evaluated using standard VUS-ROC/PR with label-only buffering and IAAFT phase-randomized surrogate nulls.</dd>
</dl>

---

## Interactive System Simulation

See how scalar time-series data streams through the five core compartments in real time:

<div class="sim-container sim-canvas-container">
  <div class="sim-header">
    <div class="sim-title">
      SYSTEM SIMULATION // CAUSAL DATA FLOW & EXPERT WEIGHT ALLOCATION
    </div>
    <div class="sim-controls">
      <button class="sim-btn sim-btn-play">Pause</button>
      <button class="sim-btn sim-btn-anomaly danger">Inject Anomaly Spike</button>
      <button class="sim-btn sim-btn-reset">Reset</button>
    </div>
  </div>
  <canvas class="sim-canvas"></canvas>
  <div class="sim-caption">
    Figure 1: Real-time interactive simulation of scalar time-series streaming through the five system compartments, illustrating Hedge Meta-Judge dynamic weight reallocation during an anomaly spike.
  </div>
</div>

---

## Core Principles

<dl class="def-list">
  <dt>Transparency</dt>
  <dd>Every matrix operation, weight update, and gating transition is fully inspectable and auditable.</dd>

  <dt>Reproducibility</dt>
  <dd>100% deterministic execution with fixed seeds, verified by SHA-256 hash comparison.</dd>

  <dt>Honesty</dt>
  <dd>Corrected evaluation metrics, disclosed limitations, and no inflated claims.</dd>
</dl>
