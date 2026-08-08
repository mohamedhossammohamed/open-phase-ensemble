# The Journey

!!! note "Preliminary Research — Pending Independent Review"
    This project is experimental. All claims are preliminary and self-reported. See the [full disclaimer](disclaimer.md).

---

## Origin: Studying a Closed-Source Reference

The genesis of **open-phase-ensemble** began with a study of a closed-source time-series verification engine. That system demonstrated a compelling premise: raw time-series data could be mapped into phase-space manifolds using Takens' delay embeddings to predict dynamics via trajectory geometry, avoiding the massive parameter counts of deep neural networks.

As we audited the mathematical bounds of pure phase-space trajectory matching, a structural limitation emerged: it excels at non-linear dynamic shifts but struggles with spatial covariance shifts, variance drops, or global point outliers.

---

## Intuition: Combining Orthogonal Tools

No single detection algorithm can be universally optimal across all anomaly types. Instead of searching for a single "silver bullet," we designed a battery of six orthogonal expert detectors:

<dl class="def-list">
  <dt>Simplex Projection (EDM)</dt>
  <dd>Phase-space trajectory geometry</dd>

  <dt>Robust Mahalanobis (Ledoit-Wolf)</dt>
  <dd>Covariance geometry &amp; variance shift</dd>

  <dt>Matrix Profile (STOMP)</dt>
  <dd>Subsequence motif &amp; discord discovery</dd>

  <dt>Isolation Forest</dt>
  <dd>Subspace partitioning &amp; global point anomalies</dd>

  <dt>AR Linear Ridge Filter</dt>
  <dd>Linear stochastic prediction residuals</dd>

  <dt>MSE Transformer Autoencoder</dt>
  <dd>Neural sequence reconstruction error</dd>
</dl>

By fusing these paradigms under an online multiplicative-weights algorithm (the **Meta-Judge**), the ensemble automatically promotes whichever expert successfully identifies the current anomaly type.

---

## Choosing to Build Openly

We chose to build entirely in the open for three primary reasons:

<dl class="def-list">
  <dt>Reproducibility</dt>
  <dd>Closed-source benchmarks cannot be audited. Open code ensures every claim can be independently verified.</dd>

  <dt>Metric Integrity</dt>
  <dd>Using standard VUS-ROC with label-only buffering instead of inflated point-adjustment protocols.</dd>

  <dt>Community Collaboration</dt>
  <dd>A modular architecture where researchers can plug in new detectors, surrogate generators, or gating strategies.</dd>
</dl>
