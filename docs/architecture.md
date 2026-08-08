# Architecture & Specifications

!!! note "Preliminary Research — Pending Independent Review"
    This project is experimental. All claims are preliminary and self-reported. See the [full disclaimer](disclaimer.md).

---

## Interactive Data Flow & System Simulation

The interactive canvas below simulates the live streaming data flow through the five core pipeline compartments. Click **"Inject Anomaly Spike"** to observe how the 6-detector battery and online Hedge Meta-Judge dynamically adapt weight allocations in real time.

<div class="sim-container sim-canvas-container">
  <div class="sim-header">
    <div class="sim-title">
      INTERACTIVE DATA FLOW // SYSTEM COMPARTMENTS SIMULATION
    </div>
    <div class="sim-controls">
      <button class="sim-btn sim-btn-play">Pause</button>
      <button class="sim-btn sim-btn-anomaly danger">Inject Anomaly Spike</button>
      <button class="sim-btn sim-btn-reset">Reset</button>
    </div>
  </div>
  <canvas class="sim-canvas"></canvas>
  <div class="sim-caption">
    Figure 1: Streaming data pipeline compartments showing real-time scalar signal processing, embedding reconstruction, multi-expert scoring, online Hedge weight adjustment, and CUSUM alarm gating.
  </div>
</div>

---

## System Compartments & Module Contracts

<div class="stat-grid">
  <div class="stat-card">
    <div class="stat-label">Module 01 // Ingestion</div>
    <div class="stat-value" style="font-size: 1.35rem;">StreamBuffer</div>
    <div class="stat-sub">Rolling Median & MAD Z-Score Standardization</div>
  </div>

  <div class="stat-card">
    <div class="stat-label">Module 02 // Representation</div>
    <div class="stat-value" style="font-size: 1.35rem;">Takens & JL</div>
    <div class="stat-sub">Phase-Space Reconstruction & HNSW ANN Index</div>
  </div>

  <div class="stat-card">
    <div class="stat-label">Module 03 // Battery</div>
    <div class="stat-value" style="font-size: 1.35rem;">6 Detectors</div>
    <div class="stat-sub">Orthogonal Dynamic, Covariance & Neural Experts</div>
  </div>

  <div class="stat-card">
    <div class="stat-label">Module 04 // Meta-Judge</div>
    <div class="stat-value" style="font-size: 1.35rem;">Hedge Fusion</div>
    <div class="stat-sub">Multiplicative Weights + Fixed-Share Floor</div>
  </div>

  <div class="stat-card">
    <div class="stat-label">Module 05 // Gating</div>
    <div class="stat-value" style="font-size: 1.35rem;">CUSUM Chart</div>
    <div class="stat-sub">Freeze on Alarm / Flush on Concept Drift</div>
  </div>
</div>

```
[Raw Scalar x_t] 
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│ Module 1: Ingestion & Preprocessing                         │
│ - StreamBuffer(window_size=200)                             │
│ - Standardized output: v_t = (x_t - med) / (1.4826 * mad)   │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Module 2: Multidimensional Representation                   │
│ - Takens Delay Embedding: X_t in R^d (tau=2, d=8)           │
│ - JL Random Projection: Z_t in R^D_target (D_target=8)      │
│ - HNSW Graph Index: ANN trajectory indexing                 │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Module 3: 6-Detector Battery                                │
│ [Simplex, Mahalanobis-LW, MatrixProfile, IForest,           │
│  ARFilterDetector, MSETransformerAutoencoder]               │
│ Output: Scores S_t in [0,1]^K, Forecasts v_hat in R^K       │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Module 4 & 5: Meta-Judge & Online Learning Loop             │
│ - Pearson Correlation Loss: ell_{t,k} = 1 - PearsonCorr(S,E)│
│ - Hedge Multiplicative Weight Update + Fixed-Share Floor    │
│ - Output: Fused Anomaly Score A_t, Fused Forecast v_hat*    │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Module 6: CUSUM Change Detection Gating                     │
│ - C_t+ > H_c: Freeze adaptation during acute alarms         │
│ - Alarm >= T_drift (200): Flush baseline & reset CUSUM      │
└─────────────────────────────┴───────────────────────────────┘
```

---

## Configuration Schema (`src/tsad/config.py`)

| Parameter | Type | Default Value | Description |
| :--- | :--- | :---: | :--- |
| `SEED` | `int` | `42` | Global random seed enforcing 100% determinism |
| `MAX_TAU` | `int` | `100` | Clamped upper bound for time delay lag $\tau$ |
| `MAX_D` | `int` | `20` | Clamped upper bound for embedding dimension $d$ |
| `D_TARGET` | `int` | `8` | Johnson-Lindenstrauss target dimension |
| `K_DETECTORS` | `int` | `6` | Total number of expert detectors in battery |
| `HEDGE_ETA` | `float` | `0.1` | Hedge learning rate parameter $\eta$ |
| `FIXED_SHARE_SIGMA`| `float` | `0.01` | Fixed-share mixing floor parameter $\sigma$ |
| `CUSUM_KC` | `float` | `0.5` | CUSUM allowance parameter $k_c$ |
| `CUSUM_HC_SIGMA_MULT`| `float` | `5.0` | CUSUM alarm threshold multiplier $H_c = 5.0 \sigma_E$ |
| `T_DRIFT` | `int` | `200` | Consecutive alarm steps before concept drift reset |
| `EPSILON` | `float` | `1e-6` | Numerical stability parameter preventing zero division |

---

## Zero-Lookahead Leakage Invariant

The system strictly enforces zero-lookahead temporal isolation:

<dl class="def-list">
  <dt>Causal Buffer Scope</dt>
  <dd>All transformations operate exclusively on historical causal buffers $W_t = \{x_1, x_2, \dots, x_t\}$. No future sample is ever accessed.</dd>

  <dt>Exact Invariant Verification</dt>
  <dd><code>tests/integration/test_no_lookahead.py</code> verifies that streaming scalar processing element-by-element yields an exact Euclidean distance of <strong>0.0</strong> compared to sequential batch processing.</dd>
</dl>
