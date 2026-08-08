# Architecture & Specifications

!!! note "Preliminary Research — Pending Independent Review"
    This project is experimental. All claims are preliminary and self-reported. See the [full disclaimer](disclaimer.md).

---

## ⚡ Interactive Data Flow & System Simulation

The interactive canvas below simulates the live streaming data flow through the five core pipeline compartments. Click **"Inject Anomaly Spike"** to observe how the 6-detector battery and online Hedge Meta-Judge dynamically adapt weight allocations in real time.

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

## 🏛️ System Compartments & Module Contracts

<div class="mat-card-grid">
  <div class="mat-card">
    <div class="mat-card-header">
      <span class="mat-card-icon">1️⃣</span>
      <div class="mat-card-title">Module 1: Ingestion</div>
    </div>
    <div class="mat-card-value">StreamBuffer</div>
    <div class="mat-card-sub">Rolling Median & MAD Z-Score Standardization</div>
  </div>

  <div class="mat-card">
    <div class="mat-card-header">
      <span class="mat-card-icon">2️⃣</span>
      <div class="mat-card-title">Module 2: Representation</div>
    </div>
    <div class="mat-card-value">Takens & JL</div>
    <div class="mat-card-sub">Phase-Space Reconstruction & HNSW ANN Index</div>
  </div>

  <div class="mat-card">
    <div class="mat-card-header">
      <span class="mat-card-icon">3️⃣</span>
      <div class="mat-card-title">Module 3: Battery</div>
    </div>
    <div class="mat-card-value">6 Detectors</div>
    <div class="mat-card-sub">Orthogonal Dynamic, Covariance & Neural Experts</div>
  </div>

  <div class="mat-card">
    <div class="mat-card-header">
      <span class="mat-card-icon">4️⃣</span>
      <div class="mat-card-title">Module 4: Meta-Judge</div>
    </div>
    <div class="mat-card-value">Hedge Fusion</div>
    <div class="mat-card-sub">Multiplicative Weights + Fixed-Share Floor</div>
  </div>

  <div class="mat-card">
    <div class="mat-card-header">
      <span class="mat-card-icon">5️⃣</span>
      <div class="mat-card-title">Module 5: Gating</div>
    </div>
    <div class="mat-card-value">CUSUM Chart</div>
    <div class="mat-card-sub">Freeze on Alarm / Flush on Concept Drift</div>
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

## ⚙️ Configuration Schema (`src/tsad/config.py`)

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

## 🛡️ Zero-Lookahead Leakage Invariant
The system strictly enforces zero-lookahead temporal isolation:
- All transformations operate on causal buffers $W_t = \{x_1, x_2, \dots, x_t\}$.
- `tests/integration/test_no_lookahead.py` verifies that streaming scalar processing element-by-element yields an exact Euclidean distance of **`0.0`** compared to sequential processing.
