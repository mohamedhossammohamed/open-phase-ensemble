# Architecture & Specifications

!!! warning "Experimental Research Disclaimer"
    This project is experimental and provided for research and educational purposes only. All performance claims are preliminary, self-reported, and have not yet been independently validated or peer-reviewed. Use at your own risk.

---

## 🏛️ System Blueprint & Module Contracts

`open-phase-ensemble` operates as a strict Directed Acyclic Graph (DAG) streaming loop:

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
