# Benchmark Assessment: open-phase-ensemble

## Executive Summary

open-phase-ensemble was evaluated on two industry-standard benchmarks:

1. **TSB-AD-U** (NeurIPS 2024 D&B Track) — 350 univariate time series, eval split. The model achieves **VUS-PR 0.1995**, ranking approximately **#28 out of 30+ methods** on the official leaderboard. The persistence baseline outperforms on the primary metric (VUS-PR 0.2249 vs 0.1995), though open-phase-ensemble shows stronger ranking quality (VUS-ROC 0.6614 vs 0.5723).

2. **NAB** (Numenta Anomaly Benchmark) — 29 series. The model achieves **VUS-PR 0.1808**, decisively beating the persistence baseline on all metrics (VUS-PR 0.1808 vs 0.0902; VUS-ROC 0.7937 vs 0.5950).

The TSB-AD-U tuning split (48 series) was also initiated but could not be completed within a practical timeframe — several series in the tuning split exceed 100K rows, and the O(n²) TSB-AD metric computation made full completion intractable. The eval split results, which are the primary benchmark used for leaderboard ranking, are complete.

---

## 1. Evaluation Setup

### Datasets
- **TSB-AD-U**: NeurIPS 2024 Datasets & Benchmarks Track — 350 eval series (out of 870 total)
- **NAB**: Numenta Anomaly Benchmark (2015) — 29 series (NAB is a subset of TSB-AD-U)
- **TSB-AD-U tuning split**: 48 series (initiated; see Section 7 for completion notes)
- **Primary metric**: VUS-PR (Volume Under Surface, Precision-Recall) — identified by TSB-AD authors as the most reliable TSAD metric

### Model Configuration
- `warmup_fraction`: 0.05
- `max_buffer`: 15 (VUS sliding window)
- `n_surrogates`: 0 (no IAAFT significance testing)
- `seed`: 42
- `downsample`: 10000 (max points per series)
- TSB-AD full metric suite enabled

### Metrics Computed
All 9 TSB-AD metrics plus standard VUS:
- VUS-PR, VUS-ROC (our implementation)
- TSB-AD VUS-PR, TSB-AD VUS-ROC (TSB-AD library)
- AUC-PR, AUC-ROC
- Standard-F1, PA-F1 (Point-Adjusted), Event-based-F1, R-based-F1, Affiliation-F

### Provenance
- Full per-series results: `results/benchmarks/full/TSB-AD-U_eval_20260810_045853.json`
- Summary: `results/benchmarks/full/summary_20260810_045853.json`
- Runtime: 8822.8s (~2.45 hours) for 350 series
- Success rate: 350/350 (100%)

---

## 2. Results: TSB-AD-U Eval Split (350 series)

### 2.1 Aggregate Metrics

| Metric | open-phase-ensemble | Persistence Baseline | Delta |
|--------|:-------------------:|:--------------------:|:-----:|
| **VUS-PR** (primary) | 0.1995 | 0.2249 | -0.0254 |
| VUS-ROC | 0.6614 | 0.5723 | +0.0891 |
| TSB-AD VUS-PR | 0.1938 | 0.2726 | -0.0788 |
| TSB-AD VUS-ROC | 0.7308 | 0.6829 | +0.0479 |
| AUC-PR | 0.1727 | 0.2561 | -0.0834 |
| AUC-ROC | 0.6992 | 0.6291 | +0.0701 |
| Standard-F1 | 0.2481 | 0.3174 | -0.0693 |
| PA-F1 | 0.6409 | 0.7665 | -0.1256 |
| Event-based-F1 | 0.4255 | 0.6050 | -0.1795 |
| R-based-F1 | 0.2606 | 0.3348 | -0.0742 |
| Affiliation-F | 0.8166 | 0.8789 | -0.0623 |

### 2.2 Win Rate vs Persistence (per-series)

| Metric | Win Rate | Interpretation |
|--------|:--------:|:--------------:|
| VUS-PR | 51.1% | Coin-flip; persistence wins on average |
| VUS-ROC | 70.0% | Open-phase ranks anomalies better |
| TSB-AD VUS-PR | 44.9% | Persistence wins more often |
| TSB-AD VUS-ROC | 51.6% | Roughly even |
| Standard-F1 | 48.4% | Roughly even |
| PA-F1 | 20.3% | Persistence dominates (bias toward recall) |
| Event-based-F1 | 24.6% | Persistence dominates |
| Affiliation-F | 25.2% | Persistence dominates |

### 2.3 Distribution Statistics

| Statistic | VUS-PR (OP) | VUS-PR (Pers) | VUS-ROC (OP) | VUS-ROC (Pers) |
|-----------|:-----------:|:-------------:|:------------:|:--------------:|
| Mean | 0.1995 | 0.2249 | 0.6614 | 0.5723 |
| Median | 0.1496 | 0.1622 | 0.6654 | — |
| Std | 0.1890 | 0.2183 | 0.1550 | — |
| N | 350 | 350 | 350 | 350 |

---

## 2b. Results: NAB (29 series)

### Aggregate Metrics

| Metric | open-phase-ensemble | Persistence Baseline | Delta |
|--------|:-------------------:|:--------------------:|:-----:|
| **VUS-PR** (primary) | 0.1808 | 0.0902 | **+0.0906** |
| VUS-ROC | 0.7937 | 0.5950 | **+0.1987** |
| TSB-AD VUS-PR | 0.2082 | 0.1284 | **+0.0798** |
| TSB-AD VUS-ROC | 0.8632 | 0.7622 | **+0.1010** |
| AUC-PR | 0.1928 | 0.1498 | **+0.0430** |
| AUC-ROC | 0.8591 | 0.6519 | **+0.2072** |
| Standard-F1 | 0.2668 | 0.2202 | **+0.0466** |
| PA-F1 | 0.4513 | 0.6243 | -0.1730 |
| Event-based-F1 | 0.3490 | 0.4946 | -0.1456 |
| R-based-F1 | 0.2228 | 0.2758 | -0.0530 |
| Affiliation-F | 0.8265 | 0.8769 | -0.0504 |

### NAB Key Finding

On NAB, open-phase-ensemble **decisively outperforms the persistence baseline** on the primary VUS-PR metric (2x improvement: 0.18 vs 0.09) and on VUS-ROC (0.79 vs 0.60). This contrasts with the TSB-AD-U eval results, where persistence held the edge. The difference suggests the model's strengths are more apparent on NAB's anomaly patterns (which tend to be more sustained and point-anomaly-like) than on the broader TSB-AD-U collection.

---

## 3. Comparison to Industry SOTA (TSB-AD-U Leaderboard)

The official TSB-AD-U leaderboard (https://thedatumorg.github.io/TSB-AD/) ranks 30+ methods by VUS-PR after hyperparameter tuning on the tuning split. Key comparison points:

| Rank | Method | VUS-PR | VUS-ROC | Notes |
|:----:|--------|:------:|:-------:|-------|
| 1 | Sub-PCA | 0.42 | 0.76 | Best overall |
| 2 | KShapeAD | 0.40 | 0.76 | |
| 3 | POLY | 0.39 | 0.76 | |
| 4 | Series2Graph | 0.39 | 0.80 | |
| 5 | MOMENT (FT) | 0.39 | 0.76 | Foundation model (fine-tuned) |
| 8 | USAD | 0.36 | 0.71 | |
| 12 | CNN | 0.34 | 0.79 | |
| 16 | IForest | 0.30 | 0.78 | |
| 20 | TimesNet | 0.26 | 0.72 | |
| 28 | Donut | 0.20 | 0.68 | |
| — | **open-phase-ensemble** | **0.20** | **0.66** | **This work (default config)** |
| — | **Persistence baseline** | **0.22** | **0.57** | **Trivial baseline** |
| 29 | LOF | 0.17 | 0.68 | |
| 30 | AnomalyTransformer | 0.12 | 0.56 | Worst |

### Key Observations

1. **VUS-PR ranking**: open-phase-ensemble (0.20) sits near the bottom of the leaderboard (~#28), below the persistence baseline (0.22) and comparable to Donut (0.20).

2. **VUS-ROC ranking**: open-phase-ensemble (0.66) is mid-to-lower pack, comparable to TimesNet (0.72) and AutoEncoder (0.69), but above the persistence baseline (0.57).

3. **Tuning caveat**: The leaderboard results benefit from hyperparameter tuning on the 48-series tuning split. Our results use default hyperparameters. The TSB-AD paper emphasizes that proper tuning can significantly improve results.

4. **Precision vs Recall tradeoff**: Open-phase-ensemble shows better ranking quality (VUS-ROC) but worse precision (VUS-PR, AUC-PR). This suggests the model can distinguish anomalous from normal regions but struggles with precise anomaly boundary localization.

5. **Persistence baseline strength**: The persistence baseline's dominance on point-adjusted metrics (PA-F1: 0.77 vs 0.64) is consistent with known biases — PA-F1 rewards high-recall methods, and persistence effectively predicts anomalies broadly.

---

## 4. Per-Series Analysis

### 4.1 Best Performing Series (VUS-PR)
Open-phase-ensemble significantly outperforms persistence on specific series:
- `281_NEK_id_5_WebService`: OP=0.89 vs PERS=0.28 (+0.61)
- `284_NEK_id_8_WebService`: OP=0.87 vs PERS=0.29 (+0.58)

### 4.2 Degenerate Cases
Several series yield VUS-PR=0.0 for both methods (e.g., OPPORTUNITY HumanActivity series), suggesting either no detectable anomalies or fundamental representation issues.

### 4.3 Pattern
Open-phase-ensemble excels on WebService and synthetic series with clear anomaly patterns, but struggles on HumanActivity and multi-modal series.

---

## 5. Assessment

### Strengths
- **Ranking quality**: VUS-ROC of 0.66 (70% win rate vs persistence) shows the model can rank anomalous regions above normal ones
- **Robustness**: 100% success rate across 350 diverse series with no crashes
- **No hyperparameter tuning**: Results are with default config; tuning could improve performance
- **Specific domains**: Strong performance on WebService and synthetic anomaly series

### Weaknesses
- **Below persistence baseline on primary metric**: VUS-PR 0.20 vs 0.22 — the model is outperformed by a trivial baseline on the most reliable TSAD metric
- **Poor precision**: AUC-PR of 0.17 vs 0.26 for persistence indicates the model produces many false positives
- **Bottom-tier leaderboard position**: ~#28 of 30+ methods
- **Event detection**: Event-based-F1 of 0.43 vs 0.60 for persistence shows weak anomaly segment localization

### Root Cause Hypotheses
1. **Threshold calibration**: The model may produce well-ranked scores but with poorly calibrated absolute thresholds, leading to low precision at default operating points
2. **Representation gaps**: The simplex/phase-space representation may not capture anomaly patterns in certain domain types (HumanActivity, multi-modal)
3. **Default configuration**: Without tuning, the warmup fraction and buffer settings may be suboptimal for the diverse TSB-AD-U series

### Recommendations
1. **Hyperparameter tuning**: Run the 48-series tuning split to optimize warmup_fraction, max_buffer, and threshold parameters
2. **Threshold calibration**: Implement adaptive thresholding (e.g., POT/EVT) to improve precision at the decision boundary
3. **Representation enhancement**: Consider domain-specific features or adaptive representation selection
4. **Ensemble calibration**: If the model excels at ranking (VUS-ROC), combine with a precision-focused method

---

## 6. Online Streaming Latency

The pipeline is designed as a strict online streaming system (`TSADPipeline.step()` processes one point at a time with no future data access). Latency was measured in true point-by-point streaming mode:

| Series length | Steady-state latency | Throughput |
|---------------|:--------------------:|:----------:|
| 500 points | 4.84 ms/point | 207 pts/s |
| 2,000 points | 5.72 ms/point | 175 pts/s |
| 10,000 points | 4.29 ms/point | 233 pts/s |
| 50,000 points | 5.55 ms/point | 180 pts/s |

**~5 ms per point, ~200 points/second** — stable regardless of series length.

### Real-time feasibility by sampling rate:

| Data sampling rate | Required latency | Feasible? |
|-------------------|:----------------:|:---------:|
| 1 Hz (server metrics, IoT) | <1000 ms | Yes, 200x margin |
| 10 Hz (industrial sensors) | <100 ms | Yes, 20x margin |
| 100 Hz (fast telemetry) | <10 ms | Yes, 2x margin |
| 1000 Hz (high-frequency) | <1 ms | No, ~5x too slow |

The system is well-suited for real-time deployment at typical monitoring sampling rates (1–100 Hz).

---

## 7. Benchmark Completion Notes

### Completed benchmarks
- **TSB-AD-U eval split**: 350/350 series, 100% success rate, ~2.5 hours runtime
- **NAB eval**: 29/29 series, 100% success rate, ~10 minutes runtime

### Tuning split (not completed)
The TSB-AD-U tuning split (48 series) was initiated but could not be completed within a practical timeframe. The tuning split contains several series with extremely high row counts (e.g., 650K, 230K, 195K rows). Even after downsampling to 10K points per series, the TSB-AD metric library's O(n²) point-adjusted F1 and VUS computations made full completion intractable — the process ran for over 2 hours without finishing the final large series.

This does not affect the validity of the eval split results, which are the primary benchmark used for leaderboard ranking. The tuning split is supplementary and primarily used for hyperparameter optimization, which was outside the scope of this evaluation (default configuration was used throughout).

---

## 8. Reproducibility

### Environment
- Python 3.14.6 (pre-release; note: CI tests against 3.10–3.12 only — results were produced on a version not covered by CI)
- TSB-AD library (NeurIPS 2024)
- scikit-learn, numpy, scipy
- Full provenance in result JSON files

### Commands
```bash
# Full eval sweep (350 series)
python scripts/run_benchmarks.py --benchmark TSB-AD-U --split eval \
    --tsb-ad-metrics --downsample 10000 \
    --output-dir results/benchmarks/full

# Tuning sweep (48 series)
python scripts/run_benchmarks.py --benchmark TSB-AD-U --split train \
    --tsb-ad-metrics --downsample 10000 \
    --output-dir results/benchmarks/full

# NAB
python scripts/run_benchmarks.py --benchmark NAB \
    --tsb-ad-metrics --downsample 10000 \
    --output-dir results/benchmarks/full
```

### Data Sources
- TSB-AD-U: https://www.thedatum.org/datasets/TSB-AD-U.zip
- NAB: https://github.com/ArroyoJack/nab
- Leaderboard: https://thedatumorg.github.io/TSB-AD/

---

## References

1. Liu, Q. & Paparrizos, J. (2024). "The Elephant in the Room: Towards A Reliable Time-Series Anomaly Detection Benchmark." NeurIPS 2024 Datasets & Benchmarks Track.
2. Paparrizos, J. et al. (2022). "TSB-UAD: An End-to-End Benchmark Suite for Univariate Time-Series Anomaly Detection." VLDB 2022.
3. Kim, S. et al. (2022). "Towards a Rigorous Evaluation of Time-Series Anomaly Detection." AAAI 2022.
