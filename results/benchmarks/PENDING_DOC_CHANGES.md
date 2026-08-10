# Pending Doc-Text Summary for BENCHMARK_ASSESSMENT.md

**Status**: NOT YET APPLIED — for user review
**Date**: 2026-08-10

## What Changed

The TSB-AD-U tuning split (48 series) has now been COMPLETED successfully:
- 48/48 series, 0 errors, 22 min runtime
- Used new resumable script with intra-series checkpointing
- Results: VUS-PR=0.2300 (default hyperparams), VUS-ROC=0.6687, persistence VUS-PR=0.2294

The previous assessment stated the tuning split "could not be completed within a practical timeframe" and that tuning was "an untested hypothesis." Both statements are now outdated.

## Three Edits Needed

### Edit 1: Line 11 — Update completion status

**Current:**
> The TSB-AD-U tuning split (48 series) was also initiated but could not be completed within a practical timeframe — several series in the tuning split exceed 100K rows, and the O(n²) TSB-AD metric computation made full completion intractable. The eval split results, which are the primary benchmark used for leaderboard ranking, are complete.

**Proposed:**
> The TSB-AD-U tuning split (48 series) has been completed with default hyperparameters using a resumable checkpointed runner (22 min, 48/48 series successful). TSB-AD's O(n²) metrics were skipped on the tuning split (only fast VUS-PR/VUS-ROC computed) to avoid the runtime bottleneck; the eval split results include the full TSB-AD metric set and are the primary benchmark used for leaderboard ranking.

### Edit 2: Line 139 — Update tuning caveat (Key Observations)

**Current:**
> 3. **Tuning caveat (untested hypothesis)**: The leaderboard results benefit from hyperparameter tuning on the 48-series tuning split; our results use default hyperparameters. Whether tuning closes the gap to baseline on *this* system is an untested hypothesis as of 2026-08-10 — the TSB-AD paper's general tuning finding does not by itself establish an effect here. See Section 5 for the fast-subset feasibility check and (if warranted) the full tuning-split run.

**Proposed (PARTIALLY CONFIRMED — tuning tested on fast subset, not yet on full split):**
> 3. **Tuning caveat (partially tested)**: The leaderboard results benefit from hyperparameter tuning on the 48-series tuning split; our eval-split results use default hyperparameters. A fast-subset feasibility check (18 series) confirmed that tuning yields +23.9% VUS-PR improvement (0.2165→0.2682), beating the persistence baseline (0.2144). The full 48-series tuning split has been completed with default hyperparams (VUS-PR=0.2300 vs persistence=0.2294 — essentially a tie), but the tuned configuration has not yet been run on the full split. See Section 5 for the fast-subset feasibility check and `results/benchmarks/EXPERIMENT_LOG.md` for full details.

### Edit 3: Line 174 — Update weakness (Weaknesses section)

**Current:**
> - **Untuned while below baseline (hypothesis, untested)**: Hyperparameter tuning may close some of the gap to baseline (untested on this system as of 2026-08-10). The TSB-AD paper's general finding that tuning improves results is not a mitigating factor here until tuning is actually run on this system; being untuned while losing to a trivial baseline is a weakness, not a strength.

**Proposed (PARTIALLY CONFIRMED):**
> - **Untuned while near baseline (partially tested)**: On the fast subset (18 series), tuning improved VUS-PR by +23.9% (0.2165→0.2682), beating persistence (0.2144). On the full 48-series tuning split with default hyperparams, the model essentially ties persistence (VUS-PR 0.2300 vs 0.2294) but significantly beats it on VUS-ROC (0.6687 vs 0.5639). The tuned configuration has not yet been validated on the full split or the 350-series eval split. Being untuned while only tying a trivial baseline on the primary metric remains a weakness.

### Edit 4: Lines 221-224 — Update Section 7 (Benchmark Completion Notes)

**Current:**
> ### Tuning split (not completed)
> The TSB-AD-U tuning split (48 series) was initiated but could not be completed within a practical timeframe. The tuning split contains several series with extremely high row counts (e.g., 650K, 230K, 195K rows). Even after downsampling to 10K points per series, the TSB-AD metric library's O(n²) point-adjusted F1 and VUS computations made full completion intractable — the process ran for over 2 hours without finishing the final large series.
>
> This does not affect the validity of the eval split results, which are the primary benchmark used for leaderboard ranking. The tuning split is supplementary and primarily used for hyperparameter optimization, which was outside the scope of this evaluation (default configuration was used throughout).

**Proposed:**
> ### Tuning split (completed, default hyperparams)
> The TSB-AD-U tuning split (48 series) has been completed with default hyperparameters using a new resumable checkpointed runner (`scripts/run_benchmarks_resumable.py`). The runner saves progress every 500 points (intra-series checkpointing) and after each series completes, enabling crash-safe resumption. Total runtime: 22 min, 48/48 series successful.
>
> TSB-AD's O(n²) metric suite was skipped on the tuning split (only fast VUS-PR/VUS-ROC computed) to avoid the runtime bottleneck that previously made completion intractable. The tuning split is used for hyperparameter optimization; a fast-subset feasibility check (18 series) confirmed +23.9% VUS-PR improvement from tuning, but the tuned configuration has not yet been run on the full split.
>
> Results: VUS-PR=0.2300 (mean), VUS-ROC=0.6687 (mean), persistence VUS-PR=0.2294. See `results/benchmarks/EXPERIMENT_LOG.md` for full details.

## Summary of Factual Changes

| Claim in current doc | Status | Evidence |
|----------------------|--------|----------|
| Tuning split "could not be completed" | FALSE | 48/48 completed in 22 min |
| Tuning is "untested hypothesis" | PARTIALLY FALSE | Tested on 18-series fast subset: +23.9% VUS-PR |
| "process ran for over 2 hours without finishing" | OBSOLETE | Original script had no checkpointing; new script fixes this |
| Default config "used throughout" | STILL TRUE for eval split | Tuning split now also run with defaults, but tuned config not yet applied |
