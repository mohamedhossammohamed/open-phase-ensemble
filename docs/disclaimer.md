# Disclaimer & Review Status

!!! warning "Experimental Research Disclaimer"
    **This project is experimental and provided for research and educational purposes only.**

    All performance claims are preliminary, self-reported, and have not been independently validated or peer-reviewed. This system must not be used for safety-critical, medical, financial, or production decisions without independent professional assessment and external review by qualified domain experts.

    Use at your own risk.

---

## Audit & Review Status

| Milestone | Status | Details |
| :--- | :---: | :--- |
| Internal Code Integrity Audit | ✅ Passed | All modules verified as real, non-stub implementations |
| Internal Results Integrity Audit | ✅ Passed | Metric inflation corrected; numbers re-verified |
| Unit, Integration & E2E Tests | ✅ 34/34 Passed | Full test suite passing on Python 3.12 |
| Zero-Lookahead Invariant | ✅ Verified | Stream vs. batch Euclidean distance = 0.0 |
| Execution Determinism | ✅ Verified | SHA-256 hash reproducibility across runs |
| VUS Metric Correction | ✅ Applied | Label-only buffering; scores never buffered |
| Independent External Peer Review | ⏳ Pending | Not yet independently reviewed or reproduced |
| Safety-Critical / Production Audit | ❌ Not Audited | Not suitable for production without external review |

---

## What Was Found and Corrected

During the internal audit, the following issues were identified and remediated:

1. **VUS score inflation** — `apply_range_buffer` was applied to both labels and predicted scores. Corrected to buffer labels only.
2. **Detector naming mismatch** — Two detectors were renamed to match their actual mathematical implementations (ARFilterDetector, MSETransformerAutoencoder).
3. **Reference comparison invalidity** — Comparison to a closed-source reference was flagged as apples-to-oranges (metric type mismatch). Removed from claims.
4. **Unit test failures** — Three failing tests were fixed by correcting the underlying implementations (StreamBuffer `__len__`, MetaJudge convex fusion, FNN threshold calibration).

---

## Requesting an Independent Review

If you are a researcher or domain expert interested in auditing or replicating these results:

1. Open a **Scientific Review Request Issue** on GitHub.
2. Submit your independent evaluation methodology and results.
3. We will link all external replication reports on this site regardless of whether they confirm or challenge our numbers.
