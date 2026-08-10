#!/usr/bin/env python3
"""Step 3: bounded, time-boxed hyperparameter tuning feasibility check on the fast subset.

Adjusts ONLY existing exposed hyperparameters (no algorithm changes):
  - HEDGE_ETA            (tsad.config, read by tsad.pipeline at construction)
  - CUSUM_KC             (tsad.config, read by tsad.pipeline at construction)
  - CUSUM_HC_SIGMA_MULT  (tsad.config, read by tsad.pipeline at construction)
  - embedding dim d_target (TSADPipeline constructor kwarg)

Strategy: coordinate descent on a 9-series search subset (shortest per domain),
then validate the best config on the full 18-series fast subset and compare to
the cached default-config baseline.
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import tsad.pipeline as pl  # noqa: E402
from tsad.benchmarks.tsb_ad import TSB_AD_U, _parse_filename  # noqa: E402
from tsad.benchmarks.base import _causal_block_downsample  # noqa: E402
from tsad.benchmarks.wrappers import TSADPipelineWrapper  # noqa: E402
from tsad.evaluation.vus import compute_vus_pr, compute_vus_roc  # noqa: E402
from step2_fast_subset import (  # noqa: E402
    run_protocol, primary_metrics, tsb_ad_metrics, domain_of,
    WARMUP_FRAC, MAX_BUFFER, DOWNSAMPLE,
)

OUT = ROOT / "results" / "benchmarks" / "step2_fastsubset"
OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "step3_tuning_results.json"
CACHE = OUT / "fastsubset_cache.npz"

# Default config values (from tsad.config) — baseline reference.
DEFAULTS = {
    "HEDGE_ETA": 0.1, "CUSUM_KC": 0.5, "CUSUM_HC_SIGMA_MULT": 5.0, "d_target": 8,
}

# Search grids (bounded).
GRID_ETA = [0.05, 0.1, 0.2]
GRID_D = [4, 8, 16]
GRID_KC = [0.25, 0.5, 1.0]
GRID_HCMULT = [3.0, 5.0, 8.0]


def make_model(eta, kc, hc_mult, d_target):
    """Set tsad.pipeline module globals (read at TSADPipeline construction) and
    return a wrapper whose factory builds a pipeline with the given embedding dim."""
    pl.HEDGE_ETA = eta
    pl.CUSUM_KC = kc
    pl.CUSUM_HC_SIGMA_MULT = hc_mult
    # D_TARGET is used as the default for d_target; pass explicitly to override.
    return TSADPipelineWrapper(d=d_target, d_target=d_target)


def select_search_subset(ds, names_full):
    """1 shortest series per domain from the full 18-series subset."""
    by_dom = defaultdict(list)
    for s in ds.iter_series("eval"):
        if s.name in set(names_full):
            by_dom[domain_of(s.name)].append((len(s.signal), s.name))
    chosen = []
    for dom in sorted(by_dom):
        chosen.append(sorted(by_dom[dom])[0][1])
    return chosen


def score_config(names, ds, eta, kc, hc_mult, d_target):
    model = make_model(eta, kc, hc_mult, d_target)
    per_series = []
    t0 = time.perf_counter()
    for name in names:
        series = next(s for s in ds.iter_series("eval") if s.name == name)
        es, el, _pe = run_protocol(series, model)
        m = primary_metrics(es, el)
        per_series.append({"name": name, "domain": domain_of(name), **m})
    return per_series, time.perf_counter() - t0


def mean_metric(rows, key):
    vals = [r[key] for r in rows if key in r]
    return float(np.mean(vals)) if vals else float("nan")


def main():
    t0 = time.perf_counter()
    ds = TSB_AD_U(data_root=ROOT / "data" / "benchmarks")
    z = np.load(CACHE, allow_pickle=True)
    names_full = z["names"].tolist()
    raw_metrics = json.loads(str(z["raw_metrics_json"]))
    pers_metrics = json.loads(str(z["pers_metrics_json"]))

    search_names = select_search_subset(ds, names_full)
    print(f"[tuning] search subset ({len(search_names)}): {search_names}")

    baseline_search = []
    # baseline (default config) on search subset — recompute to compare on same series
    rows, dt = score_config(search_names, ds, DEFAULTS["HEDGE_ETA"], DEFAULTS["CUSUM_KC"],
                            DEFAULTS["CUSUM_HC_SIGMA_MULT"], DEFAULTS["d_target"])
    baseline_search = rows
    base_pr = mean_metric(rows, "vus_pr")
    base_roc = mean_metric(rows, "vus_roc")
    print(f"[tuning] baseline (defaults) on search: VUS-PR={base_pr:.4f} VUS-ROC={base_roc:.4f} [{dt:.0f}s]")

    trials = []

    # ---- Round A: HEDGE_ETA x d_target (CUSUM at default) ----
    print("\n[tuning] Round A: HEDGE_ETA x d_target (CUSUM default)")
    best_a = None
    for eta in GRID_ETA:
        for d in GRID_D:
            rows, dt = score_config(search_names, ds, eta, DEFAULTS["CUSUM_KC"],
                                    DEFAULTS["CUSUM_HC_SIGMA_MULT"], d)
            pr = mean_metric(rows, "vus_pr")
            roc = mean_metric(rows, "vus_roc")
            trials.append({"round": "A", "HEDGE_ETA": eta, "d_target": d,
                           "CUSUM_KC": DEFAULTS["CUSUM_KC"],
                           "CUSUM_HC_SIGMA_MULT": DEFAULTS["CUSUM_HC_SIGMA_MULT"],
                           "vus_pr": pr, "vus_roc": roc, "elapsed": dt})
            tag = "*" if pr > base_pr else " "
            print(f"  {tag} eta={eta:<4} d={d:<3}  VUS-PR={pr:.4f} VUS-ROC={roc:.4f}  [{dt:.0f}s]")
            if best_a is None or pr > best_a["vus_pr"]:
                best_a = {"HEDGE_ETA": eta, "d_target": d, "vus_pr": pr, "vus_roc": roc}

    # ---- Round B: CUSUM_KC x CUSUM_HC_SIGMA_MULT (best eta/d fixed) ----
    print(f"\n[tuning] Round B: CUSUM_KC x CUSUM_HC_SIGMA_MULT (eta={best_a['HEDGE_ETA']}, d={best_a['d_target']})")
    best_b = None
    for kc in GRID_KC:
        for hcm in GRID_HCMULT:
            rows, dt = score_config(search_names, ds, best_a["HEDGE_ETA"], kc, hcm,
                                    best_a["d_target"])
            pr = mean_metric(rows, "vus_pr")
            roc = mean_metric(rows, "vus_roc")
            trials.append({"round": "B", "HEDGE_ETA": best_a["HEDGE_ETA"],
                           "d_target": best_a["d_target"], "CUSUM_KC": kc,
                           "CUSUM_HC_SIGMA_MULT": hcm, "vus_pr": pr, "vus_roc": roc,
                           "elapsed": dt})
            tag = "*" if pr > base_pr else " "
            print(f"  {tag} kc={kc:<5} hcmult={hcm:<4}  VUS-PR={pr:.4f} VUS-ROC={roc:.4f}  [{dt:.0f}s]")
            if best_b is None or pr > best_b["vus_pr"]:
                best_b = {"HEDGE_ETA": best_a["HEDGE_ETA"], "d_target": best_a["d_target"],
                          "CUSUM_KC": kc, "CUSUM_HC_SIGMA_MULT": hcm,
                          "vus_pr": pr, "vus_roc": roc}

    best = best_b
    print(f"\n[tuning] best config: {best}  (search VUS-PR {base_pr:.4f} -> {best['vus_pr']:.4f})")

    # ---- Validate best config on FULL 18-series subset ----
    print("\n[tuning] validating best config on full 18-series subset")
    val_rows, dt = score_config(names_full, ds, best["HEDGE_ETA"], best["CUSUM_KC"],
                                best["CUSUM_HC_SIGMA_MULT"], best["d_target"])
    val_pr = mean_metric(val_rows, "vus_pr")
    val_roc = mean_metric(val_rows, "vus_roc")
    # default-config full-subset reference (from cache)
    full_default_pr = mean_metric(raw_metrics, "vus_pr")
    full_default_roc = mean_metric(raw_metrics, "vus_roc")
    full_pers_pr = mean_metric(pers_metrics, "vus_pr")
    full_pers_roc = mean_metric(pers_metrics, "vus_roc")
    # win rate vs persistence for tuned vs default
    val_by_name = {r["name"]: r for r in val_rows}
    raw_by_name = {r["name"]: r for r in [{"name": n, **m} for n, m in zip(names_full, raw_metrics)]}
    pers_by_name = {r["name"]: r for r in [{"name": n, **m} for n, m in zip(names_full, pers_metrics)]}
    wins_tuned = sum(1 for n in names_full if val_by_name[n]["vus_pr"] > pers_by_name[n]["vus_pr"])
    wins_default = sum(1 for n in names_full if raw_by_name[n]["vus_pr"] > pers_by_name[n]["vus_pr"])

    print(f"  full subset: default VUS-PR={full_default_pr:.4f}  tuned VUS-PR={val_pr:.4f}  pers={full_pers_pr:.4f}")
    print(f"  full subset: default VUS-ROC={full_default_roc:.4f} tuned VUS-ROC={val_roc:.4f} pers={full_pers_roc:.4f}")
    print(f"  win-rate vs pers (VUS-PR): default {wins_default}/{len(names_full)}  tuned {wins_tuned}/{len(names_full)}")

    out = {
        "search_subset": search_names,
        "full_subset": names_full,
        "defaults": DEFAULTS,
        "grids": {"HEDGE_ETA": GRID_ETA, "d_target": GRID_D, "CUSUM_KC": GRID_KC,
                  "CUSUM_HC_SIGMA_MULT": GRID_HCMULT},
        "baseline_search": {"vus_pr": base_pr, "vus_roc": base_roc},
        "trials": trials,
        "best_config": best,
        "validation_full_subset": {
            "default_vus_pr": full_default_pr, "default_vus_roc": full_default_roc,
            "tuned_vus_pr": val_pr, "tuned_vus_roc": val_roc,
            "persistence_vus_pr": full_pers_pr, "persistence_vus_roc": full_pers_roc,
            "win_rate_vus_pr_default": wins_default, "win_rate_vus_pr_tuned": wins_tuned,
            "n_series": len(names_full),
            "per_series": val_rows,
        },
        "elapsed_seconds": time.perf_counter() - t0,
    }
    LOG.write_text(json.dumps(out, indent=2, default=float) + "\n")
    print(f"\n  -> {LOG}  (elapsed {out['elapsed_seconds']:.0f}s)")


if __name__ == "__main__":
    main()
