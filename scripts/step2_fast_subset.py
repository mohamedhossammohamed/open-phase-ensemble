#!/usr/bin/env python3
"""Step 2 fast-subset experiment: scoring + POT adaptive thresholding.

Mirrors the exact eval protocol of tsad.benchmarks.base.BenchmarkRunner._run_series
(downsample 10000 -> train_split -> warmup 0.05 -> eval window) WITHOUT modifying
any core file. Caches per-series (scores, labels) so step 3 tuning and any
post-processing can reuse them.

Then applies POT (Peaks-Over-Threshold / GPD) adaptive thresholding as a
post-processing step on the cached scores and reports:
  - VUS-PR / VUS-ROC before/after  (ranking metrics -> monotonic-invariance control)
  - TSB-AD full opt-threshold suite before/after
  - Operating-point Precision/Recall/Standard-F1/PA-F1 at:
        default(0.5)  vs  naive-quantile(0.99)  vs  POT threshold
  - Win-rate shift vs persistence on VUS-PR / VUS-ROC
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

from tsad.benchmarks.tsb_ad import TSB_AD_U, _parse_filename  # noqa: E402
from tsad.benchmarks.base import _causal_block_downsample  # noqa: E402
from tsad.benchmarks.wrappers import TSADPipelineWrapper  # noqa: E402
from tsad.evaluation.vus import compute_vus_pr, compute_vus_roc  # noqa: E402
from tsad.config import SEED  # noqa: E402

OUT = ROOT / "results" / "benchmarks" / "step2_fastsubset"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = OUT / "fastsubset_cache.npz"
LOG = OUT / "step2_results.json"

WARMUP_FRAC = 0.05
MAX_BUFFER = 15
DOWNSAMPLE = 10000
N_PER_DOMAIN = 2  # stratified fast subset size


def domain_of(name: str) -> str:
    m = _parse_filename(name)
    return m["domain"]


def select_subset(ds: TSB_AD_U) -> list[str]:
    """Stratified: N_PER_DOMAIN shortest series (by row count) per domain."""
    by_dom: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for s in ds.iter_series("eval"):
        by_dom[domain_of(s.name)].append((len(s.signal), s.name))
    chosen: list[str] = []
    for dom in sorted(by_dom):
        rows = sorted(by_dom[dom])[:N_PER_DOMAIN]
        for _, name in rows:
            chosen.append(name)
    return chosen


def run_protocol(series, model):
    """Exact mirror of BenchmarkRunner._run_series (no core modification)."""
    signal = series.signal
    labels = series.labels
    train_split = series.train_split
    if DOWNSAMPLE and len(signal) > DOWNSAMPLE:
        signal, labels, stride = _causal_block_downsample(signal, labels, DOWNSAMPLE)
        train_split = max(1, int(train_split / stride))
    model.fit(signal[:train_split], None)
    scores = model.predict(signal)
    if not np.isfinite(scores).all():
        raise ValueError("non-finite scores")
    eval_signal = signal[train_split:]
    eval_labels = labels[train_split:]
    warmup = max(1, int(len(eval_signal) * WARMUP_FRAC))
    if warmup >= len(eval_signal):
        warmup = max(1, len(eval_signal) // 10)
    eval_scores = scores[train_split + warmup:]
    eval_labels_warm = eval_labels[warmup:]
    persistence = np.zeros(len(eval_signal), dtype=np.float64)
    if len(eval_signal) > 1:
        persistence[1:] = np.abs(np.diff(eval_signal))
    persistence = persistence[warmup:]
    return eval_scores, eval_labels_warm, persistence


def tsb_ad_metrics(scores, labels):
    try:
        from TSB_AD.evaluation.metrics import get_metrics
        out = get_metrics(scores, labels.astype(int), slidingWindow=MAX_BUFFER,
                          pred=None, version="opt", thre=250)
        return {f"tsb_ad_{k.lower().replace(' ', '_')}": float(v) for k, v in out.items()}
    except Exception as e:
        return {"tsb_ad_error": str(e)}


def primary_metrics(scores, labels):
    return {
        "vus_pr": compute_vus_pr(scores, labels, max_buffer=MAX_BUFFER),
        "vus_roc": compute_vus_roc(scores, labels, max_buffer=MAX_BUFFER),
    }


def point_adjusted_f1(pred: np.ndarray, labels: np.ndarray) -> dict:
    """Standard-F1 and PA-F1 (point-adjusted) at a fixed binary threshold."""
    pred = pred.astype(bool)
    labels = labels.astype(bool)
    tp = int(np.sum(pred & labels))
    fp = int(np.sum(pred & ~labels))
    fn = int(np.sum(~pred & labels))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    # Point-adjusted: if any point in a true anomaly segment is detected,
    # the whole segment counts as detected.
    pa_pred = pred.copy()
    if labels.any():
        # find contiguous true segments
        d = np.diff(labels.astype(int), prepend=0, append=0)
        starts = np.where(d == 1)[0]
        ends = np.where(d == -1)[0]
        for st, en in zip(starts, ends):
            if pred[st:en].any():
                pa_pred[st:en] = True
    pa_tp = int(np.sum(pa_pred & labels))
    pa_fp = int(np.sum(pa_pred & ~labels))
    pa_fn = int(np.sum(~pa_pred & labels))
    pa_p = pa_tp / (pa_tp + pa_fp) if (pa_tp + pa_fp) else 0.0
    pa_r = pa_tp / (pa_tp + pa_fn) if (pa_tp + pa_fn) else 0.0
    pa_f1 = 2 * pa_p * pa_r / (pa_p + pa_r) if (pa_p + pa_r) else 0.0
    return {"precision": precision, "recall": recall, "standard_f1": f1,
            "pa_precision": pa_p, "pa_recall": pa_r, "pa_f1": pa_f1,
            "n_flagged": int(pred.sum())}


def pot_threshold(scores: np.ndarray, q: float = 1e-3, init_quantile: float = 0.95):
    """POT/GPD adaptive threshold.

    Fit GPD to exceedances over the `init_quantile` empirical threshold,
    return the anomaly threshold z_q for tail probability q, plus transformed
    scores (monotonic tail-exceedance probability) for VUS-invariance check.
    """
    from scipy.stats import genpareto
    s = np.asarray(scores, dtype=np.float64)
    u = float(np.quantile(s, init_quantile))
    exceed = s[s > u] - u
    if len(exceed) < 10:
        # too few exceedances -> fall back to a high quantile
        z_q = float(np.quantile(s, 1 - q))
        return z_q, None, {"fallback": "too_few_exceedances", "n_exc": len(exceed), "u": u}
    try:
        # fit with floc=0 (exceedances are >=0)
        c, loc, scale = genpareto.fit(exceed, floc=0.0)
    except Exception as e:
        z_q = float(np.quantile(s, 1 - q))
        return z_q, None, {"fallback": f"fit_failed:{e}", "u": u}
    xi = float(c)
    sigma = float(scale)
    n = len(s)
    n_u = len(exceed)
    # POT threshold for tail probability q (Siffer et al. 2017 SPOT formula)
    if abs(xi) < 1e-8:
        z_q = u + sigma * np.log((n_u / n) / q)
    else:
        z_q = u + (sigma / xi) * (((n_u / n) / q) ** xi - 1.0)
    # monotonic transform: tail-exceedance probability (lower = more anomalous)
    # convert to "anomaly score" = 1 - P(X>s) so higher = more anomalous (monotonic in s)
    trans = np.empty_like(s)
    below = s <= u
    trans[below] = 1.0 - (n_u / n)  # constant mass below threshold
    above = ~below
    if abs(xi) < 1e-8:
        trans[above] = 1.0 - (n_u / n) * np.exp(-((s[above] - u) / sigma))
    else:
        trans[above] = 1.0 - (n_u / n) * (1.0 + xi * (s[above] - u) / sigma) ** (-1.0 / xi)
    trans = np.clip(trans, 0.0, 1.0)
    info = {"xi": xi, "sigma": sigma, "u": u, "n_exc": n_u, "z_q": float(z_q)}
    return float(z_q), trans, info


def main():
    t0 = time.perf_counter()
    ds = TSB_AD_U(data_root=ROOT / "data" / "benchmarks")
    if not ds.is_downloaded:
        raise SystemExit(f"TSB-AD-U not downloaded at {ds.csv_dir}")

    subset = select_subset(ds)
    print(f"[subset] {len(subset)} series: {subset}")

    # ---- Score the subset (cache for reuse by step 3) ----
    cache = {"names": [], "domains": [], "eval_scores": [], "eval_labels": [],
             "persistence": [], "raw_metrics": [], "pers_metrics": []}
    if CACHE.exists():
        print(f"[cache] loading {CACHE}")
        z = np.load(CACHE, allow_pickle=True)
        if list(z["names"]) == subset:
            cache = {k: z[k].tolist() if k in ("names", "domains") else z[k] for k in z.files}
            # recompute metrics not stored as arrays
            cache["raw_metrics"] = json.loads(str(z["raw_metrics_json"])) if "raw_metrics_json" in z.files else cache.get("raw_metrics", [])
            cache["pers_metrics"] = json.loads(str(z["pers_metrics_json"])) if "pers_metrics_json" in z.files else cache.get("pers_metrics", [])
        else:
            print("[cache] subset mismatch -> re-scoring")

    if not cache["eval_scores"]:
        model = TSADPipelineWrapper()
        raw_metrics, pers_metrics = [], []
        es_all, el_all, pe_all = [], [], []
        for name in subset:
            series = next(s for s in ds.iter_series("eval") if s.name == name)
            t1 = time.perf_counter()
            es, el, pe = run_protocol(series, model)
            dt = time.perf_counter() - t1
            rm = {**primary_metrics(es, el), **tsb_ad_metrics(es, el)}
            pm = {**primary_metrics(pe, el), **tsb_ad_metrics(pe, el)}
            raw_metrics.append(rm)
            pers_metrics.append(pm)
            es_all.append(es); el_all.append(el); pe_all.append(pe)
            cache["names"].append(name)
            cache["domains"].append(domain_of(name))
            print(f"  {name:55s} n={len(el):5d} pos={int(el.sum()):5d} "
                  f"VUS-PR={rm['vus_pr']:.4f} (pers {pm['vus_pr']:.4f})  [{dt:.1f}s]")
        cache["eval_scores"] = es_all
        cache["eval_labels"] = el_all
        cache["persistence"] = pe_all
        cache["raw_metrics"] = raw_metrics
        cache["pers_metrics"] = pers_metrics
        np.savez(CACHE,
                 names=np.array(cache["names"]), domains=np.array(cache["domains"]),
                 eval_scores=np.array(cache["eval_scores"], dtype=object),
                 eval_labels=np.array(cache["eval_labels"], dtype=object),
                 persistence=np.array(cache["persistence"], dtype=object),
                 raw_metrics_json=np.array(json.dumps(raw_metrics)),
                 pers_metrics_json=np.array(json.dumps(pers_metrics)))
        print(f"[cache] saved {CACHE}")

    # ---- POT adaptive thresholding (post-processing) ----
    results = []
    for i, name in enumerate(cache["names"]):
        es = np.asarray(cache["eval_scores"][i], dtype=np.float64)
        el = np.asarray(cache["eval_labels"][i], dtype=np.int8)
        pe = np.asarray(cache["persistence"][i], dtype=np.float64)
        rm = cache["raw_metrics"][i]
        pm = cache["pers_metrics"][i]

        z_q, trans, pot_info = pot_threshold(es, q=1e-3, init_quantile=0.95)
        default_thr = 0.5
        naive_thr = float(np.quantile(es, 0.99))

        op = {
            "default_0p5": point_adjusted_f1(es > default_thr, el),
            "naive_q99": point_adjusted_f1(es > naive_thr, el),
            "pot": point_adjusted_f1(es > z_q, el),
        }
        # VUS on POT-transformed (monotonic) scores -> invariance control
        vus_after = primary_metrics(trans if trans is not None else es, el)
        tsb_after = tsb_ad_metrics(trans if trans is not None else es, el) if trans is not None else {}

        results.append({
            "name": name, "domain": cache["domains"][i],
            "n_eval": int(len(el)), "n_positive": int(el.sum()),
            "raw": rm, "persistence": pm,
            "vus_after_pot": vus_after,
            "tsb_after_pot": tsb_after,
            "pot_info": pot_info,
            "thresholds": {"default": default_thr, "naive_q99": naive_thr, "pot_z_q": z_q},
            "operating_point": op,
        })

    # ---- Aggregate + win-rate ----
    def agg(key, src="raw"):
        vals = [r[src].get(key) for r in results if r[src].get(key) is not None]
        return float(np.mean(vals)) if vals else float("nan")

    def win_rate(key):
        w = sum(1 for r in results
                if r["raw"].get(key) is not None and r["persistence"].get(key) is not None
                and r["raw"][key] > r["persistence"][key])
        return w, len(results)

    def win_rate_after(key):
        w = sum(1 for r in results
                if r["vus_after_pot"].get(key) is not None and r["persistence"].get(key) is not None
                and r["vus_after_pot"][key] > r["persistence"][key])
        return w, len(results)

    summary = {
        "subset": cache["names"],
        "n_series": len(results),
        "config": {"warmup_fraction": WARMUP_FRAC, "max_buffer": MAX_BUFFER,
                   "downsample": DOWNSAMPLE, "n_per_domain": N_PER_DOMAIN, "seed": SEED},
        "elapsed_seconds": time.perf_counter() - t0,
        "aggregate_vus": {
            "vus_pr_raw": agg("vus_pr"), "vus_pr_after_pot": agg("vus_pr", "vus_after_pot"),
            "vus_pr_persistence": agg("vus_pr", "persistence"),
            "vus_roc_raw": agg("vus_roc"), "vus_roc_after_pot": agg("vus_roc", "vus_after_pot"),
            "vus_roc_persistence": agg("vus_roc", "persistence"),
        },
        "win_rate_vus_pr": {"raw_vs_pers": win_rate("vus_pr"), "pot_vs_pers": win_rate_after("vus_pr")},
        "win_rate_vus_roc": {"raw_vs_pers": win_rate("vus_roc"), "pot_vs_pers": win_rate_after("vus_roc")},
        "operating_point_mean": {
            m: {op: float(np.mean([r["operating_point"][op][m] for r in results]))
                for op in ["default_0p5", "naive_q99", "pot"]}
            for m in ["precision", "recall", "standard_f1", "pa_f1"]
        },
        "tsb_opt_mean": {
            k: {"raw": agg(k), "after_pot": agg(k, "vus_after_pot") if False else
                (float(np.mean([r["tsb_after_pot"].get(k) for r in results if r["tsb_after_pot"].get(k) is not None])) if any(r["tsb_after_pot"].get(k) is not None for r in results) else float("nan"))}
            for k in ["tsb_ad_vus-pr", "tsb_ad_vus-roc", "tsb_ad_standard-f1", "tsb_ad_pa-f1", "tsb_ad_event-based-f1"]
        },
    }
    # fix tsb after_pot agg
    for k in summary["tsb_opt_mean"]:
        vals = [r["tsb_after_pot"].get(k) for r in results if r["tsb_after_pot"].get(k) is not None]
        summary["tsb_opt_mean"][k]["after_pot"] = float(np.mean(vals)) if vals else float("nan")
        vals_raw = [r["raw"].get(k) for r in results if r["raw"].get(k) is not None]
        summary["tsb_opt_mean"][k]["raw"] = float(np.mean(vals_raw)) if vals_raw else float("nan")

    out = {"summary": summary, "per_series": results}
    LOG.write_text(json.dumps(out, indent=2, default=float) + "\n")
    print("\n=== SUMMARY ===")
    print(f"  VUS-PR  raw={summary['aggregate_vus']['vus_pr_raw']:.4f}  "
          f"pot={summary['aggregate_vus']['vus_pr_after_pot']:.4f}  "
          f"pers={summary['aggregate_vus']['vus_pr_persistence']:.4f}")
    print(f"  VUS-ROC raw={summary['aggregate_vus']['vus_roc_raw']:.4f}  "
          f"pot={summary['aggregate_vus']['vus_roc_after_pot']:.4f}  "
          f"pers={summary['aggregate_vus']['vus_roc_persistence']:.4f}")
    print(f"  win VUS-PR raw/pers={summary['win_rate_vus_pr']['raw_vs_pers']}  "
          f"pot/pers={summary['win_rate_vus_pr']['pot_vs_pers']}")
    opm = summary["operating_point_mean"]
    print("  Operating-point (mean):")
    for m in ["precision", "recall", "standard_f1", "pa_f1"]:
        print(f"    {m:12s} default={opm[m]['default_0p5']:.4f}  q99={opm[m]['naive_q99']:.4f}  pot={opm[m]['pot']:.4f}")
    print(f"\n  Results -> {LOG}")
    print(f"  Cache   -> {CACHE}")
    print(f"  Elapsed {summary['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()
