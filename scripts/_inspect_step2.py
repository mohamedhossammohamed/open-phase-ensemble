#!/usr/bin/env python3
"""One-off inspection: verify data, TSB-AD install, cached scores, domains."""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

EVAL_JSON = ROOT / "results/benchmarks/full/TSB-AD-U_eval_20260810_045853.json"

# 1. Data availability
csv_dir = ROOT / "data/benchmarks/TSB-AD-U"
split_dir = ROOT / "data/benchmarks/TSB-AD-splits"
csvs = sorted(csv_dir.glob("*.csv")) if csv_dir.is_dir() else []
print(f"[data] TSB-AD-U csv dir exists: {csv_dir.is_dir()}, n_csv={len(csvs)}")
print(f"[data] split dir exists: {split_dir.is_dir()}")
for sp in ["TSB-AD-U-Eva.csv", "TSB-AD-U-Tuning.csv"]:
    print(f"[data] split {sp}: {(split_dir / sp).exists()}")

# 2. TSB-AD install
try:
    from TSB_AD.evaluation.metrics import get_metrics  # noqa: F401
    print("[tsb_ad] installed: True")
except Exception as e:
    print(f"[tsb_ad] installed: False ({type(e).__name__}: {e})")

# 3. Cached scores
d = json.load(open(EVAL_JSON))
sr = d["series_results"]
print(f"[cache] n_series={len(sr)}")
has_scores = sum(1 for r in sr if r.get("raw_scores"))
print(f"[cache] series with raw_scores: {has_scores}/{len(sr)}")
if sr:
    r0 = sr[0]
    print(f"[cache] sr[0] name={r0['name']}")
    print(f"[cache] sr[0] keys={list(r0.keys())}")
    rs = r0.get("raw_scores")
    print(f"[cache] sr[0] raw_scores len={len(rs) if rs else None}")
    print(f"[cache] sr[0] n_eval={r0['n_eval']} n_positive={r0['n_positive']}")
    print(f"[cache] sr[0] metrics keys={list(r0['metrics'].keys())}")
    print(f"[cache] sr[0] baselines keys={list(r0['baselines'].keys())}")

# 4. Domains + per-domain counts and mean VUS-PR (model vs persistence)
def domain(name):
    # parse domain from filename: <idx>_<dataset>_id_<id>_<domain>_tr_...
    parts = name.split("_")
    # find 'tr' token index, domain is just before it
    try:
        ti = parts.index("tr")
        return parts[ti - 1]
    except ValueError:
        return "unknown"

by_dom = defaultdict(list)
for r in sr:
    by_dom[domain(r["name"])].append(r)

print("\n[domains] per-domain: count, mean VUS-PR (model), mean VUS-PR (pers), mean VUS-ROC (model)")
for dom in sorted(by_dom):
    rows = by_dom[dom]
    mpr = sum(r["metrics"].get("vus_pr", 0.0) for r in rows) / len(rows)
    ppr = sum(r["baselines"].get("persistence", {}).get("vus_pr", 0.0) for r in rows) / len(rows)
    mroc = sum(r["metrics"].get("vus_roc", 0.0) for r in rows) / len(rows)
    # win rate VUS-PR
    wins = sum(1 for r in rows if r["metrics"].get("vus_pr", 0.0) > r["baselines"].get("persistence", {}).get("vus_pr", 0.0))
    print(f"  {dom:20s} n={len(rows):3d}  VUS-PR m={mpr:.4f} p={ppr:.4f}  VUS-ROC m={mroc:.4f}  win_pr={wins}/{len(rows)}")

# 5. Pick a stratified fast subset: a few series per domain, prefer short series (fast).
print("\n[subset] candidate fast subset (shortest 2 per domain, capped at ~20):")
import re
PAT = re.compile(r"^(?P<index>\d+)_(?P<dataset>[A-Za-z0-9_]+?)_id_(?P<id>\d+)_(?P<domain>[A-Za-z]+)_tr_(?P<tr>\d+)_1st_(?P<first>\d+)\.csv$")

# Need series lengths; load from cache n_eval is post-warmup. Use raw_scores length as proxy for full length.
subset = []
for dom in sorted(by_dom):
    rows = sorted(by_dom[dom], key=lambda r: len(r.get("raw_scores") or []))
    for r in rows[:2]:
        subset.append((dom, r["name"], len(r.get("raw_scores") or []), r["n_eval"]))
for dom, name, nfull, neval in subset:
    print(f"  {dom:20s} {name}  n_full={nfull} n_eval={neval}")
print(f"[subset] total candidates: {len(subset)}")
