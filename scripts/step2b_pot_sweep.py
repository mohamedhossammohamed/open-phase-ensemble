#!/usr/bin/env python3
"""Step 2b: POT sensitivity sweep over q and init_quantile (uses cached scores, no rerun)."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from tsad.evaluation.vus import compute_vus_pr, compute_vus_roc  # noqa: E402
from step2_fast_subset import pot_threshold, point_adjusted_f1, primary_metrics  # noqa: E402

CACHE = ROOT / "results/benchmarks/step2_fastsubset/fastsubset_cache.npz"
OUT = ROOT / "results/benchmarks/step2_fastsubset/step2b_pot_sweep.json"

z = np.load(CACHE, allow_pickle=True)
names = z["names"].tolist()
es_all = z["eval_scores"]
el_all = z["eval_labels"]
pe_all = z["persistence"]
raw_metrics = json.loads(str(z["raw_metrics_json"]))
pers_metrics = json.loads(str(z["pers_metrics_json"]))

QS = [1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
INITS = [0.90, 0.95, 0.99]

def agg_op(op_key, metric, results):
    return float(np.mean([r["operating_point"][op_key][metric] for r in results]))

sweep = []
for init_q in INITS:
    for q in QS:
        results = []
        for i in range(len(names)):
            es = np.asarray(es_all[i], dtype=np.float64)
            el = np.asarray(el_all[i], dtype=np.int8)
            z_q, trans, info = pot_threshold(es, q=q, init_quantile=init_q)
            op = {"pot": point_adjusted_f1(es > z_q, el)}
            vus = primary_metrics(trans if trans is not None else es, el)
            results.append({"operating_point": op, "vus": vus, "info": info})
        vus_pr = float(np.mean([r["vus"]["vus_pr"] for r in results]))
        vus_roc = float(np.mean([r["vus"]["vus_roc"] for r in results]))
        row = {
            "init_quantile": init_q, "q": q,
            "vus_pr": vus_pr, "vus_roc": vus_roc,
            "pot_precision": agg_op("pot", "precision", results),
            "pot_recall": agg_op("pot", "recall", results),
            "pot_standard_f1": agg_op("pot", "standard_f1", results),
            "pot_pa_f1": agg_op("pot", "pa_f1", results),
            "n_fallback": sum(1 for r in results if "fallback" in (r["info"] or {})),
        }
        sweep.append(row)
        print(f"  init={init_q:.2f} q={q:<6g}  VUS-PR={vus_pr:.4f} VUS-ROC={vus_roc:.4f}  "
              f"prec={row['pot_precision']:.3f} rec={row['pot_recall']:.3f} "
              F"f1={row['pot_standard_f1']:.3f} pa_f1={row['pot_pa_f1']:.3f}  fb={row['n_fallback']}")

# reference: raw + naive q99 + persistence
raw_vus_pr = float(np.mean([m["vus_pr"] for m in raw_metrics]))
raw_vus_roc = float(np.mean([m["vus_roc"] for m in raw_metrics]))
pers_vus_pr = float(np.mean([m["vus_pr"] for m in pers_metrics]))
naive = []
for i in range(len(names)):
    es = np.asarray(es_all[i], dtype=np.float64); el = np.asarray(el_all[i], dtype=np.int8)
    naive.append(point_adjusted_f1(es > float(np.quantile(es, 0.99)), el))
naive_pa = float(np.mean([r["pa_f1"] for r in naive]))
naive_f1 = float(np.mean([r["standard_f1"] for r in naive]))

out = {
    "reference": {"raw_vus_pr": raw_vus_pr, "raw_vus_roc": raw_vus_roc,
                  "pers_vus_pr": pers_vus_pr, "naive_q99_pa_f1": naive_pa, "naive_q99_f1": naive_f1},
    "sweep": sweep,
}
OUT.write_text(json.dumps(out, indent=2, default=float) + "\n")
print(f"\n  reference: raw VUS-PR={raw_vus_pr:.4f} ROC={raw_vus_roc:.4f} | pers VUS-PR={pers_vus_pr:.4f} | naive_q99 pa_f1={naive_pa:.4f} f1={naive_f1:.4f}")
print(f"  -> {OUT}")
