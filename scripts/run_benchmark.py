import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from tsad.pipeline import TSADPipeline
from tsad.evaluation.vus import compute_vus_roc, compute_vus_pr
from tsad.evaluation.iaaft import generate_iaaft_surrogate

def run_dataset_benchmark(name: str, signal: np.ndarray, labels: np.ndarray, max_points: int = 5000):
    if len(signal) > max_points:
        step = len(signal) // max_points
        signal = signal[::step]
        labels = labels[::step]
        
    print(f"\n--- Running Benchmark on {name} (Evaluated Points N={len(signal)}) ---")
    
    # 1. Run open-source TSAD System
    pipeline = TSADPipeline()
    scores = []
    for x in signal:
        A_t, _ = pipeline.step(x)
        scores.append(A_t)
        
    scores_arr = np.array(scores)
    vus_roc = compute_vus_roc(scores_arr, labels, max_buffer=15)
    vus_pr = compute_vus_pr(scores_arr, labels, max_buffer=15)
    
    print(f"System VUS-ROC: {vus_roc:.4f}")
    print(f"System VUS-PR:  {vus_pr:.4f}")
    
    # 2. Run IAAFT Surrogate Null Baseline
    print("Generating IAAFT surrogate null baseline...")
    surrogate_signal = generate_iaaft_surrogate(signal)
    
    surr_pipeline = TSADPipeline()
    surr_scores = []
    for x in surrogate_signal:
        A_t, _ = surr_pipeline.step(x)
        surr_scores.append(A_t)
        
    surr_scores_arr = np.array(surr_scores)
    surr_vus_roc = compute_vus_roc(surr_scores_arr, labels, max_buffer=15)
    
    edge = vus_roc - surr_vus_roc
    print(f"IAAFT Surrogate VUS-ROC: {surr_vus_roc:.4f}")
    print(f"Predictive Edge over Null: {edge:+.4f} (Target >= +0.30)")
    
    return {
        "dataset": name,
        "vus_roc": vus_roc,
        "vus_pr": vus_pr,
        "surrogate_vus_roc": surr_vus_roc,
        "predictive_edge": edge
    }

def main():
    raw_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/raw"))
    
    physio_file = os.path.join(raw_dir, "physionet/100.npz")
    if os.path.exists(physio_file):
        run_dataset_benchmark("PhysioNet MIT-BIH (Record 100)", data := np.load(physio_file)["signal"], np.load(physio_file)["labels"])
        
    cwru_file = os.path.join(raw_dir, "cwru/cwru_bearing.npz")
    if os.path.exists(cwru_file):
        run_dataset_benchmark("CWRU Bearing Dataset", data := np.load(cwru_file)["signal"], np.load(cwru_file)["labels"])

if __name__ == "__main__":
    main()
