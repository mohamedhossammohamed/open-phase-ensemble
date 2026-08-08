import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

def apply_range_buffer(arr: np.ndarray, buffer_length: int) -> np.ndarray:
    """
    Applies temporal buffer of length `buffer_length` around positive entries.
    Expands positive values by buffer_length on both sides.
    """
    if buffer_length <= 0:
        return arr.copy()
        
    n = len(arr)
    pos_indices = np.where(arr > 0)[0]
    if len(pos_indices) == 0:
        return arr.copy()
        
    buffered = arr.copy()
    for idx in pos_indices:
        start = max(0, idx - buffer_length)
        end = min(n, idx + buffer_length + 1)
        buffered[start:end] = np.maximum(buffered[start:end], arr[idx])
        
    return buffered

def compute_vus_roc(scores: np.ndarray, labels: np.ndarray, max_buffer: int = 20) -> float:
    """
    Volume Under Surface ROC (VUS-ROC) across continuous spectrum of buffer lengths l in [0, max_buffer].
    VUS-ROC = 1 / (max_buffer + 1) * sum_{l=0}^{max_buffer} AUC_ROC(labels_l, scores_l)
    """
    if len(scores) > 5000:
        step = max(1, len(scores) // 3000)
        scores = scores[::step]
        labels = labels[::step]
        
    auc_list = []
    buffer_steps = list(range(0, max_buffer + 1, max(1, max_buffer // 10)))
    
    for l in buffer_steps:
        lbl_l = apply_range_buffer(labels, buffer_length=l)
        scr_l = apply_range_buffer(scores, buffer_length=l)
        
        if len(np.unique(lbl_l)) < 2:
            auc_l = 0.5
        else:
            auc_l = float(roc_auc_score(lbl_l, scr_l))
        auc_list.append(auc_l)
        
    return float(np.mean(auc_list))

def compute_vus_pr(scores: np.ndarray, labels: np.ndarray, max_buffer: int = 20) -> float:
    """
    Volume Under Surface PR (VUS-PR) across continuous spectrum of buffer lengths l in [0, max_buffer].
    """
    if len(scores) > 5000:
        step = max(1, len(scores) // 3000)
        scores = scores[::step]
        labels = labels[::step]
        
    auc_list = []
    buffer_steps = list(range(0, max_buffer + 1, max(1, max_buffer // 10)))
    
    for l in buffer_steps:
        lbl_l = apply_range_buffer(labels, buffer_length=l)
        scr_l = apply_range_buffer(scores, buffer_length=l)
        
        if len(np.unique(lbl_l)) < 2:
            auc_l = 0.0
        else:
            precision, recall, _ = precision_recall_curve(lbl_l, scr_l)
            auc_l = float(auc(recall, precision))
        auc_list.append(auc_l)
        
    return float(np.mean(auc_list))
