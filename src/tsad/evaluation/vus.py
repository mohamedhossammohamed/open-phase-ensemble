import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score


def apply_range_buffer(labels: np.ndarray, buffer_length: int) -> np.ndarray:
    """
    Applies temporal buffer of length `buffer_length` strictly to ground truth label regions.
    Expands positive binary label regions (1s) by buffer_length on both sides.
    Predicted scores must remain un-buffered.
    """
    if buffer_length <= 0:
        return labels.copy()
        
    n = len(labels)
    pos_indices = np.where(labels > 0)[0]
    if len(pos_indices) == 0:
        return labels.copy()
        
    buffered = labels.copy()
    for idx in pos_indices:
        start = max(0, idx - buffer_length)
        end = min(n, idx + buffer_length + 1)
        buffered[start:end] = 1
        
    return buffered

def compute_vus_roc(scores: np.ndarray, labels: np.ndarray, max_buffer: int = 20) -> float:
    """
    Volume Under Surface ROC (VUS-ROC) across continuous temporal buffer sizes l in [0, max_buffer].
    Range buffering is applied strictly to ground truth labels. Predicted scores remain un-buffered.
    """
    if len(scores) > 5000:
        step = max(1, len(scores) // 3000)
        scores = scores[::step]
        labels = labels[::step]
        
    auc_list = []
    buffer_steps = list(range(0, max_buffer + 1, max(1, max_buffer // 10)))
    
    for l in buffer_steps:
        lbl_l = apply_range_buffer(labels, buffer_length=l)
        
        if len(np.unique(lbl_l)) < 2:
            auc_l = 0.5
        else:
            auc_l = float(roc_auc_score(lbl_l, scores))
        auc_list.append(auc_l)
        
    return float(np.mean(auc_list))

def compute_vus_pr(scores: np.ndarray, labels: np.ndarray, max_buffer: int = 20) -> float:
    """
    Volume Under Surface PR (VUS-PR) across continuous temporal buffer sizes l in [0, max_buffer].
    Range buffering is applied strictly to ground truth labels. Predicted scores remain un-buffered.
    """
    if len(scores) > 5000:
        step = max(1, len(scores) // 3000)
        scores = scores[::step]
        labels = labels[::step]
        
    auc_list = []
    buffer_steps = list(range(0, max_buffer + 1, max(1, max_buffer // 10)))
    
    for l in buffer_steps:
        lbl_l = apply_range_buffer(labels, buffer_length=l)
        
        if len(np.unique(lbl_l)) < 2:
            auc_l = 0.0
        else:
            precision, recall, _ = precision_recall_curve(lbl_l, scores)
            auc_l = float(auc(recall, precision))
        auc_list.append(auc_l)
        
    return float(np.mean(auc_list))
