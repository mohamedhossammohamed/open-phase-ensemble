import numpy as np

class QuantileScoreCalibrator:
    """
    eCDF Quantile Mapping for preserving anomaly score comparability across configuration changes.
    """
    def __init__(self):
        self.old_ecdf_x = None
        self.old_ecdf_y = None
        self.new_ecdf_x = None
        self.new_ecdf_y = None
        self.is_fitted = False

    def fit(self, old_scores: np.ndarray, new_scores: np.ndarray):
        """Builds empirical CDF mapping between old and new score distributions."""
        self.old_ecdf_x = np.sort(old_scores)
        self.old_ecdf_y = np.linspace(0.0, 1.0, len(old_scores))
        
        self.new_ecdf_x = np.sort(new_scores)
        self.new_ecdf_y = np.linspace(0.0, 1.0, len(new_scores))
        self.is_fitted = True

    def calibrate(self, raw_score: float) -> float:
        """
        Maps raw new score back to equivalent old score percentile severity.
        """
        if not self.is_fitted:
            return float(np.clip(raw_score, 0.0, 1.0))
            
        # Find percentile rank of raw_score in new distribution
        percentile = float(np.interp(raw_score, self.new_ecdf_x, self.new_ecdf_y))
        
        # Map percentile to value in old distribution
        calibrated_score = float(np.interp(percentile, self.old_ecdf_y, self.old_ecdf_x))
        return float(np.clip(calibrated_score, 0.0, 1.0))
