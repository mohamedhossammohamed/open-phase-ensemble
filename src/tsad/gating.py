from enum import Enum

import numpy as np

from tsad.config import CUSUM_HC_SIGMA_MULT, CUSUM_KC, EPSILON, T_DRIFT


class GatingState(Enum):
    NORMAL = "normal"
    ANOMALY_ALARM = "anomaly_alarm"
    CONCEPT_DRIFT = "concept_drift"

class CUSUMGating:
    """
    Anomaly vs. Concept Drift Gating Module using CUSUM control chart.
    Distinguishes transient anomalies (freeze adaptation) from permanent concept drift (baseline flush).
    """
    def __init__(self, k_c: float = CUSUM_KC, h_c_mult: float = CUSUM_HC_SIGMA_MULT, t_drift: int = T_DRIFT):
        self.k_c = k_c
        self.h_c_mult = h_c_mult
        self.t_drift = t_drift
        
        self.c_plus = 0.0
        self.c_minus = 0.0
        self.alarm_counter = 0
        
        self.error_buffer: list[float] = []
        self.mu_E = 0.0
        self.sigma_E = 1.0
        self.last_reference_mean = 0.0
        self.last_reference_sigma = 1.0
        self.current_state = GatingState.NORMAL

    def step(self, error: float) -> GatingState:
        """
        Updates CUSUM statistics with incoming prediction error E_t.
        Returns current GatingState.
        """
        if len(self.error_buffer) >= 5:
            arr = np.array(self.error_buffer, dtype=np.float64)
            reference_mean = float(np.mean(arr))
            reference_sigma = float(np.std(arr)) + EPSILON
        else:
            reference_mean = self.mu_E
            reference_sigma = self.sigma_E

        self.last_reference_mean = reference_mean
        self.last_reference_sigma = reference_sigma

        # Recursive CUSUM calculation
        shift_plus = error - (reference_mean + self.k_c * reference_sigma)
        shift_minus = (reference_mean - self.k_c * reference_sigma) - error
        self.c_plus = max(0.0, self.c_plus + shift_plus)
        self.c_minus = max(0.0, self.c_minus + shift_minus)

        h_c = self.h_c_mult * reference_sigma
        
        if self.c_plus > h_c or self.c_minus > h_c:
            self.alarm_counter += 1
            if self.alarm_counter >= self.t_drift:
                # Permanent concept drift confirmed -> total baseline flush
                self.current_state = GatingState.CONCEPT_DRIFT
                self.reset_baselines()
            else:
                self.current_state = GatingState.ANOMALY_ALARM
        else:
            self.alarm_counter = 0
            self.c_plus = 0.0
            self.c_minus = 0.0
            self.current_state = GatingState.NORMAL

        self.error_buffer.append(error)
        if len(self.error_buffer) > 1000:
            self.error_buffer.pop(0)
        if len(self.error_buffer) >= 5:
            arr = np.array(self.error_buffer, dtype=np.float64)
            self.mu_E = float(np.mean(arr))
            self.sigma_E = float(np.std(arr)) + EPSILON
            
        return self.current_state

    def reset_baselines(self):
        """Flushes CUSUM statistics on confirmed concept drift."""
        self.c_plus = 0.0
        self.c_minus = 0.0
        self.alarm_counter = 0
        if len(self.error_buffer) > 0:
            last_err = self.error_buffer[-1]
            self.error_buffer = [last_err]
            self.mu_E = last_err
            self.sigma_E = 1.0

    def is_adaptation_allowed(self) -> bool:
        """Returns True if model weight adaptation is allowed (NORMAL or CONCEPT_DRIFT)."""
        return self.current_state != GatingState.ANOMALY_ALARM
