from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple

class DetectorABC(ABC):
    """
    Abstract Base Class for all 6 Detectors in the Detector Battery.
    Input contract: Z_t (compressed vector), v_t (normalized scalar)
    Output contract: (anomaly_score s_t in [0, 1], forecast v_hat_{t+h})
    """
    @abstractmethod
    def score(self, Z_t: np.ndarray, v_t: float) -> Tuple[float, float]:
        """
        Calculates instantaneous anomaly score s_t and forward forecast v_hat_{t+h}.
        """
        pass

    @abstractmethod
    def update(self, v_true: float):
        """
        Updates internal model state with newly observed true value v_t.
        """
        pass
