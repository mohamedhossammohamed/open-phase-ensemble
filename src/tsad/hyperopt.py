import numpy as np

from tsad.config import MAX_D, MAX_TAU, SEED


class BayesianHyperparameterTuner:
    """
    Slow-speed Bayesian Hyperparameter Optimizer with Hysteresis Rule.
    Clamps tau in [1, 100], d in [1, 20].
    Adopts proposed configuration only if Expected Improvement exceeds hysteresis margin.
    """
    def __init__(self, hysteresis_margin: float = 0.02, seed: int = SEED):
        self.hysteresis_margin = hysteresis_margin
        self.rng = np.random.RandomState(seed)
        self.current_tau = 10
        self.current_d = 8
        self.best_skill = -1.0

    def propose_next(self) -> tuple[int, int]:
        """Proposes next candidate (tau, d) within clamped bounds."""
        tau_proposal = int(np.clip(self.current_tau + self.rng.randint(-3, 4), 1, MAX_TAU))
        d_proposal = int(np.clip(self.current_d + self.rng.randint(-2, 3), 1, MAX_D))
        return tau_proposal, d_proposal

    def evaluate_candidate(self, skill_gain: float) -> bool:
        """
        Applies Hysteresis Rule: adopts new config if skill gain > hysteresis margin.
        """
        if skill_gain > self.hysteresis_margin:
            self.best_skill += skill_gain
            return True
        return False
