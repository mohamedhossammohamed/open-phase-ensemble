"""
Global configuration constants and parameters for TSAD system.
All parameters strictly adhere to the Architectural Blueprint specification.
"""

# Deterministic execution
SEED = 42

# Representation Layer Parameters
MAX_TAU = 100
MAX_D = 20
D_TARGET = 8
R_TOL = 10.0
A_TOL = 2.0
FNN_THRESHOLD = 0.01

# Meta-Judge (Hedge Algorithm) Parameters
K_DETECTORS = 6
HEDGE_ETA = 0.1
FIXED_SHARE_SIGMA = 0.01
REPLAY_BUFFER_SIZE = 10_000
W_CORR = 100

# Gating & Hyperparameter Tuning Parameters
CUSUM_KC = 0.5
CUSUM_HC_SIGMA_MULT = 5.0
T_DRIFT = 200
HYPER_EVAL_INTERVAL = 5000

# General Numerical Stability Constant
EPSILON = 1e-6
