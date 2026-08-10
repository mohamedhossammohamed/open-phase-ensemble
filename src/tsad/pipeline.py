
import numpy as np

from tsad.config import (
    CUSUM_HC_SIGMA_MULT,
    CUSUM_KC,
    D_TARGET,
    FIXED_SHARE_SIGMA,
    HEDGE_ETA,
    K_DETECTORS,
    REPLAY_BUFFER_SIZE,
    T_DRIFT,
)
from tsad.detectors.iforest import IsolationForestDetector
from tsad.detectors.mahalanobis import RobustMahalanobisDetector
from tsad.detectors.matrix_profile import MatrixProfileDetector
from tsad.detectors.sarima import ARFilterDetector
from tsad.detectors.simplex import SimplexProjectionDetector
from tsad.detectors.transformer import MSETransformerAutoencoder
from tsad.gating import CUSUMGating
from tsad.ingestion import StreamBuffer
from tsad.learning_loop import OnlineLearningLoop
from tsad.meta_judge import MetaJudge, StratifiedReplayBuffer
from tsad.representation import (
    compress_projection,
    delay_embed,
)


class TSADPipeline:
    """
    Continuous Directed Acyclic Graph (DAG) Pipeline Orchestrator.
    Executes all 6 architectural modules in a strict online streaming loop.
    Input contract: singular raw scalar x_t
    Output contract: (fused anomaly score A_t in [0,1], fused forecast v_hat*)
    """
    def __init__(self, tau: int = 2, d: int = 8, d_target: int = D_TARGET):
        self.tau = tau
        self.d = d
        self.d_target = d_target
        
        # Module 1: Ingestion (Standardizes raw scalar stream x_t -> v_t)
        self.ingestion = StreamBuffer(window_size=200)
        
        # Module 3: Detector Battery (6 experts)
        self.detectors = [
            SimplexProjectionDetector(dim=d_target, tau=tau),
            RobustMahalanobisDetector(dim=d_target),
            MatrixProfileDetector(w_mp=max(5, tau * d)),
            IsolationForestDetector(dim=d_target),
            ARFilterDetector(),
            MSETransformerAutoencoder(dim=d_target)
        ]
        
        # Module 4: Meta-Judge & Replay Buffer
        self.meta_judge = MetaJudge(k_detectors=K_DETECTORS, eta=HEDGE_ETA, sigma=FIXED_SHARE_SIGMA)
        # NOTE: StratifiedReplayBuffer is currently write-only — add() is called
        # every step but sample() is never invoked in the live pipeline. Retained
        # for future replay-based training; does not affect any output or score.
        self.replay_buffer = StratifiedReplayBuffer(capacity=REPLAY_BUFFER_SIZE)
        
        # Module 5: Online Learning Loop
        self.learning_loop = OnlineLearningLoop(k_detectors=K_DETECTORS)
        
        # Module 6: CUSUM Gating
        self.gating = CUSUMGating(k_c=CUSUM_KC, h_c_mult=CUSUM_HC_SIGMA_MULT, t_drift=T_DRIFT)
        
        self.step_count = 0
        self.last_forecasts = np.zeros(K_DETECTORS, dtype=np.float64)
        self.last_scores = np.zeros(K_DETECTORS, dtype=np.float64)

    def step(self, x_t: float) -> tuple[float, float]:
        """
        Processes singular raw scalar observation at time t.
        Strict online execution guarantee: no future data exposed.
        """
        self.step_count += 1
        
        # 1. Ingestion & Preprocessing
        v_t = self.ingestion.step(x_t)
        
        # 2. Representation Layer
        max_lag = (self.d - 1) * self.tau
        
        # Only the newest delay window can affect the newest row.  Keeping the
        # full history here adds repeated array/window work without changing
        # the causal representation.
        embed_buf = self.ingestion.get_standardized_tail(max_lag + 1)
        if len(embed_buf) > max_lag:
            X_mat = delay_embed(embed_buf, tau=self.tau, d=self.d)
            X_t = X_mat[-1]
        else:
            X_t = np.ones(self.d, dtype=np.float64) * v_t
            
        Z_t = compress_projection(X_t.reshape(1, -1), target_d=self.d_target)[0]
        
        # 3. Online Learning Loop: update weights from previous step's forecast error
        if self.step_count > 1:
            loss_vector = self.learning_loop.step(v_t, self.last_forecasts, self.last_scores)
            
            if self.gating.is_adaptation_allowed():
                self.meta_judge.update_weights(loss_vector)
                self.replay_buffer.add(Z_t, float(np.dot(self.meta_judge.weights, self.last_scores)))
                
        # 4. Detector Battery Execution
        scores = np.zeros(K_DETECTORS, dtype=np.float64)
        forecasts = np.zeros(K_DETECTORS, dtype=np.float64)
        
        for k, det in enumerate(self.detectors):
            s_k, v_hat_k = det.score(Z_t, v_t)
            det.update(v_t)

            if hasattr(det, "add_vector"):
                det.add_vector(Z_t)

            # Defense-in-depth NaN/inf guards. StreamBuffer.step forward-fills
            # NaN/inf at ingestion, so under normal operation these never fire.
            # They are retained so a detector returning NaN/inf cannot poison the
            # fused score, the learning loop, or the gating module.
            scores[k] = float(np.nan_to_num(s_k, nan=0.0, posinf=1.0, neginf=0.0))
            forecasts[k] = float(np.nan_to_num(v_hat_k, nan=v_t, posinf=v_t, neginf=v_t))

        self.last_scores = scores
        self.last_forecasts = forecasts

        # 5. Meta-Judge Fusion
        A_t, v_hat_star = self.meta_judge.fuse(scores, forecasts)
        A_t = float(np.clip(np.nan_to_num(A_t, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0))
        v_hat_star = float(np.nan_to_num(v_hat_star, nan=v_t, posinf=v_t, neginf=v_t))
        
        # 6. Gating update based on global forecast error
        e_t = abs(v_t - v_hat_star)
        self.gating.step(e_t)
        
        return A_t, v_hat_star
