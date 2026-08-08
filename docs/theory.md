# Theoretical Foundations

!!! warning "Experimental Research Disclaimer"
    This project is experimental and provided for research and educational purposes only. All performance claims are preliminary, self-reported, and have not yet been independently validated or peer-reviewed. Use at your own risk.

---

## 1. Takens' Embedding Theorem & Phase Space Reconstruction
Floris Takens (1981) proved that the unobserved multi-dimensional attractor manifold $\mathcal{M}$ of a non-linear dynamical system can be reconstructed from a single observed scalar time series $v_t$ via delay coordinate vectors:

$$X_t = [v_t, v_{t-\tau}, v_{t-2\tau}, \dots, v_{t-(d-1)\tau}]^T \in \mathbb{R}^d$$

where $\tau$ is the time delay lag and $d$ is the embedding dimension. If $d > 2 m + 1$ (where $m$ is the manifold dimension), the delay embedding $\Phi: \mathcal{M} \to \mathbb{R}^d$ forms a smooth topological immersion preserving manifold geometry.

---

## 2. Parameter Selection: AMI and FNN

### Average Mutual Information (AMI)
Fraser & Swinney (1986) established that the optimal time delay $\tau$ corresponds to the **first local minimum** of the Average Mutual Information $I(\tau)$:

$$I(\tau) = \sum_{x, y} P(v_t, v_{t+\tau}) \log_2 \left( \frac{P(v_t, v_{t+\tau})}{P(v_t) P(v_{t+\tau})} \right)$$

This selects a lag $\tau$ where coordinates $v_t$ and $v_{t+\tau}$ are maximally independent in an information-theoretic sense.

### False Nearest Neighbors (FNN)
Kennel et al. (1992) introduced False Nearest Neighbors to determine the minimal embedding dimension $d$. Points near each other in dimension $d$ may be false projections due to geometric overlap. The fraction of FNNs is tracked as $d$ increases:

$$\frac{|v_{t+(d)\tau} - v_{nn+(d)\tau}|}{\|X_t^{(d)} - X_{nn}^{(d)}\|} > R_{tol}$$

The optimal $d$ is chosen where the FNN fraction drops below $1\%$.

---

## 3. Covariance Geometry: Ledoit-Wolf Shrinkage
When sample size $N$ is small relative to dimension $d$, sample covariance $\Sigma_{sample}$ is unstable. Ledoit & Wolf (2004) analytical shrinkage computes a well-conditioned estimator $\Sigma_{LW}$:

$$\Sigma_{LW} = (1 - \delta) \Sigma_{sample} + \delta \mu I, \quad \mu = \frac{\text{trace}(\Sigma_{sample})}{d}$$

where optimal shrinkage intensity $\delta \in [0, 1]$ minimizes expected Frobenius loss, ensuring $\Sigma_{LW}$ is positive-definite and invertible. Mahalanobis distance is computed as:

$$D_{mahal}(Z_t) = \sqrt{(Z_t - \mu_Z)^T \Sigma_{LW}^{-1} (Z_t - \mu_Z)}$$

---

## 4. Surrogate Null Models: AR and IAAFT
To test whether dynamic predictability exceeds linear autocorrelation noise, surrogate null series are generated:
- **Autoregressive (AR) Surrogates**: Linear stochastic process preserving spectral covariance.
- **Iterative Amplitude Adjusted Fourier Transform (IAAFT)**: Schreiber & Schmitz (1996) phase-randomization algorithm that exactly preserves both the empirical amplitude distribution and Fourier power spectrum of the original series.

---

## 5. Empirical Dynamic Modeling & Simplex Projection
Sugihara & May (1990) Simplex Projection performs non-parametric forecasting by tracking forward trajectories of $E+1$ nearest phase-space neighbors. Weights $w_i$ are exponentially decaying functions of distance:

$$w_i = \exp \left( -\frac{\|Z_t - Z_{nn_i}\|}{\|Z_t - Z_{nn_1}\| + \epsilon} \right), \quad \hat{v}_{t+h} = \sum_{i=1}^{E+1} \bar{w}_i v_{nn_i + h}$$

Anomaly score $s_t$ measures normalized prediction error $|v_t - \hat{v}_t|$.

---

## 6. Subsequence Motifs & STOMP Matrix Profile
Yeh et al. (2016) Matrix Profile computes the z-normalized Euclidean distance $z_i$ from every subsequence window $W$ of length $w_{mp}$ to its nearest non-overlapping historical neighbor. Peaks in the matrix profile represent structural discords (subsequence anomalies).

---

## 7. Subspace Isolation: Isolation Forest
Liu et al. (2008) Isolation Forest isolates anomalies by randomly partitioning features. Because anomalous points require fewer axis-aligned splits to isolate, path length $h(x)$ in ensemble trees $T$ measures anomaly severity:

$$s(x, n) = 2^{-\frac{\mathbb{E}(h(x))}{c(n)}}$$

---

## 8. Association Discrepancy: Anomaly Transformer
Xu et al. (2022) Anomaly Transformer evaluates **Association Discrepancy** between prior Gaussian association $P$ and learned self-attention series association $S$:

$$\text{AssocDiscrepancy}(X) = \frac{1}{L} \sum_{l=1}^L \text{KL}(P^{(l)} \| S^{(l)})$$

Anomalies exhibit low series association (hard to associate with distant points) compared to normal smooth regional transitions.

---

## 9. Meta-Judge Online Fusion: Hedge Algorithm
Freund & Schapire (1997) Hedge multiplicative weight update algorithm updates expert weights $w_{t, k}$ based on trailing correlation loss $\ell_{t, k} = 1 - \text{PearsonCorr}(S_k, E_k)$:

$$w_{t+1, k} = \frac{w_{t, k} e^{-\eta \ell_{t, k}}}{\sum_{j=1}^K w_{t, j} e^{-\eta \ell_{t, j}}}$$

Fixed-share mixing (Herbster & Warmuth, 1998) guarantees weight floor $w'_{k} \ge \frac{\sigma}{K}$ to adapt to sudden expert shifts ($\sigma = 0.01$).

---

## 10. Dimension Reduction & Search: JL & HNSW
- **Johnson-Lindenstrauss Lemma**: Random Gaussian projection $R \in \mathbb{R}^{d \times d_{target}}$ preserves pairwise Euclidean distances within $(1 \pm \epsilon)$ factor.
- **HNSW (Malkov & Yashunin, 2018)**: Hierarchical Navigable Small World graphs provide $O(\log N)$ approximate nearest neighbor queries in phase space.

---

## 11. Gating & Metrics: CUSUM and VUS

### CUSUM Change Detection
Page (1954) CUSUM chart tracks cumulative positive error shifts:

$$C_t^+ = \max(0, C_{t-1}^+ + E_t - (\mu_E + k_c \sigma_E))$$

When $C_t^+ > H_c$, model adaptation is frozen. Persistent alarms ($\ge T_{drift}$) trigger a baseline flush.

### Volume Under Surface (VUS-ROC & VUS-PR)
Paparrizos et al. (2022) VUS integrates ROC and Precision-Recall AUC curves across continuous temporal buffer thresholds $l \in [0, L_{max}]$, eliminating point-adjustment bias:

$$\text{VUS-ROC} = \frac{1}{L_{max} + 1} \sum_{l=0}^{L_{max}} \text{AUC-ROC}(\text{labels}_l, \text{scores})$$
