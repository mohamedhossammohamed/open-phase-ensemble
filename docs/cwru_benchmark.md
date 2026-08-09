# CWRU Streaming Fault-Onset Benchmark Synthesis Design

## 1. Overview & Motivation

The original CWRU benchmark preparation in `data/download.py` concatenated a healthy baseline `.mat` recording and a faulty `.mat` recording directly, accompanied by a binary step-function label (0 for healthy, 1 for faulty).

Direct concatenation introduces a phase step-discontinuity at the seam. Causal streaming detectors (e.g. prediction-error or phase-manifold based models) react to this synthetic edge as a high-frequency impulse artifact rather than a physical bearing fault onset. 

To create a physically defensible benchmark, the data pipeline now uses **Short-Time Fourier Transform (STFT) cross-fade synthesis** and **gradual ground-truth labeling**.

---

## 2. Technical Implementation

### 2.1 STFT Cross-Fade Synthesis

Instead of direct splicing:
1. The end of the healthy signal ($x_H$) and start of the faulty signal ($x_F$) of length $N_{\text{trans}}$ are extracted.
2. STFT complex spectra $Z_H, Z_F$ are computed with `nperseg=256` (≈21.3 ms at 12 kHz) and `noverlap=192` (75% overlap).
3. STFT magnitude is interpolated linearly frame-by-frame:
   $$M_{\text{interp}}(f, k) = (1 - w_k) \cdot |Z_H(f, k)| + w_k \cdot |Z_F(f, k)|$$
   where $w_k \in [0, 1]$ ramps linearly across the transition window.
4. Healthy phase $\Phi_H = \angle Z_H$ is preserved across the transition:
   $$Z_{\text{interp}} = M_{\text{interp}} \cdot e^{i \Phi_H}$$
   This preserves phase continuity across the seam and prevents phase-cancellation dropouts.
5. Inverse STFT (`istft`) reconstructs the time-domain transition waveform, which is concatenated:
   $$\text{Signal} = [x_{H, \text{pre}}] + [x_{\text{trans}}] + [x_{F, \text{post}}]$$

### 2.2 Gradual Ground-Truth Labeling

Labels are stored as `float32` arrays in $[0.0, 1.0]$:
- Healthy baseline: `0.0`
- Transition window: linear ramp from `0.0` to `1.0` matching the STFT interpolation weight $w_k$
- Fault segment: `1.0`

Manifests explicitly declare `label_semantics` and set `label_dtype: float32`.

### 2.3 Transition Window Length Choice

- **Default Value**: `4096` samples (configurable via `--transition-samples`).
- **Rationale**: At CWRU's 12 kHz sampling rate, 4096 samples corresponds to $\approx 0.341$ seconds. At 1797 RPM (0 HP condition), the shaft completes $\approx 5.7$ revolutions in 0.341 s. This window allows multiple characteristic impact periods (e.g., BPFI, BPFO, BSF) to emerge progressively without producing an artificial transient spike.

### 2.4 Baseline Volatility & Cross-Domain Load Testing

- **Baseline Volatility**: Healthy bearing signals exhibit baseline RMS fluctuations due to shaft speed variations and sensor noise (~2–5× lower amplitude than severe fault signals). Magnitude-domain STFT blending smoothly scales spectral energy without distorting the noise floor.
- **Cross-Domain Load Testing**: Supported via `--load-baseline` and `--load-fault` flags. Synthesizing a stream where healthy data comes from a 0 HP condition and faulty data from a 2 HP condition tests whether a detector false-alarms on load/speed changes alone.
- **Recording-Level Train/Eval Separation**: The healthy signal used for detector warm-up / manifold fitting must be sourced from a separate `.mat` recording (e.g. `98_Normal_1.mat`) than the healthy segment embedded in the test stream.

---

## 3. Usage Commands

```bash
# Generate CWRU streams for all available local .mat files (default 4096 transition samples)
python data/download.py --cwru-all --transition-samples 4096

# Generate a cross-load condition stream (0 HP healthy baseline to 2 HP fault segment)
python data/download.py --cwru-healthy data/raw/cwru/97_Normal_0.mat --cwru-faulty data/raw/cwru/282_B007_0.mat --load-baseline 0hp --load-fault 2hp

# Run stratified CWRU benchmark evaluation across 10 seeds
python scripts/run_benchmark.py --run-cwru --seeds 0-9
```
