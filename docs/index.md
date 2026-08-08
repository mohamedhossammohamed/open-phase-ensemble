# open-phase-ensemble

An open-source, non-parametric, multi-tool ensemble for streaming time-series anomaly detection and forecasting.

---

!!! note "Preliminary Research — Pending Independent Review"
    This project is experimental and provided for research and educational purposes only. All performance claims are preliminary and self-reported. Do not use for safety-critical, medical, financial, or production decisions without independent expert review. See the [full disclaimer](disclaimer.md).

---

## ⚡ Live Stream Simulation

<div class="card-formal" style="margin: 1.5rem 0; padding: 1rem;">
  <canvas id="hero-canvas" style="width: 100%; height: 320px; border-radius: 8px; background: rgba(0,0,0,0.4); display: block;"></canvas>
  <div style="margin-top: 0.75rem; display: flex; justify-content: space-between; align-items: center; font-family: var(--md-code-font); font-size: 0.75rem; color: var(--md-default-fg-color--muted);">
    <span>1-D Stream Signal → Phase-Space Manifold (Live Dynamic Mapping)</span>
    <span style="color: #2dd4bf; font-weight: 600;">● Live Stream</span>
  </div>
</div>

<script>
(function() {
  const canvas = document.getElementById('hero-canvas');
  if (!canvas) return;
  
  function resizeCanvas() {
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);
    return { ctx, width: rect.width, height: rect.height };
  }

  let state = resizeCanvas();
  window.addEventListener('resize', () => { if (canvas) state = resizeCanvas(); });

  let buffer = [];
  let t = 0;

  function signal(x) {
    const p = (((x % 60) / 60) + 1) % 1;
    const g = (a, c, w) => {
      let d = Math.abs(a - c);
      d = Math.min(d, 1 - d);
      return Math.exp(-d * d / (2 * w * w));
    };
    return 0.12 * g(p, 0.18, 0.025) - 0.18 * g(p, 0.30, 0.010) + g(p, 0.33, 0.012) - 0.25 * g(p, 0.36, 0.010) + 0.30 * g(p, 0.62, 0.045);
  }

  function render() {
    if (!document.getElementById('hero-canvas')) return;
    const { ctx, width, height } = state;
    ctx.clearRect(0, 0, width, height);

    for (let i = 0; i < 2; i++) {
      buffer.push(signal(t));
      t++;
      if (buffer.length > 1200) buffer.shift();
    }

    const tau = 14;
    const n = buffer.length;
    let minVal = 1e9, maxVal = -1e9;
    for (let i = Math.max(0, n - 400); i < n; i++) {
      minVal = Math.min(minVal, buffer[i]);
      maxVal = Math.max(maxVal, buffer[i]);
    }
    const pad = (maxVal - minVal) * 0.15 + 1e-6;
    minVal -= pad;
    maxVal += pad;

    // Left Panel: 1-D Signal Stream
    const leftWidth = width * 0.44;
    ctx.strokeStyle = 'rgba(96, 165, 250, 0.75)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    const pointCount = Math.min(220, n);
    for (let i = 0; i < pointCount; i++) {
      const x = 12 + (leftWidth - 24) * (i / (pointCount - 1));
      const y = height - ((buffer[n - pointCount + i] - minVal) / (maxVal - minVal)) * height;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.fillStyle = '#94a3b8';
    ctx.font = '10px sans-serif';
    ctx.fillText('1-D Stream x_t', 12, 18);

    // Right Panel: Phase-Space Reconstruction
    const offsetX = leftWidth + 20;
    const spaceWidth = width - offsetX - 16;
    const spaceHeight = height - 28;

    ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
    ctx.strokeRect(offsetX, 14, spaceWidth, spaceHeight);

    const mapX = v => offsetX + ((v - minVal) / (maxVal - minVal)) * spaceWidth;
    const mapY = v => 14 + spaceHeight - ((v - minVal) / (maxVal - minVal)) * spaceHeight;

    const trajCount = Math.min(380, n - tau);
    ctx.strokeStyle = 'rgba(45, 212, 191, 0.4)';
    ctx.lineWidth = 1.25;
    ctx.beginPath();
    for (let k = 0; k < trajCount; k++) {
      const idx = n - trajCount + k;
      const px = mapX(buffer[idx - tau]);
      const py = mapY(buffer[idx]);
      if (k === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.stroke();

    const lastIdx = n - 1;
    ctx.save();
    ctx.shadowColor = '#2dd4bf';
    ctx.shadowBlur = 8;
    ctx.fillStyle = '#5eead4';
    ctx.beginPath();
    ctx.arc(mapX(buffer[lastIdx - tau]), mapY(buffer[lastIdx]), 4, 0, 2 * Math.PI);
    ctx.fill();
    ctx.restore();

    ctx.fillStyle = '#94a3b8';
    ctx.fillText('Phase-Space Attractor', offsetX + 10, 28);

    requestAnimationFrame(render);
  }

  render();
})();
</script>

---

## 🎯 System Capabilities

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 1rem; margin: 1.5rem 0;">

  <div class="card-formal">
    <h3 style="margin-top: 0; font-size: 1rem; font-weight: 600;">Non-Parametric Foundation</h3>
    <p style="font-size: 0.82rem; margin-bottom: 0;">Reconstructs phase-space manifolds via Takens' delay embedding rather than training millions of neural parameters.</p>
  </div>

  <div class="card-formal">
    <h3 style="margin-top: 0; font-size: 1rem; font-weight: 600;">Orthogonal 6-Detector Battery</h3>
    <p style="font-size: 0.82rem; margin-bottom: 0;">Combines phase-space trajectory prediction, covariance geometry, subsequence motifs, subspace isolation, linear AR residuals, and MSE transformer autoencoding.</p>
  </div>

  <div class="card-formal">
    <h3 style="margin-top: 0; font-size: 1rem; font-weight: 600;">Online Meta-Judge</h3>
    <p style="font-size: 0.82rem; margin-bottom: 0;">Hedge multiplicative weights dynamically reweight expert detectors in real time without ground-truth anomaly labels.</p>
  </div>

  <div class="card-formal">
    <h3 style="margin-top: 0; font-size: 1rem; font-weight: 600;">Zero-Lookahead Invariant</h3>
    <p style="font-size: 0.82rem; margin-bottom: 0;">Strict element-by-element streaming. No future observation window is ever exposed during scoring.</p>
  </div>

  <div class="card-formal">
    <h3 style="margin-top: 0; font-size: 1rem; font-weight: 600;">Un-Buffered VUS Metrics</h3>
    <p style="font-size: 0.82rem; margin-bottom: 0;">Evaluated using standard Volume Under Surface (VUS-ROC/PR) with label-only temporal buffering, benchmarked against IAAFT surrogate nulls.</p>
  </div>

  <div class="card-formal">
    <h3 style="margin-top: 0; font-size: 1rem; font-weight: 600;">Self-Audited Honesty</h3>
    <p style="font-size: 0.82rem; margin-bottom: 0;">Disclosed all metric corrections, renamed over-claimed detector modules, and documented limitations openly.</p>
  </div>

</div>

---

## 📊 Preliminary Results

!!! note "Pending External Review"
    All numbers below are preliminary point estimates from a single deterministic run.

| Dataset | Evaluated Points ($N$) | System VUS-ROC | System VUS-PR | IAAFT Null VUS-ROC | Predictive Edge ($\Delta$) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **PhysioNet MIT-BIH (record 100)** | 5,143 | **0.8592** | 0.6926 | ~0.49 | **+0.3701** | Preliminary |
| **CWRU Bearing** | 5,000 | **0.9711** | 0.7111 | ~0.61 | **+0.3656** | Preliminary |

See [Results & Benchmarks](results.md) for full methodology, known limitations, and replication instructions.

---

## 🛡️ Core Principles

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1rem; margin: 1.5rem 0;">

  <div class="card-formal">
    <h3 style="margin-top: 0; color: #2dd4bf; font-size: 1.1rem; font-weight: 700;">01 · Transparency</h3>
    <p style="font-size: 0.82rem; margin-bottom: 0;">Every matrix operation, weight update, and gating transition is fully inspectable and auditable. The code is the documentation.</p>
  </div>

  <div class="card-formal">
    <h3 style="margin-top: 0; color: #60a5fa; font-size: 1.1rem; font-weight: 700;">02 · Reproducibility</h3>
    <p style="font-size: 0.82rem; margin-bottom: 0;">100% deterministic execution with fixed random seeds (`SEED=42`), verified by SHA-256 hash comparison across runs.</p>
  </div>

  <div class="card-formal">
    <h3 style="margin-top: 0; color: #34d399; font-size: 1.1rem; font-weight: 700;">03 · Scientific Rigor</h3>
    <p style="font-size: 0.82rem; margin-bottom: 0;">Corrected evaluation metrics, disclosed limitations, and no inflated claims. Scientific truth over marketing optics.</p>
  </div>

</div>
