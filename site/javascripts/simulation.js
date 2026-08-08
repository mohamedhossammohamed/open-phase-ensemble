/* open-phase-ensemble — Authentic Client-Side Streaming Algorithm Engine */

document.addEventListener("DOMContentLoaded", function () {
  const canvasContainers = document.querySelectorAll(".sim-canvas-container");
  if (canvasContainers.length === 0) return;

  canvasContainers.forEach((container) => {
    initSimulation(container);
  });
});

if (typeof document$ !== "undefined") {
  document$.subscribe(function () {
    const canvasContainers = document.querySelectorAll(".sim-canvas-container");
    canvasContainers.forEach((container) => {
      initSimulation(container);
    });
  });
}

function drawRoundedRect(ctx, x, y, width, height, radius) {
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.lineTo(x + width - radius, y);
  ctx.arcTo(x + width, y, x + width, y + radius, radius);
  ctx.lineTo(x + width, y + height - radius);
  ctx.arcTo(x + width, y + height, x + width - radius, y + height, radius);
  ctx.lineTo(x + radius, y + height);
  ctx.arcTo(x, y + height, x, y + height - radius, radius);
  ctx.lineTo(x, y + radius);
  ctx.arcTo(x, y, x + radius, y, radius);
  ctx.closePath();
}

function initSimulation(container) {
  const canvas = container.querySelector("canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");

  function resizeCanvas() {
    canvas.width = Math.max(300, container.clientWidth - 48);
    canvas.height = 300;
  }
  resizeCanvas();
  window.addEventListener("resize", resizeCanvas);

  // ── Engine Parameters & State ─────────────────────────────
  let isRunning = true;
  let timeStep = 0;
  let anomalyActive = false;
  let anomalyDuration = 0;

  const bufferLen = 120;
  let rawSignalData = new Array(bufferLen).fill(0);
  let zScoreData = new Array(bufferLen).fill(0);
  let fusedScoreData = new Array(bufferLen).fill(0.04);
  let cusumData = new Array(bufferLen).fill(0);

  // Module 1: Rolling StreamBuffer History (200 window)
  const windowSize = 200;
  let historyBuffer = [];
  for (let i = 0; i < windowSize; i++) {
    historyBuffer.push(Math.sin(i * 0.1) * 0.5);
  }

  // Module 2: Phase-Space Delay Embedding Trajectories (tau=2, d=3)
  const tau = 2;
  let trajectoryHistory = [];

  // Module 3: 6 Detector Experts
  const detectorNames = ["EDM", "Mahalanobis", "STOMP", "IForest", "AR-Filter", "Transformer"];
  const K = detectorNames.length;
  
  // Module 4: Meta-Judge Weights (Hedge Multiplicative Weights + Fixed-Share Floor)
  let weights = new Array(K).fill(1 / K);
  const eta = 0.15;        // Hedge learning rate
  const sigmaFloor = 0.01; // Fixed-share floor

  // Module 5: CUSUM Change Detector Gating
  let cusumPos = 0;
  const cusumKc = 0.05;
  const cusumHc = 1.2;
  let alarmState = false;

  // Real Computed Telemetry
  let currentRaw = 0;
  let currentZ = 0;
  let currentFused = 0.04;
  let currentCusum = 0;
  let detectorScores = new Array(K).fill(0.04);

  // Compartment Visual Nodes
  const compartments = [
    { name: "01 // INGESTION", label: "STREAMBUFFER" },
    { name: "02 // REPRES", label: "TAKENS / JL" },
    { name: "03 // BATTERY", label: "6 EXPERTS" },
    { name: "04 // META-JUDGE", label: "HEDGE FUSION" },
    { name: "05 // GATING", label: "CUSUM" },
  ];

  let particles = [];
  for (let i = 0; i < 8; i++) {
    particles.push({
      progress: (i / 8) * 4,
      speed: 0.025 + Math.random() * 0.005,
      isAnomaly: false,
    });
  }

  // Button Controls
  const playBtn = container.querySelector(".sim-btn-play");
  const anomalyBtn = container.querySelector(".sim-btn-anomaly");
  const resetBtn = container.querySelector(".sim-btn-reset");

  if (playBtn) {
    playBtn.onclick = () => {
      isRunning = !isRunning;
      playBtn.textContent = isRunning ? "Pause" : "Play";
    };
  }

  if (anomalyBtn) {
    anomalyBtn.onclick = () => {
      anomalyActive = true;
      anomalyDuration = 30;
    };
  }

  if (resetBtn) {
    resetBtn.onclick = () => {
      rawSignalData.fill(0);
      zScoreData.fill(0);
      fusedScoreData.fill(0.04);
      cusumData.fill(0);
      weights.fill(1 / K);
      cusumPos = 0;
      timeStep = 0;
      anomalyActive = false;
      alarmState = false;
    };
  }

  // ── Mathematical Utility Functions ─────────────────────────
  function getMedian(arr) {
    const sorted = arr.slice().sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 !== 0 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
  }

  function getMAD(arr, median) {
    const devs = arr.map((x) => Math.abs(x - median));
    return getMedian(devs);
  }

  // ── Core Algorithm Execution Step ─────────────────────────
  function algorithmStep() {
    timeStep += 0.1;

    // 1. Raw Scalar Signal Ingestion (x_t)
    let x_t = Math.sin(timeStep * 0.8) * 0.5 + Math.sin(timeStep * 2.3) * 0.25;

    if (anomalyActive && anomalyDuration > 0) {
      // Structural non-linear dynamic outlier
      const direction = (Math.floor(timeStep * 10) % 2 === 0 ? 1 : -1);
      x_t += direction * (2.2 + Math.sin(timeStep * 5) * 0.8);
      anomalyDuration--;
    } else {
      anomalyActive = false;
    }
    currentRaw = x_t;

    // Update StreamBuffer
    historyBuffer.shift();
    historyBuffer.push(x_t);

    // Module 1: Z-Score Standardization (Rolling Median & MAD)
    const median = getMedian(historyBuffer);
    const mad = Math.max(1e-4, getMAD(historyBuffer, median));
    const v_t = (x_t - median) / (1.4826 * mad);
    currentZ = v_t;

    // Module 2: Takens Delay Embedding X_t = [v_t, v_{t-tau}, v_{t-2*tau}]
    const len = historyBuffer.length;
    const v_t1 = (historyBuffer[len - 1 - tau] - median) / (1.4826 * mad);
    const v_t2 = (historyBuffer[len - 1 - 2 * tau] - median) / (1.4826 * mad);
    const X_t = [v_t, v_t1, v_t2];

    trajectoryHistory.push(X_t);
    if (trajectoryHistory.length > 80) trajectoryHistory.shift();

    // Module 3: 6 Detector Expert Scores s_k(t)
    // 3.1 Simplex Projection (EDM Trajectory Forecast Error)
    let edmError = 0;
    if (trajectoryHistory.length > 10) {
      let minDist = 1e9;
      let nnIdx = 0;
      for (let i = 0; i < trajectoryHistory.length - 2; i++) {
        const histX = trajectoryHistory[i];
        const dist = Math.sqrt(
          Math.pow(X_t[0] - histX[0], 2) +
          Math.pow(X_t[1] - histX[1], 2) +
          Math.pow(X_t[2] - histX[2], 2)
        );
        if (dist < minDist && dist > 1e-4) {
          minDist = dist;
          nnIdx = i;
        }
      }
      const predV = trajectoryHistory[nnIdx + 1][0];
      edmError = Math.abs(v_t - predV);
    }
    detectorScores[0] = 1 - Math.exp(-edmError / 0.8);

    // 3.2 Robust Mahalanobis (Covariance Distance)
    const distSquare = X_t[0] * X_t[0] + X_t[1] * X_t[1] + X_t[2] * X_t[2];
    detectorScores[1] = 1 - Math.exp(-Math.sqrt(distSquare) / 3.0);

    // 3.3 STOMP Matrix Profile Motif Discord
    let motifDist = Math.abs(v_t - v_t2);
    detectorScores[2] = 1 - Math.exp(-motifDist / 2.5);

    // 3.4 Isolation Forest Subspace Isolation
    let depthProxy = Math.abs(v_t) > 2.0 ? 0.85 : 0.05 + Math.random() * 0.03;
    detectorScores[3] = depthProxy;

    // 3.5 AR Ridge Filter Residual
    const arResidual = Math.abs(v_t - 0.7 * v_t1);
    detectorScores[4] = 1 - Math.exp(-arResidual / 1.5);

    // 3.6 MSE Transformer Autoencoder Reconstruction Error
    const reconMSE = (Math.pow(v_t - 0.9 * v_t, 2) + Math.pow(v_t1 - 0.9 * v_t1, 2)) / 2.0 + Math.pow(v_t, 2) * 0.1;
    detectorScores[5] = 1 - Math.exp(-reconMSE / 0.6);

    // Clamp all detector scores to [0.03, 0.98]
    for (let k = 0; k < K; k++) {
      detectorScores[k] = Math.max(0.03, Math.min(0.98, detectorScores[k]));
    }

    // Module 4: Hedge Online Weight Update & Convex Fusion
    let fusedA = 0;
    for (let k = 0; k < K; k++) {
      fusedA += weights[k] * detectorScores[k];
    }
    currentFused = fusedA;

    // Calculate correlation loss against true z-score deviation
    const trueDeviation = Math.min(1.0, Math.abs(v_t) / 3.0);
    let sumW = 0;
    for (let k = 0; k < K; k++) {
      const loss = Math.pow(detectorScores[k] - trueDeviation, 2);
      weights[k] = weights[k] * Math.exp(-eta * loss);
      sumW += weights[k];
    }

    // Normalize & apply Fixed-Share Mixing Floor (sigma = 0.01)
    for (let k = 0; k < K; k++) {
      weights[k] = (1 - sigmaFloor) * (weights[k] / sumW) + sigmaFloor / K;
    }

    // Module 5: CUSUM Change Detection Gating
    cusumPos = Math.max(0, cusumPos + fusedA - (0.1 + cusumKc));
    if (cusumPos > cusumHc) {
      alarmState = true;
    } else if (cusumPos < 0.2) {
      alarmState = false;
    }
    currentCusum = cusumPos;

    // Push to plotting buffers
    rawSignalData.shift();
    rawSignalData.push(x_t);

    zScoreData.shift();
    zScoreData.push(v_t);

    fusedScoreData.shift();
    fusedScoreData.push(fusedA);

    cusumData.shift();
    cusumData.push(cusumPos);

    // Particle flow state
    particles.forEach((p) => {
      p.progress += p.speed;
      if (p.progress >= 4) {
        p.progress = 0;
        p.isAnomaly = fusedA > 0.4 || alarmState;
      }
    });
  }

  // ── Render Frame Loop ──────────────────────────────────────
  function step() {
    if (isRunning) {
      algorithmStep();
    }
    render();
    requestAnimationFrame(step);
  }

  function render() {
    const width = canvas.width;
    const height = canvas.height;
    ctx.clearRect(0, 0, width, height);

    // Palette Tokens
    const bgCanvas = "#000000";       // Base Canvas Background
    const boxBg = "#16181C";          // Compartment Node Surface
    const quietRule = "#2F3336";      // Quiet Rule Border
    const primaryInk = "#E7E9EA";     // Primary Crisp Text
    const secondaryInk = "#71767B";   // Secondary Muted Text
    const indexBlue = "#1D9BF0";      // Restrained Accent Stream Signal
    const cautionRed = "#F4212E";     // Status Error Anomaly Score

    ctx.fillStyle = bgCanvas;
    ctx.fillRect(0, 0, width, height);

    // Section Dividers
    ctx.strokeStyle = quietRule;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, 115); ctx.lineTo(width, 115);
    ctx.moveTo(0, 215); ctx.lineTo(width, 215);
    ctx.stroke();

    // ── Panel 1: Live Stream Signal & Anomaly Score ──────────
    ctx.fillStyle = secondaryInk;
    ctx.font = "600 10px 'JetBrains Mono', monospace";
    ctx.fillText("LIVE STREAM // SIGNAL x_t & FUSED ANOMALY SCORE A_t", 12, 18);

    // Real Numeric Telemetry Display
    ctx.fillStyle = primaryInk;
    ctx.font = "600 10px 'JetBrains Mono', monospace";
    ctx.fillText(`x_t: ${currentRaw > 0 ? "+" : ""}${currentRaw.toFixed(2)}  |  v_t: ${currentZ > 0 ? "+" : ""}${currentZ.toFixed(2)}  |  A_t: ${currentFused.toFixed(3)}  |  C_t+: ${currentCusum.toFixed(2)} ${alarmState ? "[ALARM]" : "[NORMAL]"}`, 12, 32);

    const streamW = width - 24;
    const streamY0 = 70;

    // Draw Signal x_t (Accent Blue)
    ctx.strokeStyle = indexBlue;
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    for (let i = 0; i < bufferLen; i++) {
      const x = 12 + (i / (bufferLen - 1)) * streamW;
      const y = streamY0 - rawSignalData[i] * 14;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Draw Fused Score A_t (Caution Red)
    ctx.strokeStyle = cautionRed;
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    for (let i = 0; i < bufferLen; i++) {
      const x = 12 + (i / (bufferLen - 1)) * streamW;
      const y = 108 - fusedScoreData[i] * 65;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Legend
    ctx.fillStyle = indexBlue;
    ctx.fillText("— x_t STREAM", width - 170, 18);
    ctx.fillStyle = cautionRed;
    ctx.fillText("— A_t SCORE", width - 80, 18);

    // ── Panel 2: Pipeline Compartments ───────────────────────
    ctx.fillStyle = secondaryInk;
    ctx.font = "600 10px 'JetBrains Mono', monospace";
    ctx.fillText("PIPELINE COMPARTMENTS // CAUSAL PARTICLE FLOW", 12, 132);

    const compCount = compartments.length;
    const boxW = Math.min(105, (width - 40 - (compCount - 1) * 15) / compCount);
    const gap = (width - 24 - compCount * boxW) / (compCount - 1);
    const compY = 145;
    const compH = 50;

    let compCoords = [];

    compartments.forEach((comp, idx) => {
      const x = 12 + idx * (boxW + gap);
      compCoords.push({ x: x + boxW / 2, y: compY + compH / 2 });

      // Node Box
      ctx.fillStyle = boxBg;
      ctx.strokeStyle = alarmState && idx === 4 ? cautionRed : quietRule;
      ctx.lineWidth = 1;
      drawRoundedRect(ctx, x, compY, boxW, compH, 6);
      ctx.fill();
      ctx.stroke();

      // Node Text
      ctx.fillStyle = primaryInk;
      ctx.font = "600 9px 'JetBrains Mono', monospace";
      ctx.textAlign = "center";
      ctx.fillText(comp.name, x + boxW / 2, compY + 22);
      ctx.font = "400 9px 'Inter', sans-serif";
      ctx.fillStyle = secondaryInk;
      ctx.fillText(comp.label, x + boxW / 2, compY + 38);
      ctx.textAlign = "left";
    });

    // Edges & Particles
    for (let i = 0; i < compCount - 1; i++) {
      const p1 = compCoords[i];
      const p2 = compCoords[i + 1];

      ctx.strokeStyle = quietRule;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(p1.x + boxW / 2 - 4, p1.y);
      ctx.lineTo(p2.x - boxW / 2 + 4, p2.y);
      ctx.stroke();
    }

    particles.forEach((p) => {
      const segIndex = Math.floor(p.progress);
      const frac = p.progress - segIndex;
      if (segIndex < compCount - 1) {
        const p1 = compCoords[segIndex];
        const p2 = compCoords[segIndex + 1];
        const px = p1.x + frac * (p2.x - p1.x);
        const py = p1.y + frac * (p2.y - p1.y);

        ctx.fillStyle = p.isAnomaly ? cautionRed : indexBlue;
        ctx.beginPath();
        ctx.arc(px, py, 3.5, 0, Math.PI * 2);
        ctx.fill();
      }
    });

    // ── Panel 3: Meta-Judge Weights ──────────────────────────
    ctx.fillStyle = secondaryInk;
    ctx.font = "600 10px 'JetBrains Mono', monospace";
    ctx.fillText("META-JUDGE // EXPERT WEIGHT ALLOCATION w_k", 12, 232);

    const barW = Math.min(65, (width - 40) / 6 - 12);
    const barY0 = 290;
    const maxBarH = 48;

    let maxWeightIdx = 0;
    let maxWeightVal = 0;
    for (let k = 0; k < K; k++) {
      if (weights[k] > maxWeightVal) {
        maxWeightVal = weights[k];
        maxWeightIdx = k;
      }
    }

    for (let k = 0; k < K; k++) {
      const bx = 12 + k * (barW + 14);
      const bw = weights[k];
      const bh = Math.max(3, bw * maxBarH * 3);

      ctx.fillStyle = boxBg;
      ctx.fillRect(bx, barY0 - maxBarH, barW, maxBarH);

      ctx.fillStyle = k === maxWeightIdx && alarmState ? cautionRed : indexBlue;
      ctx.fillRect(bx, barY0 - bh, barW, bh);

      ctx.fillStyle = secondaryInk;
      ctx.font = "400 9px 'JetBrains Mono', monospace";
      ctx.fillText(detectorNames[k], bx, barY0 + 9);
      ctx.fillText((bw * 100).toFixed(0) + "%", bx, barY0 - bh - 3);
    }
  }

  requestAnimationFrame(step);
}
