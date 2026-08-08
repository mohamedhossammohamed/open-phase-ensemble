/* open-phase-ensemble — xAI Inspired Data Flow & System Compartment Simulation */

document.addEventListener("DOMContentLoaded", function () {
  const canvasContainers = document.querySelectorAll(".sim-canvas-container");
  if (canvasContainers.length === 0) return;

  canvasContainers.forEach((container) => {
    initSimulation(container);
  });
});

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

  // Simulation State
  let isRunning = true;
  let timeStep = 0;
  let anomalyActive = false;
  let anomalyDuration = 0;

  const bufferLen = 120;
  let signalData = new Array(bufferLen).fill(0);
  let fusedScoreData = new Array(bufferLen).fill(0.05);

  const detectorNames = ["EDM", "Mahalanobis", "STOMP", "IForest", "AR-Filter", "Transformer"];
  let weights = [0.166, 0.166, 0.166, 0.166, 0.166, 0.166];

  const compartments = [
    { name: "01 // INGESTION", label: "STREAMBUFFER", color: "#ffffff" },
    { name: "02 // REPRES", label: "TAKENS / JL", color: "#ffffff" },
    { name: "03 // BATTERY", label: "6 EXPERTS", color: "#ffffff" },
    { name: "04 // META-JUDGE", label: "HEDGE FUSION", color: "#ffffff" },
    { name: "05 // GATING", label: "CUSUM", color: "#ff7a17" },
  ];

  let particles = [];
  for (let i = 0; i < 8; i++) {
    particles.push({
      progress: (i / 8) * 4,
      speed: 0.02 + Math.random() * 0.01,
      isAnomaly: false,
    });
  }

  const playBtn = container.querySelector(".sim-btn-play");
  const anomalyBtn = container.querySelector(".sim-btn-anomaly");
  const resetBtn = container.querySelector(".sim-btn-reset");

  if (playBtn) {
    playBtn.addEventListener("click", () => {
      isRunning = !isRunning;
      playBtn.textContent = isRunning ? "PAUSE" : "PLAY";
    });
  }

  if (anomalyBtn) {
    anomalyBtn.addEventListener("click", () => {
      anomalyActive = true;
      anomalyDuration = 25;
    });
  }

  if (resetBtn) {
    resetBtn.addEventListener("click", () => {
      signalData.fill(0);
      fusedScoreData.fill(0.05);
      weights = [0.166, 0.166, 0.166, 0.166, 0.166, 0.166];
      timeStep = 0;
      anomalyActive = false;
    });
  }

  function step() {
    if (isRunning) {
      timeStep += 0.1;

      let val = Math.sin(timeStep * 0.8) * 0.5 + Math.sin(timeStep * 2.5) * 0.2;
      let score = 0.05 + Math.random() * 0.03;

      if (anomalyActive && anomalyDuration > 0) {
        val += (Math.random() > 0.5 ? 1 : -1) * (2.0 + Math.random() * 1.5);
        score = 0.85 + Math.random() * 0.12;
        anomalyDuration--;

        weights[0] = Math.min(0.45, weights[0] + 0.02);
        weights[1] = Math.min(0.35, weights[1] + 0.015);
        for (let k = 2; k < 6; k++) {
          weights[k] = Math.max(0.05, weights[k] - 0.01);
        }
      } else {
        anomalyActive = false;
        for (let k = 0; k < 6; k++) {
          weights[k] += (0.166 - weights[k]) * 0.05;
        }
      }

      signalData.shift();
      signalData.push(val);

      fusedScoreData.shift();
      fusedScoreData.push(score);

      particles.forEach((p) => {
        p.progress += p.speed;
        if (p.progress >= 4) {
          p.progress = 0;
          p.isAnomaly = anomalyActive;
        }
      });
    }

    render();
    requestAnimationFrame(step);
  }

  function render() {
    const width = canvas.width;
    const height = canvas.height;
    ctx.clearRect(0, 0, width, height);

    // Strict xAI Monochrome Palette
    const bgCanvas = "#0a0a0a";
    const hairline = "#212327";
    const inkWhite = "#ffffff";
    const bodyMid = "#7d8187";
    const sunsetOrange = "#ff7a17";

    ctx.fillStyle = bgCanvas;
    ctx.fillRect(0, 0, width, height);

    // Dividers
    ctx.strokeStyle = hairline;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, 115); ctx.lineTo(width, 115);
    ctx.moveTo(0, 215); ctx.lineTo(width, 215);
    ctx.stroke();

    // ── Panel 1: Live Stream ──────────────────────────────────
    ctx.fillStyle = bodyMid;
    ctx.font = "10px Geist Mono, JetBrains Mono, monospace";
    ctx.fillText("LIVE STREAM // SIGNAL x_t & FUSED ANOMALY SCORE A_t", 12, 18);

    const streamW = width - 24;
    const streamY0 = 65;

    // Draw Signal x_t (Pure White)
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    for (let i = 0; i < bufferLen; i++) {
      const x = 12 + (i / (bufferLen - 1)) * streamW;
      const y = streamY0 - signalData[i] * 18;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Draw Fused Score A_t (Sunset Orange #ff7a17)
    ctx.strokeStyle = sunsetOrange;
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    for (let i = 0; i < bufferLen; i++) {
      const x = 12 + (i / (bufferLen - 1)) * streamW;
      const y = 105 - fusedScoreData[i] * 70;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Legend
    ctx.fillStyle = "#ffffff";
    ctx.fillText("— x_t STREAM", width - 170, 18);
    ctx.fillStyle = sunsetOrange;
    ctx.fillText("— A_t SCORE", width - 80, 18);

    // ── Panel 2: Pipeline Compartments ───────────────────────
    ctx.fillStyle = bodyMid;
    ctx.fillText("PIPELINE COMPARTMENTS // REAL-TIME PARTICLE FLOW", 12, 132);

    const compCount = compartments.length;
    const boxW = Math.min(105, (width - 40 - (compCount - 1) * 15) / compCount);
    const gap = (width - 24 - compCount * boxW) / (compCount - 1);
    const compY = 145;
    const compH = 50;

    let compCoords = [];

    compartments.forEach((comp, idx) => {
      const x = 12 + idx * (boxW + gap);
      compCoords.push({ x: x + boxW / 2, y: compY + compH / 2 });

      // Node Box (Hairline border on charcoal #191919)
      ctx.fillStyle = "#191919";
      ctx.strokeStyle = hairline;
      ctx.lineWidth = 1;
      drawRoundedRect(ctx, x, compY, boxW, compH, 6);
      ctx.fill();
      ctx.stroke();

      // Node Text
      ctx.fillStyle = inkWhite;
      ctx.font = "10px Geist Mono, JetBrains Mono, monospace";
      ctx.textAlign = "center";
      ctx.fillText(comp.name, x + boxW / 2, compY + 22);
      ctx.font = "9px Geist Mono, JetBrains Mono, monospace";
      ctx.fillStyle = bodyMid;
      ctx.fillText(comp.label, x + boxW / 2, compY + 38);
      ctx.textAlign = "left";
    });

    // Edges & Particles
    for (let i = 0; i < compCount - 1; i++) {
      const p1 = compCoords[i];
      const p2 = compCoords[i + 1];

      ctx.strokeStyle = hairline;
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

        ctx.fillStyle = p.isAnomaly ? sunsetOrange : inkWhite;
        ctx.beginPath();
        ctx.arc(px, py, 3.5, 0, Math.PI * 2);
        ctx.fill();
      }
    });

    // ── Panel 3: Meta-Judge Weights ──────────────────────────
    ctx.fillStyle = bodyMid;
    ctx.fillText("META-JUDGE // EXPERT WEIGHT ALLOCATION w_k", 12, 232);

    const barW = Math.min(65, (width - 40) / 6 - 12);
    const barY0 = 290;
    const maxBarH = 48;

    for (let k = 0; k < 6; k++) {
      const bx = 12 + k * (barW + 14);
      const bw = weights[k];
      const bh = bw * maxBarH * 3;

      ctx.fillStyle = "#191919";
      ctx.fillRect(bx, barY0 - maxBarH, barW, maxBarH);

      ctx.fillStyle = k === 0 && weights[0] > 0.25 ? sunsetOrange : inkWhite;
      ctx.fillRect(bx, barY0 - bh, barW, bh);

      ctx.fillStyle = bodyMid;
      ctx.font = "9px Geist Mono, JetBrains Mono, monospace";
      ctx.fillText(detectorNames[k], bx, barY0 + 9);
      ctx.fillText((bw * 100).toFixed(0) + "%", bx, barY0 - bh - 3);
    }
  }

  requestAnimationFrame(step);
}
