/* open-phase-ensemble — Real-Time Streaming Data & System Compartment Simulation */

document.addEventListener("DOMContentLoaded", function () {
  const canvasContainers = document.querySelectorAll(".sim-canvas-container");
  if (canvasContainers.length === 0) return;

  canvasContainers.forEach((container) => {
    initSimulation(container);
  });
});

function initSimulation(container) {
  const canvas = container.querySelector("canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");

  // Resize canvas to parent width
  function resizeCanvas() {
    canvas.width = container.clientWidth - 32;
    canvas.height = 300;
  }
  resizeCanvas();
  window.addEventListener("resize", resizeCanvas);

  // Simulation State
  let isRunning = true;
  let timeStep = 0;
  let anomalyActive = false;
  let anomalyDuration = 0;

  // Signal Buffer (100 history points)
  const bufferLen = 120;
  let signalData = new Array(bufferLen).fill(0);
  let fusedScoreData = new Array(bufferLen).fill(0.05);

  // Detector Weights (6 experts)
  const detectorNames = ["EDM", "Mahalanobis", "STOMP", "IForest", "AR-Filter", "Transformer"];
  let weights = [0.166, 0.166, 0.166, 0.166, 0.166, 0.166];

  // Compartment Nodes
  const compartments = [
    { name: "1. Ingestion", label: "StreamBuffer", color: "#3b82f6" },
    { name: "2. Representation", label: "Takens / JL", color: "#8b5cf6" },
    { name: "3. Battery", label: "6 Experts", color: "#ec4899" },
    { name: "4. Meta-Judge", label: "Hedge Fusion", color: "#10b981" },
    { name: "5. Gating", label: "CUSUM", color: "#f59e0b" },
  ];

  // Particles traveling along edges
  let particles = [];
  for (let i = 0; i < 8; i++) {
    particles.push({
      progress: (i / 8) * 4, // 0 to 4
      speed: 0.02 + Math.random() * 0.01,
      isAnomaly: false,
    });
  }

  // Control Buttons
  const playBtn = container.querySelector(".sim-btn-play");
  const anomalyBtn = container.querySelector(".sim-btn-anomaly");
  const resetBtn = container.querySelector(".sim-btn-reset");

  if (playBtn) {
    playBtn.addEventListener("click", () => {
      isRunning = !isRunning;
      playBtn.textContent = isRunning ? "Pause" : "Play";
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

  // Animation Loop
  function step() {
    if (isRunning) {
      timeStep += 0.1;

      // 1. Generate Next Scalar x_t
      let val = Math.sin(timeStep * 0.8) * 0.5 + Math.sin(timeStep * 2.5) * 0.2;
      let score = 0.05 + Math.random() * 0.03;

      if (anomalyActive && anomalyDuration > 0) {
        val += (Math.random() > 0.5 ? 1 : -1) * (2.0 + Math.random() * 1.5);
        score = 0.85 + Math.random() * 0.12;
        anomalyDuration--;

        // Dynamic Hedge Weight Adaptation during anomaly
        weights[0] = Math.min(0.45, weights[0] + 0.02); // EDM spikes
        weights[1] = Math.min(0.35, weights[1] + 0.015);
        for (let k = 2; k < 6; k++) {
          weights[k] = Math.max(0.05, weights[k] - 0.01);
        }
      } else {
        anomalyActive = false;
        // Slowly relax weights back to uniform floor
        for (let k = 0; k < 6; k++) {
          weights[k] += (0.166 - weights[k]) * 0.05;
        }
      }

      signalData.shift();
      signalData.push(val);

      fusedScoreData.shift();
      fusedScoreData.push(score);

      // Move particles
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

  // Render Canvas Scene
  function render() {
    const width = canvas.width;
    const height = canvas.height;
    ctx.clearRect(0, 0, width, height);

    const isDark = document.body.getAttribute("data-md-color-scheme") === "slate";
    const bgGrid = isDark ? "rgba(255,255,255,0.04)" : "rgba(0,0,0,0.04)";
    const textColor = isDark ? "#d1d5db" : "#374151";
    const mutedColor = isDark ? "#6b7280" : "#9ca3af";

    // ── Layout Panels ──────────────────────────────────────────
    // Top Panel: Streaming Waveform & Score (Height: 110px)
    // Middle Panel: Compartment Pipeline & Particles (Height: 90px)
    // Bottom Panel: Meta-Judge Weights (Height: 80px)

    // Panel Dividers & Labels
    ctx.strokeStyle = bgGrid;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, 115); ctx.lineTo(width, 115);
    ctx.moveTo(0, 215); ctx.lineTo(width, 215);
    ctx.stroke();

    // ── Panel 1: Live Time-Series Stream ───────────────────────
    ctx.fillStyle = mutedColor;
    ctx.font = "10px Inter, sans-serif";
    ctx.fillText("LIVE STREAM x_t & FUSED ANOMALY SCORE A_t", 10, 15);

    const streamW = width - 20;
    const streamH = 80;
    const streamY0 = 65;

    // Draw Signal x_t
    ctx.strokeStyle = isDark ? "#60a5fa" : "#2563eb";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i < bufferLen; i++) {
      const x = 10 + (i / (bufferLen - 1)) * streamW;
      const y = streamY0 - signalData[i] * 18;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Draw Fused Score A_t (Overlay in Coral/Red)
    ctx.strokeStyle = "#f43f5e";
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    for (let i = 0; i < bufferLen; i++) {
      const x = 10 + (i / (bufferLen - 1)) * streamW;
      const y = 105 - fusedScoreData[i] * 70;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Threshold Line A_t = 0.5
    ctx.strokeStyle = "rgba(244,63,94,0.3)";
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(10, 105 - 0.5 * 70);
    ctx.lineTo(10 + streamW, 105 - 0.5 * 70);
    ctx.stroke();
    ctx.setLineDash([]);

    // Legend
    ctx.fillStyle = isDark ? "#60a5fa" : "#2563eb";
    ctx.fillText("— Observation x_t", width - 180, 15);
    ctx.fillStyle = "#f43f5e";
    ctx.fillText("— Fused Score A_t", width - 85, 15);

    // ── Panel 2: Compartment Data Flow Pipeline ────────────────
    ctx.fillStyle = mutedColor;
    ctx.fillText("SYSTEM COMPARTMENTS & PARTICLE FLOW", 10, 130);

    const compCount = compartments.length;
    const boxW = Math.min(100, (width - 40 - (compCount - 1) * 15) / compCount);
    const gap = (width - 20 - compCount * boxW) / (compCount - 1);
    const compY = 145;
    const compH = 50;

    let compCoords = [];

    // Draw Nodes
    compartments.forEach((comp, idx) => {
      const x = 10 + idx * (boxW + gap);
      compCoords.push({ x: x + boxW / 2, y: compY + compH / 2 });

      // Node Box
      ctx.fillStyle = isDark ? "#1f2937" : "#ffffff";
      ctx.strokeStyle = comp.color;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.roundRect(x, compY, boxW, compH, 6);
      ctx.fill();
      ctx.stroke();

      // Node Text
      ctx.fillStyle = textColor;
      ctx.font = "bold 9px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(comp.name, x + boxW / 2, compY + 20);
      ctx.font = "8px Inter, sans-serif";
      ctx.fillStyle = mutedColor;
      ctx.fillText(comp.label, x + boxW / 2, compY + 36);
      ctx.textAlign = "left";
    });

    // Draw Connector Edges & Animated Particles
    for (let i = 0; i < compCount - 1; i++) {
      const p1 = compCoords[i];
      const p2 = compCoords[i + 1];

      ctx.strokeStyle = isDark ? "rgba(255,255,255,0.15)" : "rgba(0,0,0,0.15)";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(p1.x + boxW / 2 - 5, p1.y);
      ctx.lineTo(p2.x - boxW / 2 + 5, p2.y);
      ctx.stroke();
    }

    // Render Moving Particles
    particles.forEach((p) => {
      const segIndex = Math.floor(p.progress);
      const frac = p.progress - segIndex;
      if (segIndex < compCount - 1) {
        const p1 = compCoords[segIndex];
        const p2 = compCoords[segIndex + 1];
        const px = p1.x + frac * (p2.x - p1.x);
        const py = p1.y + frac * (p2.y - p1.y);

        ctx.fillStyle = p.isAnomaly ? "#f43f5e" : "#10b981";
        ctx.beginPath();
        ctx.arc(px, py, 4, 0, Math.PI * 2);
        ctx.fill();
      }
    });

    // ── Panel 3: Meta-Judge Dynamic Weights ────────────────────
    ctx.fillStyle = mutedColor;
    ctx.fillText("META-JUDGE HEDGE WEIGHTS w_k", 10, 230);

    const barW = Math.min(60, (width - 40) / 6 - 10);
    const barY0 = 290;
    const maxBarH = 50;

    for (let k = 0; k < 6; k++) {
      const bx = 10 + k * (barW + 12);
      const bw = weights[k];
      const bh = bw * maxBarH * 3; // scaled

      ctx.fillStyle = isDark ? "#374151" : "#e5e7eb";
      ctx.fillRect(bx, barY0 - maxBarH, barW, maxBarH);

      ctx.fillStyle = k === 0 && weights[0] > 0.25 ? "#f43f5e" : "#0d9488";
      ctx.fillRect(bx, barY0 - bh, barW, bh);

      ctx.fillStyle = textColor;
      ctx.font = "8px Inter, sans-serif";
      ctx.fillText(detectorNames[k], bx, barY0 + 9);
      ctx.fillText((bw * 100).toFixed(0) + "%", bx, barY0 - bh - 3);
    }
  }

  // Start Animation Loop
  requestAnimationFrame(step);
}
