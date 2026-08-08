---
hide:
  - navigation
  - toc
---

<script src="https://cdn.tailwindcss.com"></script>
<link href="https://cdn.jsdelivr.net/fontsource/fonts/space-grotesk@latest/latin-400-normal.css" rel="stylesheet">
<link href="https://cdn.jsdelivr.net/fontsource/fonts/space-grotesk@latest/latin-500-normal.css" rel="stylesheet">
<link href="https://cdn.jsdelivr.net/fontsource/fonts/space-grotesk@latest/latin-700-normal.css" rel="stylesheet">
<link href="https://cdn.jsdelivr.net/fontsource/fonts/inter@latest/latin-400-normal.css" rel="stylesheet">
<link href="https://cdn.jsdelivr.net/fontsource/fonts/inter@latest/latin-500-normal.css" rel="stylesheet">
<link href="https://cdn.jsdelivr.net/fontsource/fonts/jetbrains-mono@latest/latin-400-normal.css" rel="stylesheet">

<style>
  :root{
    --bg:#0b1220; --surface:#101a2e; --line:rgba(255,255,255,.08);
    --cyan:#22d3ee; --indigo:#6366f1; --amber:#f59e0b; --green:#34d399;
    --text:#e6edf7; --muted:#94a3b8;
  }
  *{-webkit-tap-highlight-color:transparent;scroll-behavior:smooth;}
  html{scroll-behavior:smooth;}
  body, .md-main, .md-container, .md-content { background: var(--bg) !important; color: var(--text); }
  .md-header, .md-tabs { display: none !important; }
  h1,h2,h3,.display{font-family:'Space Grotesk',sans-serif;}
  .mono{font-family:'JetBrains Mono',monospace;}
  .glass{background:rgba(255,255,255,.03);border:1px solid var(--line);backdrop-filter:blur(10px);}
  .card{background:var(--surface);border:1px solid var(--line);border-radius:16px;transition:transform .25s ease,border-color .25s ease,box-shadow .25s ease;}
  .card:hover{transform:translateY(-4px);border-color:rgba(34,211,238,.35);box-shadow:0 12px 32px rgba(2,8,20,.5);}
  .grad-text{background:linear-gradient(90deg,#7dd3fc,#22d3ee,#818cf8);-webkit-background-clip:text;background-clip:text;color:transparent;}
  .grid-bg{background-image:linear-gradient(rgba(148,163,184,.06) 1px,transparent 1px),linear-gradient(90deg,rgba(148,163,184,.06) 1px,transparent 1px);background-size:44px 44px;}
  .reveal{opacity:0;transform:translateY(24px);transition:opacity .7s ease,transform .7s ease;}
  .reveal.in{opacity:1;transform:none;}
  .badge{display:inline-flex;align-items:center;gap:.4rem;font-size:.72rem;font-weight:500;padding:.3rem .7rem;border-radius:999px;border:1px solid var(--line);}
  .pill{display:inline-flex;align-items:center;font-size:.7rem;padding:.2rem .6rem;border-radius:999px;font-weight:600;}
  .navlink{color:var(--muted);font-size:.875rem;font-weight:500;transition:color .2s;}
  .navlink:hover{color:var(--text);}
  .btn-primary{background:linear-gradient(135deg,#0ea5e9,#6366f1);color:#fff;font-weight:600;transition:transform .2s,box-shadow .2s;box-shadow:0 6px 20px rgba(56,189,248,.3);}
  .btn-primary:hover{transform:translateY(-2px);box-shadow:0 10px 28px rgba(56,189,248,.4);}
  .btn-ghost{border:1px solid var(--line);color:var(--text);font-weight:500;transition:background .2s,border-color .2s;}
  .btn-ghost:hover{background:rgba(255,255,255,.06);border-color:rgba(34,211,238,.4);}
  pre{font-family:'JetBrains Mono',monospace;font-size:.8rem;line-height:1.6;background:#0a0f1c;border:1px solid var(--line);border-radius:12px;padding:16px;overflow:auto;color:#a5f3fc;}
  table{width:100%;border-collapse:collapse;font-size:.875rem;}
  th{font-family:'JetBrains Mono',monospace;font-size:.7rem;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);text-align:left;padding:.7rem .8rem;border-bottom:1px solid var(--line);}
  td{padding:.85rem .8rem;border-bottom:1px solid var(--line);vertical-align:top;}
  .edgebar{height:6px;border-radius:999px;background:rgba(255,255,255,.08);overflow:hidden;min-width:90px;}
  .edgebar>div{height:100%;border-radius:999px;background:linear-gradient(90deg,#0ea5e9,#34d399);}
  .callout-warn{border:1px solid rgba(245,158,11,.35);background:linear-gradient(135deg,rgba(245,158,11,.08),rgba(245,158,11,.03));border-radius:14px;}
  .num{font-family:'Space Grotesk',sans-serif;font-weight:700;}
  .section-label{font-family:'JetBrains Mono',monospace;font-size:.72rem;letter-spacing:.18em;text-transform:uppercase;color:var(--cyan);}
  .divider{height:1px;background:linear-gradient(90deg,transparent,rgba(148,163,184,.25),transparent);}
  @keyframes float{0%,100%{transform:translateY(0);}50%{transform:translateY(-8px);}}
  .float{animation:float 6s ease-in-out infinite;}
</style>

<!-- ============ NAV ============ -->
<header class="fixed top-0 inset-x-0 z-50 glass border-b border-white/5">
  <nav class="max-w-6xl mx-auto px-4 sm:px-6 h-16 flex items-center justify-between">
    <a href="#top" class="flex items-center gap-3">
      <img src="https://image.qwenlm.ai/public_source/10cec850-dab4-4cc6-9508-a52b88f4dc44/1f5677168-b4c7-423e-9c63-442c93d53bcd.png" alt="open-phase-ensemble logo" class="h-9 w-9 rounded-lg object-cover"/>
      <span class="display font-bold text-lg tracking-tight">open-phase-<span class="grad-text">ensemble</span></span>
    </a>
    <div class="hidden md:flex items-center gap-6">
      <a class="navlink" href="#what">Overview</a>
      <a class="navlink" href="architecture/">Architecture</a>
      <a class="navlink" href="results/">Results</a>
      <a class="navlink" href="theory/">Theory</a>
      <a class="navlink" href="developer/">Developer</a>
    </div>
    <a href="https://github.com/mohamedhossammohamed/open-phase-ensemble" target="_blank" class="btn-ghost rounded-lg px-4 py-2 text-sm flex items-center gap-2">
      <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27s1.36.09 2 .27c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8Z"/></svg>
      GitHub
    </a>
  </nav>
</header>

<!-- ============ HERO ============ -->
<section id="top" class="relative pt-32 pb-20 overflow-hidden grid-bg">
  <div class="absolute inset-0 pointer-events-none" style="background:radial-gradient(700px 400px at 70% 10%,rgba(34,211,238,.12),transparent 60%),radial-gradient(600px 400px at 20% 80%,rgba(99,102,241,.12),transparent 60%);"></div>
  <div class="relative max-w-6xl mx-auto px-4 sm:px-6 grid lg:grid-cols-2 gap-12 items-center">
    <div>
      <div class="flex flex-wrap gap-2 mb-6">
        <span class="badge text-amber-300 border-amber-400/30 bg-amber-400/10">⚠ Preliminary · Pending Review</span>
        <span class="badge text-emerald-300 border-emerald-400/30 bg-emerald-400/10">Apache-2.0</span>
        <span class="badge text-sky-300 border-sky-400/30 bg-sky-400/10">34/34 Tests</span>
      </div>
      <h1 class="display text-4xl sm:text-5xl lg:text-6xl font-bold leading-tight tracking-tight">
        Open science for <span class="grad-text">time-series anomaly detection.</span>
      </h1>
      <p class="text-slate-400 text-lg mt-6 leading-relaxed">
        A non-parametric, multi-tool ensemble that reconstructs phase-space geometry from raw streams —
        combining <b class="text-slate-200">six orthogonal detectors</b> under an online, label-free
        <b class="text-slate-200">Meta-Judge</b>, with strict zero-lookahead streaming and deterministic execution.
      </p>
      <div class="flex flex-wrap gap-3 mt-8">
        <a href="results/" class="btn-primary rounded-xl px-6 py-3">See the Results</a>
        <a href="https://github.com/mohamedhossammohamed/open-phase-ensemble" target="_blank" class="btn-ghost rounded-xl px-6 py-3">View Source</a>
      </div>
      <div class="grid grid-cols-3 gap-4 mt-10 max-w-md">
        <div><div class="num text-2xl text-cyan-300">6</div><div class="text-xs text-slate-500 mt-1">Orthogonal detectors</div></div>
        <div><div class="num text-2xl text-emerald-300">0.9711</div><div class="text-xs text-slate-500 mt-1">VUS-ROC (CWRU)</div></div>
        <div><div class="num text-2xl text-indigo-300">+0.37</div><div class="text-xs text-slate-500 mt-1">Predictive edge</div></div>
      </div>
    </div>
    <div class="relative">
      <canvas id="hero" class="w-full h-[340px] rounded-2xl glass"></canvas>
      <div class="absolute bottom-4 left-4 mono text-[11px] text-slate-500">flat signal → phase-space shape · live</div>
    </div>
  </div>
</section>

<!-- ============ DISCLAIMER ============ -->
<section class="max-w-6xl mx-auto px-4 sm:px-6 -mt-4 pb-16">
  <div class="callout-warn p-5 sm:p-6 flex gap-4 items-start reveal">
    <div class="text-2xl">🧭</div>
    <div>
      <h3 class="font-semibold text-amber-200">Experimental Research — Pending Independent Review</h3>
      <p class="text-sm text-amber-100/70 mt-1 leading-relaxed">
        This project is provided for research and education. All performance claims are preliminary and self-reported,
        and have not been independently validated or peer-reviewed. Do not use for safety-critical, medical, financial,
        or production decisions without expert review. We publish our own audits and corrections openly.
      </p>
    </div>
  </div>
</section>

<!-- ============ WHAT ============ -->
<section id="what" class="max-w-6xl mx-auto px-4 sm:px-6 py-16">
  <div class="reveal">
    <span class="section-label">01 · Overview</span>
    <h2 class="display text-3xl sm:text-4xl font-bold mt-3">Why this exists</h2>
    <p class="text-slate-400 mt-4 max-w-3xl leading-relaxed">
      Time-series anomaly detection suffers from methodological fragmentation and evaluation pitfalls.
      Flawed metrics inflate random guessers to &gt;90%. Heavily parameterized deep models stay opaque and expensive.
      <span class="text-slate-200">open-phase-ensemble</span> takes a different, fully inspectable path.
    </p>
  </div>
  <div class="grid sm:grid-cols-2 lg:grid-cols-3 gap-5 mt-10">
    <div class="card p-6 reveal"><div class="text-2xl mb-3">🪢</div><h3 class="font-semibold">Non-parametric foundation</h3><p class="text-sm text-slate-400 mt-2 leading-relaxed">Reconstructs phase-space manifolds via Takens' delay embedding instead of training millions of parameters.</p></div>
    <div class="card p-6 reveal"><div class="text-2xl mb-3">🔋</div><h3 class="font-semibold">Orthogonal 6-detector battery</h3><p class="text-sm text-slate-400 mt-2 leading-relaxed">Phase-space prediction, covariance geometry, subsequence motifs, subspace isolation, linear autoregression, and MSE reconstruction.</p></div>
    <div class="card p-6 reveal"><div class="text-2xl mb-3">⚖️</div><h3 class="font-semibold">Online Meta-Judge</h3><p class="text-sm text-slate-400 mt-2 leading-relaxed">Hedge multiplicative-weights dynamically reweights detectors in real time — no ground-truth labels required.</p></div>
    <div class="card p-6 reveal"><div class="text-2xl mb-3">🔒</div><h3 class="font-semibold">Zero-lookahead invariant</h3><p class="text-sm text-slate-400 mt-2 leading-relaxed">Strict element-by-element streaming. No future data is ever exposed during scoring.</p></div>
    <div class="card p-6 reveal"><div class="text-2xl mb-3">🧪</div><h3 class="font-semibold">Methodological rigor</h3><p class="text-sm text-slate-400 mt-2 leading-relaxed">Standard VUS-ROC/PR with label-only buffering, tested against IAAFT phase-randomized surrogate nulls.</p></div>
    <div class="card p-6 reveal"><div class="text-2xl mb-3">🧾</div><h3 class="font-semibold">Self-audited honesty</h3><p class="text-sm text-slate-400 mt-2 leading-relaxed">We caught and corrected our own metric inflation, renamed over-claimed detectors, and documented every limitation.</p></div>
  </div>
</section>

<div class="divider max-w-6xl mx-auto"></div>

<!-- ============ ARCHITECTURE ============ -->
<section id="architecture" class="max-w-6xl mx-auto px-4 sm:px-6 py-16">
  <div class="reveal">
    <span class="section-label">02 · Architecture</span>
    <h2 class="display text-3xl sm:text-4xl font-bold mt-3">A transparent pipeline</h2>
    <p class="text-slate-400 mt-4 max-w-3xl leading-relaxed">Every matrix operation, weight update, and gating transition is fully inspectable. No black boxes.</p>
  </div>
  <div class="card p-6 sm:p-8 mt-10 overflow-x-auto reveal">
    <svg viewBox="0 0 960 420" class="w-full min-w-[760px]" font-family="JetBrains Mono, monospace">
      <defs>
        <marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#64748b"/></marker>
      </defs>
      <!-- Module 1 -->
      <rect x="20" y="170" width="150" height="80" rx="12" fill="rgba(34,211,238,.08)" stroke="#22d3ee"/>
      <text x="95" y="200" text-anchor="middle" fill="#7dd3fc" font-size="13" font-weight="700">Ingestion</text>
      <text x="95" y="222" text-anchor="middle" fill="#94a3b8" font-size="10">StreamBuffer</text>
      <text x="95" y="238" text-anchor="middle" fill="#94a3b8" font-size="10">Median / MAD</text>
      <!-- Module 2 -->
      <rect x="200" y="170" width="150" height="80" rx="12" fill="rgba(99,102,241,.08)" stroke="#818cf8"/>
      <text x="275" y="200" text-anchor="middle" fill="#a5b4fc" font-size="13" font-weight="700">Representation</text>
      <text x="275" y="222" text-anchor="middle" fill="#94a3b8" font-size="10">Takens → JL → HNSW</text>
      <text x="275" y="238" text-anchor="middle" fill="#94a3b8" font-size="10">feature vector Z_t</text>
      <!-- Module 3 battery -->
      <rect x="380" y="40" width="220" height="340" rx="14" fill="rgba(255,255,255,.02)" stroke="rgba(148,163,184,.3)" stroke-dasharray="5 5"/>
      <text x="490" y="66" text-anchor="middle" fill="#94a3b8" font-size="12" font-weight="700">6-Detector Battery</text>
      <g font-size="10.5">
        <rect x="400" y="80"  width="180" height="42" rx="8" fill="rgba(52,211,153,.08)" stroke="#34d399"/><text x="490" y="106" text-anchor="middle" fill="#6ee7b7">Simplex Projection (EDM)</text>
        <rect x="400" y="132" width="180" height="42" rx="8" fill="rgba(52,211,153,.08)" stroke="#34d399"/><text x="490" y="158" text-anchor="middle" fill="#6ee7b7">Ledoit-Wolf Mahalanobis</text>
        <rect x="400" y="184" width="180" height="42" rx="8" fill="rgba(52,211,153,.08)" stroke="#34d399"/><text x="490" y="210" text-anchor="middle" fill="#6ee7b7">STOMP Matrix Profile</text>
        <rect x="400" y="236" width="180" height="42" rx="8" fill="rgba(52,211,153,.08)" stroke="#34d399"/><text x="490" y="262" text-anchor="middle" fill="#6ee7b7">Isolation Forest</text>
        <rect x="400" y="288" width="180" height="42" rx="8" fill="rgba(52,211,153,.08)" stroke="#34d399"/><text x="490" y="314" text-anchor="middle" fill="#6ee7b7">AR Ridge Filter</text>
        <rect x="400" y="340" width="180" height="28" rx="8" fill="rgba(52,211,153,.08)" stroke="#34d399"/><text x="490" y="358" text-anchor="middle" fill="#6ee7b7">MSE Transformer AE</text>
      </g>
      <!-- Module 4-5 -->
      <rect x="630" y="150" width="150" height="120" rx="12" fill="rgba(245,158,11,.08)" stroke="#f59e0b"/>
      <text x="705" y="185" text-anchor="middle" fill="#fcd34d" font-size="13" font-weight="700">Meta-Judge</text>
      <text x="705" y="207" text-anchor="middle" fill="#94a3b8" font-size="10">Hedge weights</text>
      <text x="705" y="223" text-anchor="middle" fill="#94a3b8" font-size="10">Pearson loss ↺</text>
      <text x="705" y="245" text-anchor="middle" fill="#fcd34d" font-size="11">fused A_t</text>
      <!-- Module 6 -->
      <rect x="810" y="170" width="130" height="80" rx="12" fill="rgba(248,113,113,.08)" stroke="#f87171"/>
      <text x="875" y="200" text-anchor="middle" fill="#fca5a5" font-size="13" font-weight="700">CUSUM</text>
      <text x="875" y="222" text-anchor="middle" fill="#94a3b8" font-size="10">adapt / freeze / reset</text>
      <!-- arrows -->
      <g stroke="#64748b" stroke-width="1.6" fill="none" marker-end="url(#ar)">
        <path d="M170,210 L200,210"/>
        <path d="M350,210 L380,210"/>
        <path d="M600,210 L630,210"/>
        <path d="M780,210 L810,210"/>
      </g>
    </svg>
  </div>
</section>

<div class="divider max-w-6xl mx-auto"></div>

<!-- ============ RESULTS ============ -->
<section id="results" class="max-w-6xl mx-auto px-4 sm:px-6 py-16">
  <div class="reveal">
    <span class="section-label">03 · Results</span>
    <h2 class="display text-3xl sm:text-4xl font-bold mt-3">Preliminary benchmarks</h2>
    <p class="text-slate-400 mt-4 max-w-3xl leading-relaxed">
      Evaluated with standard VUS-ROC/PR (label-only range buffering). A positive predictive edge over the IAAFT
      surrogate null confirms detection stems from genuine non-linear dynamics, not linear autocorrelation.
    </p>
  </div>
  <div class="card mt-10 overflow-x-auto reveal">
    <table>
      <thead><tr><th>Dataset</th><th>N</th><th>VUS-ROC</th><th>VUS-PR</th><th>IAAFT Null</th><th>Edge (Δ)</th><th></th></tr></thead>
      <tbody>
        <tr>
          <td class="font-medium">PhysioNet MIT-BIH <span class="mono text-slate-500">(rec 100)</span></td>
          <td class="mono text-slate-400">5,143</td>
          <td class="num text-cyan-300 text-lg">0.8592</td>
          <td class="mono text-slate-400">0.6926</td>
          <td class="mono text-slate-500">~0.49</td>
          <td class="num text-emerald-300">+0.37</td>
          <td><div class="edgebar"><div style="width:74%"></div></div></td>
        </tr>
        <tr>
          <td class="font-medium">CWRU Bearing</td>
          <td class="mono text-slate-400">5,000</td>
          <td class="num text-cyan-300 text-lg">0.9711</td>
          <td class="mono text-slate-400">0.7111</td>
          <td class="mono text-slate-500">~0.61</td>
          <td class="num text-emerald-300">+0.37</td>
          <td><div class="edgebar"><div style="width:97%"></div></div></td>
        </tr>
      </tbody>
    </table>
  </div>
  <div class="flex flex-wrap gap-2 mt-4 reveal">
    <span class="pill bg-amber-400/10 text-amber-300 border border-amber-400/30">Preliminary</span>
    <span class="pill bg-amber-400/10 text-amber-300 border border-amber-400/30">Single-run point estimates</span>
    <span class="pill bg-sky-400/10 text-sky-300 border border-sky-400/30">Deterministic system scores</span>
    <span class="pill bg-slate-400/10 text-slate-400 border border-slate-400/30">Stochastic null varies per run</span>
  </div>
</section>

<div class="divider max-w-6xl mx-auto"></div>

<!-- ============ PRINCIPLES ============ -->
<section class="max-w-6xl mx-auto px-4 sm:px-6 py-16">
  <div class="reveal"><span class="section-label">04 · Principles</span><h2 class="display text-3xl sm:text-4xl font-bold mt-3">Core principles</h2></div>
  <div class="grid md:grid-cols-3 gap-5 mt-10">
    <div class="card p-6 reveal"><div class="num text-3xl text-cyan-300">01</div><h3 class="font-semibold mt-3">Transparency</h3><p class="text-sm text-slate-400 mt-2">Every operation is inspectable and auditable. The code is the documentation.</p></div>
    <div class="card p-6 reveal"><div class="num text-3xl text-indigo-300">02</div><h3 class="font-semibold mt-3">Reproducibility</h3><p class="text-sm text-slate-400 mt-2">100% deterministic execution with fixed seeds, verified by SHA-256 hash comparison.</p></div>
    <div class="card p-6 reveal"><div class="num text-3xl text-emerald-300">03</div><h3 class="font-semibold mt-3">Honesty</h3><p class="text-sm text-slate-400 mt-2">Corrected metrics, disclosed limitations, no inflated claims. Truth over optics.</p></div>
  </div>
</section>

<div class="divider max-w-6xl mx-auto"></div>

<!-- ============ LIMITATIONS ============ -->
<section id="limitations" class="max-w-6xl mx-auto px-4 sm:px-6 py-16">
  <div class="reveal"><span class="section-label">05 · Integrity</span><h2 class="display text-3xl sm:text-4xl font-bold mt-3">Known limitations & audit findings</h2>
  <p class="text-slate-400 mt-4 max-w-3xl">We publish our own corrections. This is what scientific integrity looks like in practice.</p></div>
  <div class="grid md:grid-cols-2 gap-5 mt-10">
    <div class="card p-6 reveal"><h3 class="font-semibold text-amber-200">Metric correction applied</h3><p class="text-sm text-slate-400 mt-2 leading-relaxed">An internal audit found an earlier <span class="mono">vus.py</span> buffered both labels <i>and</i> scores, inflating VUS-ROC by +0.06–0.12. Fixed to buffer labels only. All published numbers reflect the corrected evaluation.</p></div>
    <div class="card p-6 reveal"><h3 class="font-semibold text-amber-200">Detector naming aligned</h3><p class="text-sm text-slate-400 mt-2 leading-relaxed">Renamed <span class="mono">ARFilterDetector</span> (online AR ridge filter, not full SARIMA) and <span class="mono">MSETransformerAutoencoder</span> (MSE reconstruction, not association discrepancy) to match their true math.</p></div>
    <div class="card p-6 reveal"><h3 class="font-semibold text-amber-200">Reference comparison not valid</h3><p class="text-sm text-slate-400 mt-2 leading-relaxed">Direct comparison to the closed-source <span class="mono">phase_space_matcher</span> (83.96–86.02%) is scientifically invalid — metric mismatch (VUS-ROC vs PA-F1), evaluation-length differences, and an unverifiable protocol. We report our numbers independently, without claiming superiority.</p></div>
    <div class="card p-6 reveal"><h3 class="font-semibold text-amber-200">Single-run estimates</h3><p class="text-sm text-slate-400 mt-2 leading-relaxed">Results are single deterministic runs. Multi-seed 95% confidence intervals are planned for a future release.</p></div>
  </div>
</section>

<div class="divider max-w-6xl mx-auto"></div>

<!-- ============ QUICKSTART ============ -->
<section id="quickstart" class="max-w-6xl mx-auto px-4 sm:px-6 py-16">
  <div class="reveal"><span class="section-label">06 · Quickstart</span><h2 class="display text-3xl sm:text-4xl font-bold mt-3">Run it in minutes</h2></div>
  <div class="grid lg:grid-cols-2 gap-5 mt-10">
    <div class="reveal"><pre>git clone https://github.com/mohamedhossammohamed/open-phase-ensemble.git
cd open-phase-ensemble
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

# Full test suite (34/34 passing)
PYTHONPATH=src pytest tests/ -v

# Benchmark evaluation
PYTHONPATH=src python scripts/run_benchmark.py</pre></div>
    <div class="reveal"><pre>from tsad.pipeline import TSADPipeline

pipeline = TSADPipeline(tau=2, d=8)

for x in [0.1, 0.2, 0.15, 0.18, 8.5, 0.12]:
    A_t, v_hat = pipeline.step(x)
    print(f"x={x:5.2f}  A_t={A_t:.4f}  forecast={v_hat:.4f}")</pre></div>
  </div>
</section>

<!-- ============ FOOTER ============ -->
<footer class="border-t border-white/5 mt-8">
  <div class="max-w-6xl mx-auto px-4 sm:px-6 py-12 grid md:grid-cols-3 gap-10">
    <div>
      <div class="flex items-center gap-3 mb-4">
        <img src="https://image.qwenlm.ai/public_source/10cec850-dab4-4cc6-9508-a52b88f4dc44/1f5677168-b4c7-423e-9c63-442c93d53bcd.png" class="h-9 w-9 rounded-lg object-cover" alt="logo"/>
        <span class="display font-bold">open-phase-ensemble</span>
      </div>
      <p class="text-sm text-slate-500 leading-relaxed">Open-source, non-parametric, multi-tool ensemble for streaming time-series anomaly detection and forecasting.</p>
    </div>
    <div>
      <h4 class="font-semibold mb-3 text-sm">Resources</h4>
      <ul class="space-y-2 text-sm text-slate-400">
        <li><a class="hover:text-cyan-300" href="https://github.com/mohamedhossammohamed/open-phase-ensemble">GitHub Repository</a></li>
        <li><a class="hover:text-cyan-300" href="results/">Benchmarks</a></li>
        <li><a class="hover:text-cyan-300" href="disclaimer/">Audit & Limitations</a></li>
        <li><a class="hover:text-cyan-300" href="developer/">Quickstart</a></li>
      </ul>
    </div>
    <div>
      <h4 class="font-semibold mb-3 text-sm">Cite</h4>
      <pre class="text-[11px]">@software{open_phase_ensemble2026,
  title  = {open-phase-ensemble: Non-Parametric
            Multi-Tool Ensemble for Time-Series
            Anomaly Detection},
  year   = {2026},
  url    = {github.com/mohamedhossammohamed/
            open-phase-ensemble}
}</pre>
    </div>
  </div>
  <div class="text-center text-xs text-slate-600 pb-8">Licensed under Apache-2.0 · Built openly, audited honestly.</div>
</footer>

<script>
// Reveal on scroll
const io=new IntersectionObserver(es=>es.forEach(e=>{if(e.isIntersecting){e.target.classList.add('in');io.unobserve(e.target);}}),{threshold:.12});
document.querySelectorAll('.reveal').forEach(el=>io.observe(el));

// Hero: flat signal -> phase-space shape
const cv=document.getElementById('hero');
function fit(){const d=devicePixelRatio||1,r=cv.getBoundingClientRect();cv.width=r.width*d;cv.height=r.height*d;const c=cv.getContext('2d');c.setTransform(d,0,0,d,0,0);return{c,w:r.width,h:r.height};}
let F=fit();addEventListener('resize',()=>F=fit());
let buf=[],t=0;
function heart(x){const p=(((x%60)/60)+1)%1;const g=(a,c,w)=>{let d=Math.abs(a-c);d=Math.min(d,1-d);return Math.exp(-d*d/(2*w*w));};
 return .12*g(p,.18,.025)-.18*g(p,.30,.010)+g(p,.33,.012)-.25*g(p,.36,.010)+.30*g(p,.62,.045);}
function loop(){
  const {c,w,h}=F;c.clearRect(0,0,w,h);
  for(let i=0;i<2;i++){buf.push(heart(t));t++;if(buf.length>1400)buf.shift();}
  const tau=14,n=buf.length;
  let mn=1e9,mx=-1e9;for(let i=Math.max(0,n-500);i<n;i++){mn=Math.min(mn,buf[i]);mx=Math.max(mx,buf[i]);}
  const pad=(mx-mn)*.14+1e-6;mn-=pad;mx+=pad;
  // left: flat signal
  const lw=w*.42;
  c.strokeStyle='rgba(148,211,253,.7)';c.lineWidth=1.4;c.beginPath();
  const L=Math.min(240,n);
  for(let i=0;i<L;i++){const x=8+(lw-16)*(i/(L-1));const y=h-((buf[n-L+i]-mn)/(mx-mn))*h;i?c.lineTo(x,y):c.moveTo(x,y);}
  c.stroke();
  c.fillStyle='#64748b';c.font='10px JetBrains Mono';c.fillText('raw 1-D stream',10,16);
  // right: phase space
  const ox=lw+16,iw=w-ox-12,ih=h-24;
  c.strokeStyle='rgba(255,255,255,.1)';c.strokeRect(ox,12,iw,ih);
  const px=v=>ox+((v-mn)/(mx-mn))*iw,py=v=>12+ih-((v-mn)/(mx-mn))*ih;
  const cnt=Math.min(420,n-tau);
  c.strokeStyle='rgba(34,211,238,.25)';c.lineWidth=1.2;c.beginPath();
  for(let k=0;k<cnt;k++){const i=n-cnt+k;const X=px(buf[i-tau]),Y=py(buf[i]);k?c.lineTo(X,Y):c.moveTo(X,Y);}
  c.stroke();
  const i2=n-1;
  c.save();c.shadowColor='#22d3ee';c.shadowBlur=12;c.fillStyle='#a5f3fc';
  c.beginPath();c.arc(px(buf[i2-tau]),py(buf[i2]),4,0,7);c.fill();c.restore();
  c.fillStyle='#64748b';c.fillText('phase-space shape',ox+8,26);
  requestAnimationFrame(loop);
}
loop();
</script>
