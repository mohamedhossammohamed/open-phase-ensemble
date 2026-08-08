# Contributing to open-phase-ensemble

Thank you for your interest in contributing to `open-phase-ensemble`! We welcome scientific reviews, bug reports, feature requests, and code contributions from researchers, engineers, and time-series practitioners.

---

## 🔬 Scientific Honesty & Peer Review
`open-phase-ensemble` prioritizes scientific rigor and transparent open evaluation:
1. **No Overclaiming**: All performance claims must be treated as preliminary and subject to independent reproduction.
2. **Zero-Lookahead Invariant**: All additions to the pipeline must strictly respect the zero-lookahead online streaming constraint.
3. **Execution Determinism**: All stochastic operations must accept a random seed (`SEED=42`).

---

## 🛠️ Development Setup

1. **Clone & Virtual Environment**:
   ```bash
   git clone https://github.com/mohamedhossammohamed/open-phase-ensemble.git
   cd open-phase-ensemble
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -e .[dev]
   ```

2. **Running Tests**:
   ```bash
   PYTHONPATH=src pytest tests/ -v
   ```

3. **Building Documentation**:
   ```bash
   mkdocs serve
   ```

---

## 📝 Pull Request Guidelines
- Include unit tests for all new functions or detectors under `tests/unit/`.
- Ensure all integration and determinism invariants pass before submitting.
- Follow PEP8 styling (enforced via `ruff` / `black` and `mypy`).
