# Developer Documentation

!!! note "Preliminary Research — Pending Independent Review"
    This project is experimental. All claims are preliminary and self-reported. See the [full disclaimer](disclaimer.md).

---

## 💻 Environment Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/mohamedhossammohamed/open-phase-ensemble.git
   cd open-phase-ensemble
   ```

2. **Create & Activate Virtual Environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -e .
   ```

3. **Install Documentation Tools**:
   ```bash
   pip install mkdocs pymdown-extensions
   ```

---

## 🧩 Adding a New Detector Module

All detectors implement the `DetectorABC` contract defined in `src/tsad/detectors/base.py`:

```python
from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple

class DetectorABC(ABC):
    @abstractmethod
    def score(self, Z_t: np.ndarray, v_t: float) -> Tuple[float, float]:
        """
        Calculates instantaneous anomaly score s_t in [0, 1] and forward forecast v_hat_{t+h}.
        """
        pass

    @abstractmethod
    def update(self, v_true: float):
        """
        Updates internal model state with newly observed true scalar v_t.
        """
        pass
```

### Steps to Register a New Detector:
1. Create a new module file in `src/tsad/detectors/your_detector.py` inheriting from `DetectorABC`.
2. Ensure your detector output score $s_t$ is calibrated using 3-sigma or robust median statistics so normal baseline output is $\approx 0.0$.
3. Register the new detector class in `src/tsad/pipeline.py` inside `self.detectors`.
4. Update `K_DETECTORS` constant in `src/tsad/config.py`.
5. Add unit tests in `tests/unit/test_detectors.py`.

---

## 🧪 Running Test Suites

```bash
# Run unit tests only (29 tests)
PYTHONPATH=src pytest tests/unit/ -v

# Run integration tests (2 tests)
PYTHONPATH=src pytest tests/integration/ -v

# Run end-to-end tests (3 tests)
PYTHONPATH=src pytest tests/e2e/ -v

# Run full test suite (34 tests)
PYTHONPATH=src pytest tests/ -v
```

---

## 🌐 Local Documentation Server

```bash
mkdocs serve
```

Open `http://127.0.0.1:8000/` in your browser to view live documentation changes.
