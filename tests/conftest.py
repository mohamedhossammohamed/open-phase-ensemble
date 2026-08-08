import os
import random
import numpy as np
import pytest

from tsad.config import SEED

@pytest.fixture(autouse=True)
def set_deterministic_seeds():
    """Ensure strict execution determinism across all tests."""
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    
    try:
        import torch
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
