"""seed.py — guardrail 1: fixed, reproducible run seed."""
import os
import random

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"[seed] global seed set to {seed}")
    # Deliberately NOT calling torch.use_deterministic_algorithms(True): the
    # fused attention kernels in base_model/mmdit.py have no deterministic
    # backend and would either crash or silently fall back to much slower
    # kernels. This fixes model init noise, dropout, sampling noise, and
    # DataLoader worker RNG -- not bit-exact reproducibility across hardware.