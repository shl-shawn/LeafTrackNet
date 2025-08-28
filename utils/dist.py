import os
import random
import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_ddp():
    return "LOCAL_RANK" in os.environ


def get_local_rank():
    return int(os.environ.get("LOCAL_RANK", 0))

def is_main_process() -> bool:
    """True for rank-0 or non-DDP runs."""
    if "RANK" not in os.environ:
        return True
    try:
        return int(os.environ.get("RANK", "0")) == 0
    except ValueError:
        return True