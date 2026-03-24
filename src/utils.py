import random
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(path: str = "config/default.yaml") -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(cfg: dict) -> torch.device:
    prefer_cuda = cfg.get("device", {}).get("prefer_cuda", True)
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def complex_to_real(z: torch.Tensor) -> torch.Tensor:
    return torch.stack([z.real, z.imag], dim=-1)


def real_to_complex(x: torch.Tensor) -> torch.Tensor:
    return x[..., 0] + 1j * x[..., 1]
