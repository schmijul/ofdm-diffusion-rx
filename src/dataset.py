from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from src.classical_receiver import run_classical_frame
from src.utils import complex_to_real


@dataclass
class DatasetTensors:
    x_equalized: torch.Tensor
    x_clean: torch.Tensor
    snr_db: torch.Tensor


class EqualizedSymbolDataset(Dataset):
    def __init__(self, tensors: DatasetTensors):
        self.x_equalized = tensors.x_equalized.float()
        self.x_clean = tensors.x_clean.float()
        self.snr_db = tensors.snr_db.float()

    def __len__(self) -> int:
        return self.x_clean.shape[0]

    def __getitem__(self, idx: int):
        return self.x_equalized[idx], self.x_clean[idx], self.snr_db[idx]


def _sample_snr_uniform(snr_min_db: float, snr_max_db: float) -> float:
    snr = torch.empty(1).uniform_(snr_min_db, snr_max_db)
    return float(snr.item())


def generate_symbol_dataset(
    cfg: dict,
    n_samples: int,
    snr_min_db: float,
    snr_max_db: float,
    method: str = "ls_mmse",
) -> DatasetTensors:
    xeq_list = []
    xclean_list = []
    snr_list = []

    n_collected = 0
    while n_collected < n_samples:
        snr_db = _sample_snr_uniform(snr_min_db, snr_max_db)
        out = run_classical_frame(cfg, snr_db=snr_db, method=method, perfect_csi=False)

        x_eq = complex_to_real(out["equalized_symbols"])
        x_clean = complex_to_real(out["tx_symbols"])
        n_take = min(n_samples - n_collected, x_clean.shape[0])

        xeq_list.append(x_eq[:n_take])
        xclean_list.append(x_clean[:n_take])
        snr_list.append(torch.full((n_take, 1), snr_db))
        n_collected += n_take

    xeq = torch.cat(xeq_list, dim=0)
    xclean = torch.cat(xclean_list, dim=0)
    snr = torch.cat(snr_list, dim=0)

    return DatasetTensors(x_equalized=xeq, x_clean=xclean, snr_db=snr)
