import torch


def zf_equalize(y: torch.Tensor, h_hat: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return y / (h_hat + eps)


def mmse_equalize(y: torch.Tensor, h_hat: torch.Tensor, noise_power: float) -> torch.Tensor:
    denom = h_hat.abs().pow(2) + noise_power
    return torch.conj(h_hat) * y / denom
