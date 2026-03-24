import torch


def bit_error_rate(bits_tx: torch.Tensor, bits_rx: torch.Tensor) -> float:
    if bits_tx.numel() != bits_rx.numel():
        raise ValueError("Bit vectors must be same length")
    return float((bits_tx != bits_rx).float().mean().item())


def symbol_error_rate(sym_tx: torch.Tensor, sym_rx: torch.Tensor) -> float:
    if sym_tx.numel() != sym_rx.numel():
        raise ValueError("Symbol vectors must be same length")
    return float((sym_tx != sym_rx).float().mean().item())
