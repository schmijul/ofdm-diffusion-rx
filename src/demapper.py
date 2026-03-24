import torch

# Gray-coded 16-QAM levels on each axis.
_AXIS_LEVELS = torch.tensor([-3.0, -1.0, 1.0, 3.0])


def bits_to_qam16(bits: torch.Tensor) -> torch.Tensor:
    bits = bits.view(-1, 4).float()
    i_idx = (bits[:, 0] * 2 + bits[:, 1]).long()
    q_idx = (bits[:, 2] * 2 + bits[:, 3]).long()

    level_map = torch.tensor([-3.0, -1.0, 1.0, 3.0], device=bits.device)
    i = level_map[i_idx]
    q = level_map[q_idx]

    # Normalize average symbol energy to 1.
    return (i + 1j * q) / (10.0**0.5)


def qam16_to_bits(symbols: torch.Tensor) -> torch.Tensor:
    s = symbols * (10.0**0.5)
    i = s.real
    q = s.imag

    def quantize_axis(x: torch.Tensor) -> torch.Tensor:
        boundaries = torch.tensor([-2.0, 0.0, 2.0], device=x.device)
        idx = torch.zeros_like(x, dtype=torch.long)
        idx = idx + (x > boundaries[0]).long()
        idx = idx + (x > boundaries[1]).long()
        idx = idx + (x > boundaries[2]).long()
        return idx.clamp(0, 3)

    i_idx = quantize_axis(i)
    q_idx = quantize_axis(q)

    bits = torch.empty((symbols.numel(), 4), dtype=torch.long, device=symbols.device)
    bits[:, 0] = (i_idx // 2)
    bits[:, 1] = (i_idx % 2)
    bits[:, 2] = (q_idx // 2)
    bits[:, 3] = (q_idx % 2)
    return bits.reshape(-1)
