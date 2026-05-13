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


def qam16_to_bits_with_priors(
    symbols: torch.Tensor,
    bit_one_probs: list[float] | tuple[float, ...],
    prior_weight: float,
) -> torch.Tensor:
    if prior_weight <= 0.0 or len(bit_one_probs) != 4:
        return qam16_to_bits(symbols).long()

    device = symbols.device
    dtype = symbols.real.dtype
    levels = _AXIS_LEVELS.to(device=device, dtype=dtype) / (10.0**0.5)
    grid_i, grid_q = torch.meshgrid(levels, levels, indexing="ij")
    const = grid_i.reshape(-1) + 1j * grid_q.reshape(-1)

    bits_lut = torch.tensor(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 1, 1],
            [0, 1, 0, 0],
            [0, 1, 0, 1],
            [0, 1, 1, 0],
            [0, 1, 1, 1],
            [1, 0, 0, 0],
            [1, 0, 0, 1],
            [1, 0, 1, 0],
            [1, 0, 1, 1],
            [1, 1, 0, 0],
            [1, 1, 0, 1],
            [1, 1, 1, 0],
            [1, 1, 1, 1],
        ],
        device=device,
        dtype=torch.long,
    )

    probs = torch.tensor(bit_one_probs, device=device, dtype=dtype).clamp(1e-4, 1.0 - 1e-4)
    logp1 = torch.log(probs).unsqueeze(0)
    logp0 = torch.log(1.0 - probs).unsqueeze(0)
    bits_lut_f = bits_lut.to(dtype=dtype)
    log_prior = torch.sum(bits_lut_f * logp1 + (1.0 - bits_lut_f) * logp0, dim=1)

    dist2 = torch.abs(symbols.unsqueeze(1) - const.unsqueeze(0)).pow(2)
    score = dist2 - prior_weight * log_prior.unsqueeze(0)
    best = torch.argmin(score, dim=1)
    return bits_lut[best].reshape(-1).long()
