import torch


def generate_rayleigh_taps(n_taps: int, exp_decay: float, device: torch.device) -> torch.Tensor:
    power = exp_decay ** torch.arange(n_taps, device=device, dtype=torch.float32)
    power = power / power.sum()
    std = torch.sqrt(power / 2.0)
    real = torch.randn(n_taps, device=device) * std
    imag = torch.randn(n_taps, device=device) * std
    return (real + 1j * imag).to(torch.complex64)


def generate_tdl_a_taps(n_taps: int, device: torch.device) -> torch.Tensor:
    # Compact TDL-A style power profile approximation (first taps only).
    # Relative powers in dB (truncated profile): [0, -2.2, -4.0, -6.0, -8.2, -10.5, -13.5, -16.0]
    base_db = torch.tensor([0.0, -2.2, -4.0, -6.0, -8.2, -10.5, -13.5, -16.0], device=device)
    if n_taps <= base_db.numel():
        p_db = base_db[:n_taps]
    else:
        extra = torch.linspace(-18.0, -24.0, n_taps - base_db.numel(), device=device)
        p_db = torch.cat([base_db, extra], dim=0)

    power = 10.0 ** (p_db / 10.0)
    power = power / power.sum()
    std = torch.sqrt(power / 2.0)
    real = torch.randn(n_taps, device=device) * std
    imag = torch.randn(n_taps, device=device) * std
    return (real + 1j * imag).to(torch.complex64)


def apply_multipath(x: torch.Tensor, taps: torch.Tensor) -> torch.Tensor:
    # x: [n_symbols, n_time]
    n_symbols, n_time = x.shape
    y = torch.zeros_like(x)
    for s in range(n_symbols):
        for k in range(taps.numel()):
            if k == 0:
                y[s] += taps[k] * x[s]
            else:
                y[s, k:] += taps[k] * x[s, :-k]
    return y


def snr_db_to_linear(snr_db: float) -> float:
    return float(10.0 ** (float(snr_db) / 10.0))


def effective_sample_snr_linear(
    snr_db: float,
    snr_definition: str = "sample",
    bits_per_symbol: int = 4,
    data_subcarrier_fraction: float = 1.0,
) -> float:
    if data_subcarrier_fraction <= 0.0 or data_subcarrier_fraction > 1.0:
        raise ValueError("data_subcarrier_fraction must be in (0,1]")
    if bits_per_symbol <= 0:
        raise ValueError("bits_per_symbol must be positive")

    base = snr_db_to_linear(snr_db)
    mode = str(snr_definition).lower()
    if mode in {"sample", "sample_snr"}:
        return base
    if mode in {"esn0", "esn0_db"}:
        return base
    if mode in {"ebn0", "ebn0_db"}:
        return base * float(bits_per_symbol) * float(data_subcarrier_fraction)

    raise ValueError(f"Unsupported snr_definition: {snr_definition}")


def add_awgn(
    x: torch.Tensor,
    snr_db: float,
    snr_definition: str = "sample",
    bits_per_symbol: int = 4,
    data_subcarrier_fraction: float = 1.0,
) -> tuple[torch.Tensor, float]:
    sig_power = x.abs().pow(2).mean().item()
    snr_linear = effective_sample_snr_linear(
        snr_db,
        snr_definition=snr_definition,
        bits_per_symbol=bits_per_symbol,
        data_subcarrier_fraction=data_subcarrier_fraction,
    )
    noise_power = sig_power / snr_linear
    noise_std = (noise_power / 2.0) ** 0.5

    noise = noise_std * (torch.randn_like(x.real) + 1j * torch.randn_like(x.imag))
    return x + noise.to(torch.complex64), float(noise_power)


def channel_frequency_response(taps: torch.Tensor, n_subcarriers: int) -> torch.Tensor:
    h_time = torch.zeros(n_subcarriers, dtype=torch.complex64, device=taps.device)
    h_time[: taps.numel()] = taps
    return torch.fft.fft(h_time)
