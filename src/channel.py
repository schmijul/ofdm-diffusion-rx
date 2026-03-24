import torch


def generate_rayleigh_taps(n_taps: int, exp_decay: float, device: torch.device) -> torch.Tensor:
    power = exp_decay ** torch.arange(n_taps, device=device, dtype=torch.float32)
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


def add_awgn(x: torch.Tensor, snr_db: float) -> tuple[torch.Tensor, float]:
    sig_power = x.abs().pow(2).mean().item()
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = sig_power / snr_linear
    noise_std = (noise_power / 2.0) ** 0.5

    noise = noise_std * (torch.randn_like(x.real) + 1j * torch.randn_like(x.imag))
    return x + noise.to(torch.complex64), float(noise_power)


def channel_frequency_response(taps: torch.Tensor, n_subcarriers: int) -> torch.Tensor:
    h_time = torch.zeros(n_subcarriers, dtype=torch.complex64, device=taps.device)
    h_time[: taps.numel()] = taps
    return torch.fft.fft(h_time)
