import torch


def get_pilot_indices(n_subcarriers: int, n_pilot_subcarriers: int) -> torch.Tensor:
    step = n_subcarriers // n_pilot_subcarriers
    return torch.arange(0, n_subcarriers, step)[:n_pilot_subcarriers]


def get_data_indices(n_subcarriers: int, pilot_indices: torch.Tensor) -> torch.Tensor:
    mask = torch.ones(n_subcarriers, dtype=torch.bool, device=pilot_indices.device)
    mask[pilot_indices] = False
    return torch.arange(n_subcarriers, device=pilot_indices.device)[mask]


def build_ofdm_grid(
    data_symbols: torch.Tensor,
    n_subcarriers: int,
    n_ofdm_symbols: int,
    pilot_indices: torch.Tensor,
    pilot_symbol: complex = 1.0 + 0.0j,
) -> torch.Tensor:
    grid = torch.zeros((n_ofdm_symbols, n_subcarriers), dtype=torch.complex64, device=data_symbols.device)
    data_indices = get_data_indices(n_subcarriers, pilot_indices)

    n_data_per_symbol = data_indices.numel()
    expected = n_ofdm_symbols * n_data_per_symbol
    if data_symbols.numel() != expected:
        raise ValueError(f"Expected {expected} data symbols, got {data_symbols.numel()}")

    grid[:, pilot_indices] = torch.tensor(pilot_symbol, dtype=torch.complex64, device=data_symbols.device)
    grid[:, data_indices] = data_symbols.view(n_ofdm_symbols, n_data_per_symbol)
    return grid


def ofdm_modulate(grid: torch.Tensor, cp_length: int) -> torch.Tensor:
    time_no_cp = torch.fft.ifft(grid, dim=-1)
    cp = time_no_cp[:, -cp_length:]
    return torch.cat([cp, time_no_cp], dim=-1)


def ofdm_demodulate(rx_time: torch.Tensor, n_subcarriers: int, cp_length: int) -> torch.Tensor:
    if rx_time.shape[-1] != n_subcarriers + cp_length:
        raise ValueError("Invalid OFDM symbol length")
    no_cp = rx_time[:, cp_length:]
    return torch.fft.fft(no_cp, dim=-1)
