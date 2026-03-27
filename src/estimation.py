import torch


def ls_channel_estimate(
    y_grid: torch.Tensor,
    pilot_indices: torch.Tensor,
    pilot_symbol: complex = 1.0 + 0.0j,
) -> torch.Tensor:
    x_p = torch.tensor(pilot_symbol, dtype=torch.complex64, device=y_grid.device)
    h_p = y_grid[:, pilot_indices] / x_p
    return h_p


def interpolate_channel(
    h_pilots: torch.Tensor,
    pilot_indices: torch.Tensor,
    n_subcarriers: int,
) -> torch.Tensor:
    n_sym = h_pilots.shape[0]
    h_full = torch.zeros((n_sym, n_subcarriers), dtype=torch.complex64, device=h_pilots.device)

    xp = pilot_indices.float()
    x = torch.arange(n_subcarriers, device=h_pilots.device).float()
    for s in range(n_sym):
        h_real = h_pilots[s].real
        h_imag = h_pilots[s].imag

        # OFDM channel response is periodic over subcarriers; use cyclic linear interpolation.
        xp_ext = torch.cat([xp[-1:] - n_subcarriers, xp, xp[:1] + n_subcarriers], dim=0)
        hr_ext = torch.cat([h_real[-1:], h_real, h_real[:1]], dim=0)
        hi_ext = torch.cat([h_imag[-1:], h_imag, h_imag[:1]], dim=0)

        right = torch.searchsorted(xp_ext, x, right=True)
        left = right - 1

        x0 = xp_ext[left]
        x1 = xp_ext[right]
        w = (x - x0) / (x1 - x0)
        hr = (1.0 - w) * hr_ext[left] + w * hr_ext[right]
        hi = (1.0 - w) * hi_ext[left] + w * hi_ext[right]

        h_full[s] = hr + 1j * hi

    return h_full


def dft_project_channel_response(h_freq: torch.Tensor, n_taps_keep: int) -> torch.Tensor:
    if n_taps_keep <= 0:
        raise ValueError("n_taps_keep must be positive")
    n_subcarriers = h_freq.shape[-1]
    if n_taps_keep > n_subcarriers:
        raise ValueError("n_taps_keep cannot exceed n_subcarriers")

    h_time = torch.fft.ifft(h_freq, dim=-1)
    h_time[..., n_taps_keep:] = 0.0
    return torch.fft.fft(h_time, dim=-1).to(torch.complex64)


def estimate_channel_response(
    y_grid: torch.Tensor,
    pilot_indices: torch.Tensor,
    n_subcarriers: int,
    method: str = "linear",
    dft_tap_truncation: int | None = None,
) -> torch.Tensor:
    h_p = ls_channel_estimate(y_grid, pilot_indices)
    h_lin = interpolate_channel(h_p, pilot_indices, n_subcarriers)

    method_name = str(method).lower()
    if method_name == "linear":
        return h_lin
    if method_name in {"dft", "dft_linear", "dft_projection"}:
        if dft_tap_truncation is None:
            raise ValueError("dft_tap_truncation must be set when using dft-based estimation")
        return dft_project_channel_response(h_lin, int(dft_tap_truncation))

    raise ValueError(f"Unknown channel estimation method: {method}")
