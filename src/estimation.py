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
