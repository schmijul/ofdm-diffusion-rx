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

        hr = torch.empty_like(x)
        hi = torch.empty_like(x)

        # Piecewise linear interpolation with boundary clamping.
        for i in range(n_subcarriers):
            xi = x[i]
            if xi <= xp[0]:
                hr[i] = h_real[0]
                hi[i] = h_imag[0]
                continue
            if xi >= xp[-1]:
                hr[i] = h_real[-1]
                hi[i] = h_imag[-1]
                continue

            right = torch.searchsorted(xp, xi, right=False)
            left = right - 1
            x0 = xp[left]
            x1 = xp[right]
            w = (xi - x0) / (x1 - x0)
            hr[i] = (1.0 - w) * h_real[left] + w * h_real[right]
            hi[i] = (1.0 - w) * h_imag[left] + w * h_imag[right]

        h_full[s] = hr + 1j * hi

    return h_full
