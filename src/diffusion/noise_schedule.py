import math

import torch


class NoiseSchedule:
    def __init__(self, n_timesteps: int, beta_start: float, beta_end: float, schedule: str = "linear"):
        self.n_timesteps = n_timesteps
        if schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, n_timesteps)
        elif schedule == "cosine":
            betas = self._cosine_betas(n_timesteps)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        self.betas = betas.clamp(1e-8, 0.999)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    @staticmethod
    def _cosine_betas(n_timesteps: int, s: float = 0.008) -> torch.Tensor:
        t = torch.arange(n_timesteps + 1, dtype=torch.float32)
        f = torch.cos(((t / n_timesteps) + s) / (1 + s) * math.pi / 2.0).pow(2)
        alpha_bars = f / f[0]
        betas = 1.0 - (alpha_bars[1:] / alpha_bars[:-1])
        return betas.clamp(1e-8, 0.999)

    def to(self, device: torch.device) -> "NoiseSchedule":
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)
        return self

    def sigma2_from_t(self, t: torch.Tensor) -> torch.Tensor:
        a_bar = self.alpha_bars[t]
        return (1.0 - a_bar) / a_bar

    def timestep_from_sigma2(self, sigma2: torch.Tensor) -> torch.Tensor:
        target = sigma2.reshape(-1, 1)
        all_sigma2 = ((1.0 - self.alpha_bars) / self.alpha_bars).reshape(1, -1)
        idx = torch.argmin(torch.abs(all_sigma2 - target), dim=1)
        return idx.long()
