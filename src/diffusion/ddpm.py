import torch
import torch.nn.functional as F

from src.diffusion.noise_schedule import NoiseSchedule


class DDPM:
    def __init__(self, model, schedule: NoiseSchedule, device: torch.device):
        self.model = model
        self.schedule = schedule.to(device)
        self.device = device

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(x0)
        a_bar = self.schedule.alpha_bars[t].unsqueeze(1)
        xt = torch.sqrt(a_bar) * x0 + torch.sqrt(1.0 - a_bar) * noise
        return xt, noise

    def p_losses(self, x0: torch.Tensor, snr_db: torch.Tensor) -> torch.Tensor:
        batch = x0.shape[0]
        t = torch.randint(0, self.schedule.n_timesteps, (batch,), device=self.device)
        xt, noise = self.q_sample(x0, t)
        noise_pred = self.model(xt, t, snr_db)
        return F.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def p_sample(self, xt: torch.Tensor, t: int, snr_db: torch.Tensor) -> torch.Tensor:
        tt = torch.full((xt.shape[0],), t, device=self.device, dtype=torch.long)
        beta_t = self.schedule.betas[tt].unsqueeze(1)
        alpha_t = self.schedule.alphas[tt].unsqueeze(1)
        a_bar_t = self.schedule.alpha_bars[tt].unsqueeze(1)

        noise_pred = self.model(xt, tt, snr_db)
        mean = (xt - (beta_t / torch.sqrt(1.0 - a_bar_t)) * noise_pred) / torch.sqrt(alpha_t)

        if t > 0:
            z = torch.randn_like(xt)
            sigma = torch.sqrt(beta_t)
            return mean + sigma * z
        return mean

    @torch.no_grad()
    def denoise_from_equalized(self, x_eq: torch.Tensor, snr_db: torch.Tensor) -> torch.Tensor:
        sigma2_snr = (1.0 / (10.0 ** (snr_db / 10.0))).squeeze(1)
        sigma2_emp = self.estimate_residual_variance(x_eq)
        sigma2 = torch.maximum(sigma2_snr, sigma2_emp)
        t_start = self.schedule.timestep_from_sigma2(sigma2)

        x = x_eq.clone()
        t0 = int(torch.max(t_start).item())
        for t in range(t0, -1, -1):
            active = t_start >= t
            if not torch.any(active):
                continue
            x_active = self.p_sample(x[active], t, snr_db[active])
            x[active] = x_active
        return x

    @staticmethod
    def estimate_residual_variance(x_eq: torch.Tensor) -> torch.Tensor:
        # Nearest hard decision to normalized 16-QAM on each axis.
        levels = torch.tensor([-3.0, -1.0, 1.0, 3.0], device=x_eq.device, dtype=x_eq.dtype) / (10.0**0.5)
        xr = x_eq[:, 0].unsqueeze(1)
        xi = x_eq[:, 1].unsqueeze(1)
        i_hat = levels[torch.argmin(torch.abs(xr - levels.unsqueeze(0)), dim=1)]
        q_hat = levels[torch.argmin(torch.abs(xi - levels.unsqueeze(0)), dim=1)]
        x_hd = torch.stack([i_hat, q_hat], dim=1)

        resid = x_eq - x_hd
        sigma2 = torch.sum(resid * resid, dim=1).clamp_min(1e-6)
        return sigma2
