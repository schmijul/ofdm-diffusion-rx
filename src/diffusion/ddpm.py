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
        snr_linear = 10.0 ** (snr_db / 10.0)
        sigma2 = (1.0 / snr_linear).squeeze(1)
        t_start = self.schedule.timestep_from_sigma2(sigma2)

        x = x_eq
        t0 = int(torch.max(t_start).item())
        for t in range(t0, -1, -1):
            x = self.p_sample(x, t, snr_db)
        return x
