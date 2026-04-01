import torch
import torch.nn.functional as F

from src.diffusion.noise_schedule import NoiseSchedule


class DDPM:
    def __init__(
        self,
        model,
        schedule: NoiseSchedule,
        device: torch.device,
        inference_steps: int | None = None,
        prior_context: torch.Tensor | None = None,
        bit_loss_weight: float = 0.0,
        bit_logit_temperature: float = 24.0,
    ):
        self.model = model
        self.schedule = schedule.to(device)
        self.device = device
        self.inference_steps = int(inference_steps) if inference_steps is not None else self.schedule.n_timesteps
        self.prior_context = prior_context.to(device=device, dtype=torch.float32) if prior_context is not None else None
        self.bit_loss_weight = float(bit_loss_weight)
        self.bit_logit_temperature = float(bit_logit_temperature)

    def _expand_context(self, context: torch.Tensor | None, batch_size: int, dtype: torch.dtype) -> torch.Tensor | None:
        ctx = context if context is not None else self.prior_context
        if ctx is None:
            return None
        if ctx.dim() == 1:
            ctx = ctx.unsqueeze(0).expand(batch_size, -1)
        elif ctx.dim() == 2 and ctx.shape[0] == 1:
            ctx = ctx.expand(batch_size, -1)
        elif ctx.dim() != 2 or ctx.shape[0] != batch_size:
            raise ValueError(f"Invalid context shape {tuple(ctx.shape)} for batch size {batch_size}")
        return ctx.to(device=self.device, dtype=dtype)

    @staticmethod
    def _qam16_target_bits_from_clean(x0: torch.Tensor) -> torch.Tensor:
        x = x0 * (10.0**0.5)
        boundaries = x.new_tensor([-2.0, 0.0, 2.0])

        def _idx(axis: torch.Tensor) -> torch.Tensor:
            idx = torch.zeros_like(axis, dtype=torch.long)
            idx = idx + (axis > boundaries[0]).long()
            idx = idx + (axis > boundaries[1]).long()
            idx = idx + (axis > boundaries[2]).long()
            return idx.clamp(0, 3)

        i_idx = _idx(x[:, 0])
        q_idx = _idx(x[:, 1])
        bits = torch.empty((x0.shape[0], 4), device=x0.device, dtype=x0.dtype)
        bits[:, 0] = (i_idx // 2).to(x0.dtype)
        bits[:, 1] = (i_idx % 2).to(x0.dtype)
        bits[:, 2] = (q_idx // 2).to(x0.dtype)
        bits[:, 3] = (q_idx % 2).to(x0.dtype)
        return bits

    def _qam16_soft_bit_logits(self, x: torch.Tensor) -> torch.Tensor:
        levels = x.new_tensor([-3.0, -1.0, 1.0, 3.0]) / (10.0**0.5)
        axis_bits = x.new_tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        temp = max(self.bit_logit_temperature, 1e-3)

        logits_out = []
        for axis in (0, 1):
            ax = x[:, axis].unsqueeze(1)
            scores = -temp * (ax - levels.unsqueeze(0)).pow(2)
            probs = torch.softmax(scores, dim=1)
            p_bits = torch.matmul(probs, axis_bits).clamp(1e-5, 1.0 - 1e-5)
            logits_out.append(torch.log(p_bits / (1.0 - p_bits)))
        return torch.cat(logits_out, dim=1)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(x0)
        a_bar = self.schedule.alpha_bars[t].unsqueeze(1)
        xt = torch.sqrt(a_bar) * x0 + torch.sqrt(1.0 - a_bar) * noise
        return xt, noise

    def p_losses(self, x0: torch.Tensor, snr_db: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        batch = x0.shape[0]
        t = torch.randint(0, self.schedule.n_timesteps, (batch,), device=self.device)
        xt, noise = self.q_sample(x0, t)
        ctx = self._expand_context(context, batch, x0.dtype)
        noise_pred = self.model(xt, t, snr_db, ctx)
        mse = F.mse_loss(noise_pred, noise)
        if self.bit_loss_weight <= 0.0:
            return mse

        a_bar = self.schedule.alpha_bars[t].unsqueeze(1)
        x0_pred = (xt - torch.sqrt(1.0 - a_bar) * noise_pred) / torch.sqrt(a_bar)
        target_bits = self._qam16_target_bits_from_clean(x0)
        pred_logits = self._qam16_soft_bit_logits(x0_pred)
        bit_loss = F.binary_cross_entropy_with_logits(pred_logits, target_bits)
        return mse + self.bit_loss_weight * bit_loss

    @torch.no_grad()
    def p_sample(self, xt: torch.Tensor, t: int, snr_db: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        tt = torch.full((xt.shape[0],), t, device=self.device, dtype=torch.long)
        beta_t = self.schedule.betas[tt].unsqueeze(1)
        alpha_t = self.schedule.alphas[tt].unsqueeze(1)
        a_bar_t = self.schedule.alpha_bars[tt].unsqueeze(1)
        ctx = self._expand_context(context, xt.shape[0], xt.dtype)

        noise_pred = self.model(xt, tt, snr_db, ctx)
        mean = (xt - (beta_t / torch.sqrt(1.0 - a_bar_t)) * noise_pred) / torch.sqrt(alpha_t)

        if t > 0:
            z = torch.randn_like(xt)
            sigma = torch.sqrt(beta_t)
            return mean + sigma * z
        return mean

    @torch.no_grad()
    def denoise_from_equalized(
        self, x_eq: torch.Tensor, snr_db: torch.Tensor, context: torch.Tensor | None = None
    ) -> torch.Tensor:
        sigma2_snr = (1.0 / (10.0 ** (snr_db / 10.0))).squeeze(1)
        sigma2_emp = self.estimate_residual_variance(x_eq)
        sigma2 = torch.maximum(sigma2_snr, sigma2_emp)
        t_start = self.schedule.timestep_from_sigma2(sigma2)

        x = x_eq.clone()
        t0 = int(torch.max(t_start).item())
        steps = min(max(self.inference_steps, 1), t0 + 1)
        t_seq = torch.linspace(t0, 0, steps=steps, device=x.device).round().to(torch.long)
        t_seq = torch.unique_consecutive(t_seq)
        ctx = self._expand_context(context, x.shape[0], x.dtype)

        for t_tensor in t_seq:
            t = int(t_tensor.item())
            active = t_start >= t
            if not torch.any(active):
                continue
            ctx_active = ctx[active] if ctx is not None else None
            x_active = self.p_sample(x[active], t, snr_db[active], ctx_active)
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
